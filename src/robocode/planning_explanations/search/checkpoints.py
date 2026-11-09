"""Generic statement/loop snapshots and replayed edits of live Python state.

This small trusted-code runner deliberately has a one-position input interface.
It is independent of the existing four-argument expression-trace runner. It is
not a security sandbox and never interprets local names as semantic roles.
"""

# Exact built-in types are part of the restricted representation language.
# pylint: disable=unidiomatic-typecheck

import ast
import copy
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from types import CodeType
from typing import Any

from ..instrument import ALLOWED, UnsupportedProgram, is_value
from .selectors import locate, parse
from .tasks import TaskCase

BUILTINS = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "int": int,
    "list": list,
    "dict": dict,
    "abs": abs,
}
METHODS = {"append", "pop", "remove", "clear", "reverse"}


@dataclass
class Snapshot:
    """One concrete configuration; deepcopy preserves aliases within its locals."""

    index: int
    site: str
    occurrence: int
    kind: str
    values: dict
    output: object = None
    before: dict | None = None
    aliases: tuple = ()


@dataclass(frozen=True)
class Edit:
    """A declared local/container edit, never a replacement of the final output."""

    site: str
    occurrence: int
    variable: str
    operation: str
    value: int = 0
    role: str = ""
    node: int | None = None


@dataclass
class Trace:
    """Recorded configurations of one complete original-program invocation."""

    position: int
    output: object
    snapshots: list[Snapshot]
    edits: int = 0
    case: TaskCase | None = None

    @property
    def input(self) -> int | TaskCase:
        """Original immutable invocation, including its task when present."""
        return self.position if self.case is None else self.case


class _Recorder:
    def __init__(self, edit: Edit | None, budget: int):
        self.edit = edit
        self.budget = budget
        self.snapshots: list[Snapshot] = []
        self.visits: Counter[str] = Counter()
        self.edits = 0

    def capture(self, site: str, kind: str, bindings: dict, output=None) -> dict:
        """Snapshot locals and apply a declared live edit at exactly one visit."""
        if len(self.snapshots) >= self.budget:
            raise RuntimeError("Concrete checkpoint budget exhausted")
        values = {k: v for k, v in bindings.items() if not k.startswith("__")}
        if not is_value(values):  # type: ignore[no-untyped-call]
            raise UnsupportedProgram("Snapshots require scalars and acyclic containers")
        occurrence = self.visits[site]
        self.visits[site] += 1
        before = None
        changes = {}
        edit = self.edit
        if edit and (site, occurrence) == (edit.site, edit.occurrence):
            if kind in ("entry", "exit"):
                raise ValueError("Cannot edit input entry or final output")
            before = copy.deepcopy(values)
            container, key = locate(values, edit.variable)
            target = container[key]
            if edit.operation == "set":
                container[key] = edit.value
                name, path = parse(edit.variable)
                if not path:
                    changes[name] = edit.value
            elif type(target) is not list:
                raise ValueError("Frontier edits require a live list")
            elif edit.operation == "clear":
                target.clear()
            elif edit.operation == "reverse":
                target.reverse()
            elif edit.operation == "drop":
                target.pop(edit.value)
            else:
                raise ValueError("Unknown edit")
            self.edits += 1
        groups: dict[int, list[str]] = {}
        for name, value in values.items():
            if type(value) in (list, dict):
                groups.setdefault(id(value), []).append(name)
        aliases = tuple(tuple(names) for names in groups.values() if len(names) > 1)
        self.snapshots.append(
            Snapshot(
                len(self.snapshots),
                site,
                occurrence,
                kind,
                copy.deepcopy(values),
                output,
                before,
                aliases,
            )
        )
        return changes


class _Instrumenter:
    def __init__(self, names: set[str], source: str):
        self.names = sorted(names)
        self.source = source
        self.sites: dict[str, dict] = {}

    def hook(self, site: str, kind: str, anchor: ast.stmt, output="None", rebind=True):
        """Inject capture and explicit local rebinding without frame mutation."""
        self.sites[site] = {
            "kind": kind,
            "line": anchor.lineno,
            "source": self.source.splitlines()[anchor.lineno - 1].strip(),
        }
        code = f"__changes = __capture({site!r}, {kind!r}, locals(), {output})\n"
        if rebind:
            for name in self.names:
                code += f"if {name!r} in __changes:\n    {name} = __changes[{name!r}]\n"
        nodes = ast.parse(code).body
        for node in nodes:
            ast.copy_location(node, anchor)
        return nodes

    def block(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        """Instrument statements, recurring loop tests, and every return."""
        result: list[ast.stmt] = []
        for node in statements:
            site = f"{node.lineno}:{node.col_offset}:{type(node).__name__}"
            if isinstance(node, ast.While):
                # Observe before EVERY test, including the final false test.
                body = self.hook(site, "loop", node)
                body += [
                    ast.copy_location(
                        ast.If(ast.UnaryOp(ast.Not(), node.test), [ast.Break()], []),
                        node,
                    )
                ]
                body += self.block(node.body)
                result.append(
                    ast.copy_location(ast.While(ast.Constant(True), body, []), node)
                )
                continue
            result += self.hook(site, "statement", node)
            if isinstance(node, ast.Return):
                result.append(
                    ast.copy_location(
                        ast.Assign(
                            [ast.Name("__result", ast.Store())],
                            node.value or ast.Constant(None),
                        ),
                        node,
                    )
                )
                result += self.hook(f"exit:{site}", "exit", node, "__result", False)
                result.append(
                    ast.copy_location(
                        ast.Return(ast.Name("__result", ast.Load())), node
                    )
                )
            else:
                if isinstance(node, (ast.If, ast.For)):
                    node.body = self.block(node.body)
                    node.orelse = self.block(node.orelse)
                result.append(node)
        return result


class Program:
    """Compile plain and checkpointed versions of a restricted policy."""

    def __init__(self, source: str):
        self.source = source
        self.digest = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise UnsupportedProgram("Search runner expects one function")
        function = tree.body[0]
        args = function.args
        if (
            len(args.args) != 1
            or args.defaults
            or args.kwonlyargs
            or args.posonlyargs
            or args.vararg
            or args.kwarg
            or function.decorator_list
            or function.returns
            or args.args[0].annotation
        ):
            raise UnsupportedProgram(
                "Search runner expects one plain positional argument"
            )
        forbidden = (
            ast.Import,
            ast.ImportFrom,
            ast.ListComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Set,
            ast.comprehension,
        )
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED) or isinstance(node, forbidden):
                raise UnsupportedProgram(
                    f"Unsupported search-runner syntax: {type(node).__name__}"
                )
            if isinstance(node, ast.FunctionDef) and node is not function:
                raise UnsupportedProgram("Helpers are outside the first search runner")
            if isinstance(node, (ast.While, ast.For)) and node.orelse:
                raise UnsupportedProgram("Loop else is outside the first search runner")
            if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef)):
                name = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.arg if isinstance(node, ast.arg) else node.name
                )
                if (
                    name.startswith("__")
                    or name == "locals"
                    or (
                        name in BUILTINS
                        and not (
                            isinstance(node, ast.Name)
                            and isinstance(node.ctx, ast.Load)
                        )
                    )
                ):
                    raise UnsupportedProgram("Reserved identifier")
            if isinstance(node, ast.Attribute) and node.attr not in METHODS:
                raise UnsupportedProgram("Unsupported container method")
            if isinstance(node, ast.Call):
                allowed = (
                    isinstance(node.func, ast.Name) and node.func.id in BUILTINS
                ) or (
                    isinstance(node.func, ast.Attribute) and node.func.attr in METHODS
                )
                if not allowed:
                    raise UnsupportedProgram(
                        "Only permitted builtins/container methods"
                    )
        self.name = function.name
        self.argument = args.args[0].arg
        self.plain = compile(tree, "<search-policy>", "exec")
        names = {
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        } | {self.argument}
        transformed = copy.deepcopy(function)
        instrumenter = _Instrumenter(names, source)
        transformed.body = instrumenter.hook(
            "entry", "entry", function, rebind=False
        ) + instrumenter.block(transformed.body)
        # Record implicit return too, so missing termination is never silently accepted.
        implicit = ast.copy_location(ast.Return(ast.Constant(None)), function)
        transformed.body += instrumenter.block([implicit])
        transformed_tree = ast.fix_missing_locations(ast.Module([transformed], []))
        self.code = compile(transformed_tree, "<search-policy>", "exec")
        self.sites = instrumenter.sites

    def _execute(
        self, code: CodeType, position: int | TaskCase, capture, budget: int
    ) -> Any:
        if not isinstance(position, TaskCase) and (
            type(position) is not int or not 0 <= position < 6
        ):
            raise ValueError("Position must be an integer from 0 to 5 or a TaskCase")
        namespace = {
            "__builtins__": {**BUILTINS, "locals": locals},
            "__capture": capture,
        }
        exec(code, namespace)  # pylint: disable=exec-used
        remaining = budget

        def guard(frame, event, _arg):
            nonlocal remaining
            if frame.f_code.co_filename == "<search-policy>" and event == "line":
                remaining -= 1
                if remaining < 0:
                    raise RuntimeError("Concrete line budget exhausted")
            return guard

        previous = sys.gettrace()
        try:
            sys.settrace(guard)
            return namespace[self.name](
                position.payload() if isinstance(position, TaskCase) else position
            )
        finally:
            sys.settrace(previous)

    def run(
        self, position: int | TaskCase, edit: Edit | None = None, budget: int = 20_000
    ) -> Trace:
        """Replay from fresh globals and optionally edit a live checkpoint."""
        recorder = _Recorder(edit, 2000)
        output = self._execute(self.code, position, recorder.capture, budget)
        case = position if isinstance(position, TaskCase) else None
        start = position.start if isinstance(position, TaskCase) else position
        return Trace(start, output, recorder.snapshots, recorder.edits, case)

    def original(self, position: int | TaskCase):
        """Run the uninstrumented policy, also used for separate task assessment."""
        return self._execute(self.plain, position, None, 20_000)

    def observe(self, position: int | TaskCase) -> Trace:
        """Check observation against the genuinely uninstrumented function."""
        plain = self.original(position)
        trace = self.run(position)
        if plain != trace.output:
            raise AssertionError("Observation changed the policy output")
        return trace
