"""Generic expression-value tracing and single-checkpoint substitution.

This is a deliberately restricted Python runner, not a security sandbox. It
accepts one function using assignments, for loops, if statements and a small
set of builtins. Observation returns the original object, preserving aliases;
only trace snapshots are copied. Patches replace an expression's returned value.
"""

# The supported runtime data subset deliberately excludes custom subclasses.
# pylint: disable=unidiomatic-typecheck

import ast
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

BUILTINS = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "int": int,
    "list": list,
    "dict": dict,
}
ALLOWED = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.For,
    ast.If,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Dict,
    ast.Tuple,
    ast.Subscript,
    ast.Load,
    ast.Store,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
)


class UnsupportedProgram(ValueError):
    """The source is outside the prototype's explicit Python subset."""


def validate(tree):
    """Reject unsupported constructs before compiling either version."""
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsupportedProgram("Supply exactly one function")
    function = tree.body[0]
    if function.name.startswith("__") or function.name in BUILTINS:
        raise UnsupportedProgram("Reserved function identifier")
    if (
        function.decorator_list
        or function.returns
        or function.args.defaults
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
        or function.args.posonlyargs
    ):
        raise UnsupportedProgram(
            "Decorators, annotations and optional arguments unsupported"
        )
    if len(function.args.args) != 4:
        raise UnsupportedProgram(
            "Expected positional state, transitions, utilities, cache"
        )
    if any(arg.annotation for arg in function.args.args):
        raise UnsupportedProgram("Annotations unsupported")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise UnsupportedProgram(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise UnsupportedProgram("Nested functions unsupported")
        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if name.startswith("__") or (
                name in BUILTINS
                and (isinstance(node, ast.arg) or isinstance(node.ctx, ast.Store))
            ):
                raise UnsupportedProgram("Reserved identifier")
        if isinstance(node, ast.Attribute) and node.attr != "append":
            raise UnsupportedProgram("Only the append method is supported")
        if isinstance(node, ast.Call):
            allowed = (
                isinstance(node.func, ast.Name) and node.func.id in BUILTINS
            ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "append")
            if not allowed:
                raise UnsupportedProgram(
                    "Only whitelisted builtin calls and append supported"
                )


def is_value(value, ancestors=frozenset()):
    """Trace only the supported data types, never callable or iterator objects."""
    if type(value) in (int, float, bool, str, type(None)):
        return True
    if id(value) in ancestors:
        raise UnsupportedProgram("Cyclic containers unsupported")
    ancestors = ancestors | {id(value)}
    if type(value) in (list, tuple):
        return all(is_value(v, ancestors) for v in value)
    if type(value) is dict:
        return all(
            type(k) in (str, int) and is_value(v, ancestors) for k, v in value.items()
        )
    return False


@dataclass(frozen=True)
class Selector:
    """Selection only: checkpoint, visit number, and literal container path."""

    site: str
    occurrence: int
    path: tuple = ()

    def read(self, trace):
        """Select an existing value; no arithmetic or rollout is permitted."""
        value = trace.event(self).value
        for key in self.path:
            value = value[key]
        return value


@dataclass(frozen=True)
class Patch:
    """One substitution at an expression checkpoint."""

    selector: Selector
    value: object


@dataclass
class Event:
    """A snapshot at an expression checkpoint, in execution order."""

    index: int
    site: str
    occurrence: int
    value: object
    patched: bool = False
    before: object = None


@dataclass
class Trace:
    """An ordinary or patched execution; these populations stay separate."""

    output: object
    events: list
    branches: list
    args_after: tuple
    patch_count: int
    _index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self._index = {(e.site, e.occurrence): e for e in self.events}

    def event(self, selector):
        """Retrieve a checkpoint without searching or computing new values."""
        return self._index[(selector.site, selector.occurrence)]

    @property
    def route(self):
        """A generic executed branch signature, without semantic route labels."""
        encoded = json.dumps(self.branches, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:12]


class Recorder:
    """Per-run recorder; never reused between ordinary and patched executions."""

    def __init__(self, patch=None, max_events=5000):
        self.patch = patch
        self.max_events = max_events
        self.events = []
        self.branches = []
        self.visits = Counter()
        self.patch_count = 0
        self.calls = 0

    def capture(self, site, value):
        """Observe an expression result, optionally replacing that result."""
        self.calls += 1
        if self.calls > self.max_events:
            raise RuntimeError("Expression checkpoint budget exceeded")
        occurrence = self.visits[site]
        self.visits[site] += 1
        if not is_value(value):
            return value
        before = None
        patched = False
        if self.patch and (site, occurrence) == (
            self.patch.selector.site,
            self.patch.selector.occurrence,
        ):
            before = copy.deepcopy(value)
            if self.patch.selector.path:
                value = copy.deepcopy(value)
                parent = value
                for key in self.patch.selector.path[:-1]:
                    parent = parent[key]
                parent[self.patch.selector.path[-1]] = copy.deepcopy(self.patch.value)
            else:
                value = copy.deepcopy(self.patch.value)
            self.patch_count += 1
            patched = True
        self.events.append(
            Event(
                len(self.events),
                site,
                occurrence,
                copy.deepcopy(value),
                patched,
                before,
            )
        )
        return value

    def branch(self, site, value):
        """Record the actual branch decision without changing its truth value."""
        self.branches.append((site, bool(value)))
        return value


class Instrumenter(ast.NodeTransformer):
    """Insert a hook around expressions, preserving Python evaluation order."""

    def __init__(self, source):
        self.source = source
        self.sites = {}

    def visit(self, node):
        # A callee/method itself is not a scalar/list/dictionary checkpoint.
        original = node
        node = super().visit(node)
        if isinstance(original, ast.expr) and not isinstance(original, ast.Attribute):
            if isinstance(getattr(original, "ctx", ast.Load()), ast.Load):
                site = (
                    f"{original.lineno}:{original.col_offset}:"
                    f"{original.end_lineno}:{original.end_col_offset}:"
                    f"{type(original).__name__}"
                )
                self.sites[site] = {
                    "line": original.lineno,
                    "column": original.col_offset,
                    "kind": type(original).__name__,
                    "expression": ast.get_source_segment(self.source, original),
                }
                node = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="__capture", ctx=ast.Load()),
                        args=[ast.Constant(site), node],
                        keywords=[],
                    ),
                    original,
                )
        return node

    def visit_If(self, node):  # pylint: disable=invalid-name
        """Record branch signatures independently of identifier spellings."""
        node = self.generic_visit(node)
        node.test = ast.Call(
            func=ast.Name(id="__branch", ctx=ast.Load()),
            args=[ast.Constant(f"if:{node.lineno}"), node.test],
            keywords=[],
        )
        return node


class Program:
    """Compile plain and instrumented versions of the same restricted source."""

    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        validate(tree)
        self.name = tree.body[0].name
        self.plain_code = compile(tree, "<plain-policy>", "exec")
        instrumenter = Instrumenter(source)
        transformed = ast.fix_missing_locations(instrumenter.visit(copy.deepcopy(tree)))
        self.code = compile(transformed, "<traced-policy>", "exec")
        self.sites = instrumenter.sites

    def run(self, case, patch=None):
        """Execute from fresh inputs and globals, returning a complete trace."""
        recorder = Recorder(patch)
        namespace = {
            "__builtins__": BUILTINS,
            "__capture": recorder.capture,
            "__branch": recorder.branch,
        }
        exec(self.code, namespace)  # pylint: disable=exec-used
        args = case.args()
        output = namespace[self.name](*args)
        return Trace(
            output, recorder.events, recorder.branches, args, recorder.patch_count
        )

    def check_observation(self, case):
        """Check output AND input mutation against an uninstrumented execution."""
        namespace = {"__builtins__": BUILTINS}
        exec(self.plain_code, namespace)  # pylint: disable=exec-used
        args = case.args()
        output = namespace[self.name](*args)
        trace = self.run(case)
        if output != trace.output or args != trace.args_after:
            raise AssertionError(f"Observation changed behavior on {case.label}")
        return trace
