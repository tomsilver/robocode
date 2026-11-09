"""Generic expression-value tracing and single-checkpoint substitution.

This is a deliberately restricted Python runner, not a security sandbox. A program is
a sequence of allowlisted imports (``heapq``, ``math``) and function definitions. The
LAST top-level function is the policy; every other function, top-level or nested, is
a helper. Bodies may use assignments, for and while loops, if statements,
comprehensions, generator expressions, recursion, and allowlisted builtins and
container methods. Observation returns the original object, preserving aliases;
only trace snapshots are copied. Patches replace an expression's returned value.

A checkpoint is an expression's source span; its visit number counts executions in
order. Within one route (the same control-flow decisions) calls happen in the same
order, so "visit k of this line inside a helper" already identifies its caller.
"""

# The supported runtime data subset deliberately excludes custom subclasses.
# pylint: disable=unidiomatic-typecheck

import ast
import copy
import hashlib
import heapq
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from types import ModuleType

BUILTINS = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "sorted": sorted,
    "reversed": reversed,
    "enumerate": enumerate,
    "zip": zip,
    "any": any,
    "all": all,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
}
MODULES = {"heapq": heapq, "math": math}
# Container methods a program may call. Sets work but are never recorded.
METHODS = frozenset(
    {"append", "extend", "insert", "pop", "get", "items", "keys", "values", "add"}
)
# Loop iterations per run, counted in both the plain and the traced program, so a
# runaway loop fails instead of hanging the analyzer.
MAX_LOOP_ITERATIONS = 10_000
ALLOWED = (
    ast.Module,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.For,
    ast.While,
    ast.If,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.Tuple,
    ast.ListComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Subscript,
    ast.Slice,
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
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.And,
    ast.Or,
)


class UnsupportedProgram(ValueError):
    """The source is outside the prototype's explicit Python subset."""


class ProgramFailure(RuntimeError):
    """The program itself raised (or looped too long) on an ordinary input."""


def reserved(name):
    """Names the runner uses itself."""
    return name.startswith("__") or name in BUILTINS


def import_bindings(nodes):
    """Names bound by allowlisted import statements, mapped to their objects."""
    bindings = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in MODULES:
                    raise UnsupportedProgram(f"Import not allowed: {alias.name}")
                bindings[alias.asname or alias.name] = MODULES[alias.name]
            continue
        if node.level or node.module not in MODULES:
            raise UnsupportedProgram(f"Import not allowed: {node.module}")
        for alias in node.names:
            module = MODULES[node.module]
            if alias.name.startswith("_") or not hasattr(module, alias.name):
                raise UnsupportedProgram(
                    f"Import not allowed: {node.module}.{alias.name}"
                )
            bindings[alias.asname or alias.name] = getattr(module, alias.name)
    if any(reserved(name) for name in bindings):
        raise UnsupportedProgram("Reserved identifier")
    return bindings


def check_function(function):
    """Plain positional parameters only, for the policy and helpers alike."""
    if reserved(function.name):
        raise UnsupportedProgram("Reserved function identifier")
    arguments = function.args
    if (
        function.decorator_list
        or function.returns
        or arguments.defaults
        or arguments.kwonlyargs
        or arguments.vararg
        or arguments.kwarg
        or arguments.posonlyargs
        or any(arg.annotation for arg in arguments.args)
    ):
        raise UnsupportedProgram(
            "Decorators, annotations and optional arguments unsupported"
        )


def validate(tree):
    """Reject unsupported constructs; return the policy's name and import bindings."""
    imports, functions = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            raise UnsupportedProgram(
                "Only imports and function definitions are allowed at the top level"
            )
    if not functions:
        raise UnsupportedProgram("Supply a function; the last one is the policy")
    if len(functions[-1].args.args) != 4:
        raise UnsupportedProgram(
            "The last function is the policy and takes positional state, "
            "transitions, utilities, cache"
        )
    bindings = import_bindings(imports)
    modules = {name for name, value in bindings.items() if type(value) is ModuleType}
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    unsupported = next((n for n in ast.walk(tree) if not isinstance(n, ALLOWED)), None)
    if unsupported is not None:
        raise UnsupportedProgram(f"Unsupported syntax: {type(unsupported).__name__}")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            check_function(node)
        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if name.startswith("__") or (
                name in BUILTINS
                and (isinstance(node, ast.arg) or isinstance(node.ctx, ast.Store))
            ):
                raise UnsupportedProgram("Reserved identifier")
        if isinstance(node, ast.Attribute):
            on_module = isinstance(node.value, ast.Name) and node.value.id in modules
            if (
                not isinstance(node.ctx, ast.Load)
                or node.attr.startswith("_")
                or not (on_module or node.attr in METHODS)
            ):
                raise UnsupportedProgram(f"Unsupported attribute: .{node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Attribute) and not (
                isinstance(func, ast.Name)
                and (func.id in BUILTINS or func.id in defined or func.id in bindings)
            ):
                raise UnsupportedProgram(
                    "Only builtin, helper, imported and container-method calls "
                    "are supported"
                )
    return functions[-1].name, bindings


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


def replaced(value, path, new):
    """``value`` with the element at ``path`` replaced; tuples are rebuilt.

    Lists and dictionaries along the path are modified in place, so callers pass a
    copy.
    """
    if not path:
        return copy.deepcopy(new)
    key, rest = path[0], path[1:]
    if type(value) is tuple:
        return value[:key] + (replaced(value[key], rest, new),) + value[key + 1 :]
    value[key] = replaced(value[key], rest, new)
    return value


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
            value = replaced(
                copy.deepcopy(value), self.patch.selector.path, self.patch.value
            )
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


def loop_counter(limit=MAX_LOOP_ITERATIONS):
    """A per-run tick that fails once the program has looped ``limit`` times."""
    count = 0

    def tick():
        nonlocal count
        count += 1
        if count > limit:
            raise RuntimeError("Loop iteration budget exceeded")

    return tick


def hook(name, *args):
    """A call to one of the runner's hooks."""
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[]
    )


class Instrumenter(ast.NodeTransformer):
    """Insert a hook around expressions, preserving Python evaluation order.

    ``if`` and ``while`` tests and comprehension filters are also recorded as branch
    decisions; together they form an execution's route.
    """

    def __init__(self, source):
        self.source = source
        self.sites = {}

    def visit(self, node):
        # Callees, methods and slices are not scalar/list/dictionary checkpoints.
        original = node
        node = super().visit(node)
        if isinstance(original, ast.expr) and not isinstance(
            original, (ast.Attribute, ast.Slice)
        ):
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
                    hook("__capture", ast.Constant(site), node), original
                )
        return node

    def visit_If(self, node):  # pylint: disable=invalid-name
        """Record branch signatures independently of identifier spellings."""
        node = self.generic_visit(node)
        node.test = hook("__branch", ast.Constant(f"if:{node.lineno}"), node.test)
        return node

    def visit_While(self, node):  # pylint: disable=invalid-name
        """Each loop test is a branch decision, so iteration counts set the route."""
        node = self.generic_visit(node)
        node.test = hook("__branch", ast.Constant(f"while:{node.lineno}"), node.test)
        return node

    def visit_comprehension(self, node):
        """A comprehension filter decides like an ``if`` statement."""
        node = self.generic_visit(node)
        node.ifs = [
            hook(
                "__branch",
                ast.Constant(f"filter:{test.lineno}:{test.col_offset}"),
                test,
            )
            for test in node.ifs
        ]
        return node


class LoopGuard(ast.NodeTransformer):
    """Count every loop iteration, identically in the plain and traced programs."""

    def visit_For(self, node):  # pylint: disable=invalid-name
        """Tick before each iteration's body."""
        node = self.generic_visit(node)
        node.body.insert(0, ast.Expr(hook("__tick")))
        return node

    visit_While = visit_For


class Program:
    """Compile plain and instrumented versions of the same restricted source."""

    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        self.name, self.bindings = validate(tree)
        # Imports are bound directly in each run's namespace instead of executed.
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        plain = LoopGuard().visit(copy.deepcopy(tree))
        self.plain_code = compile(
            ast.fix_missing_locations(plain), "<plain-policy>", "exec"
        )
        instrumenter = Instrumenter(source)
        traced = LoopGuard().visit(instrumenter.visit(copy.deepcopy(tree)))
        self.code = compile(
            ast.fix_missing_locations(traced), "<traced-policy>", "exec"
        )
        self.sites = instrumenter.sites

    def _namespace(self, hooks=None):
        """Fresh globals: builtins, imported names, a loop counter and any hooks."""
        return {
            "__builtins__": BUILTINS,
            "__tick": loop_counter(),
            **self.bindings,
            **(hooks or {}),
        }

    def run(self, case, patch=None):
        """Execute from fresh inputs and globals, returning a complete trace."""
        recorder = Recorder(patch)
        # A dictionary, because "__name" keywords would be mangled inside a class.
        namespace = self._namespace(
            {"__capture": recorder.capture, "__branch": recorder.branch}
        )
        exec(self.code, namespace)  # pylint: disable=exec-used
        args = case.args()
        output = namespace[self.name](*args)
        return Trace(
            output, recorder.events, recorder.branches, args, recorder.patch_count
        )

    def output(self, case):
        """The plain program's action on ``case``, with no recording or patches."""
        namespace = self._namespace()
        exec(self.plain_code, namespace)  # pylint: disable=exec-used
        return namespace[self.name](*case.args())

    def check_observation(self, case):
        """Check output AND input mutation against an uninstrumented execution."""
        try:
            namespace = self._namespace()
            exec(self.plain_code, namespace)  # pylint: disable=exec-used
            args = case.args()
            output = namespace[self.name](*args)
            trace = self.run(case)
        except UnsupportedProgram:
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Any error raised by the analyzed program itself, reported as such.
            raise ProgramFailure(
                f"{type(exc).__name__} on {case.label}: {exc}"
            ) from exc
        if output != trace.output or args != trace.args_after:
            raise AssertionError(f"Observation changed behavior on {case.label}")
        return trace
