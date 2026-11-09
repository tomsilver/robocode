"""Original policies, with no role annotations or manual checkpoints."""

FIFO = """\
def policy(pos):
    root = pos // 2
    todo = [root]
    parent = {root: None}
    while todo:
        x = todo.pop(0)
        if x >= 2:
            while parent[x] is not None and parent[x] != root:
                x = parent[x]
            action = 0 if x == root else x - root
            return action
        for y in (x - 1, x + 1):
            if 0 <= y < 3 and y not in parent:
                parent[y] = x
                todo.append(y)
    return 0
"""

RENAMED = """\
def f(p):
    a = p // 2 + 10
    b = [a]
    c = {a: None}
    while b:
        d = b.pop(0)
        if d >= 12:
            while c[d] is not None and c[d] != a:
                d = c[d]
            e = 0 if d == a else d - a
            return e
        for e in (d - 1, d + 1):
            if 10 <= e < 13 and e not in c:
                c[e] = d
                b.append(e)
    return 0
"""

REACTIVE = """\
def policy(pos):
    action = 1 if pos < 4 else 0
    return action
"""

DECOY = """\
def policy(pos):
    root = pos // 2
    todo = [root]
    parent = {root: None}
    while todo:
        x = todo.pop(0)
        if x >= 2:
            break
        for y in (x - 1, x + 1):
            if 0 <= y < 3 and y not in parent:
                parent[y] = x
                todo.append(y)
    action = 1 if pos < 4 else 0
    return action
"""

LOOKUP = """\
def policy(pos):
    action = [1, 1, 1, 1, 0, 0][pos]
    return action
"""

FIXTURES = {
    "fifo": FIFO,
    "renamed": RENAMED,
    "lifo": FIFO.replace("pop(0)", "pop(-1)"),
    "reactive": REACTIVE,
    "lookup": LOOKUP,
    "decoy": DECOY,
}

GROUND_TRUTH = {
    "fifo": "planning: FIFO search",
    "lifo": "planning: LIFO search",
    "renamed": "planning: renamed FIFO search",
    "reactive": "not planning: direct conditional action",
    "lookup": "not planning: precomputed action lookup",
    "decoy": "search computed but ignored by the returned decision",
}
