"""Ordinary Python fixtures: no annotations, hooks or internal role metadata.

The fourth argument is an externally precomputed two-step policy. It lets the
lookup and decoy controls match the two-step planner on EVERY ordinary case.
"""

ONE_STEP = """\
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        value = u[successor]
        scores.append(value)
    return int(scores[1] > scores[0])
"""

TWO_STEP = """\
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            leaf = t[successor][b]
            value = u[leaf]
            values.append(value)
        scores.append(max(values))
    return int(scores[1] > scores[0])
"""

RENAMED = """\
def f(x, m, w, k):
    d = {}
    for i in range(2):
        p = m[x][i]
        z = {}
        for j in range(2):
            r = m[p][j]
            z[j] = w[r]
        d[i] = max(z[0], z[1])
    return int(d[1] > d[0])
"""

LOOKUP = """\
def f(x, m, w, k):
    return k[x]
"""

CONDITIONAL = """\
def policy(s, t, u, cache):
    if s % 2 == 0:
        return 0
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            leaf = t[successor][b]
            value = u[leaf]
            values.append(value)
        scores.append(max(values))
    return int(scores[1] > scores[0])
"""

DECOY = TWO_STEP.replace("return int(scores[1] > scores[0])", "return cache[s]")

FIXTURES = {
    "one_step": ONE_STEP,
    "two_step": TWO_STEP,
    "renamed_dictionary": RENAMED,
    "lookup": LOOKUP,
    "conditional": CONDITIONAL,
    "decoy": DECOY,
}
