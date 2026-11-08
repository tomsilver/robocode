# pylint: skip-file
# Tests: several top-level functions, recursion, a generator expression and a list
# comprehension. The textbook recursive lookahead; the LAST function is the policy.
# Prediction (before running, fix 4): SUPPORTED (depth 2, leaf). The first state is
# computed once per action and passed as an argument, so the call-site value feeds
# both of its leaves; leaf utilities are the depth-0 `u[state]` reads.
def value(t, u, state, depth):
    if depth == 0:
        return u[state]
    return max(value(t, u, t[state][b], depth - 1) for b in range(2))


def policy(s, t, u, cache):
    scores = [value(t, u, t[s][a], 1) for a in range(2)]
    return int(scores[1] > scores[0])
