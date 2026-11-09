# pylint: skip-file
# Tests: whether an internal model is detected from what it computes, not how it is
# written. The two-step search, but successors come from a formula in a helper
# function, never from `t`.
# Prediction (before running, fix 7): SUPPORTED (depth 2, leaf) with an own transition
# table equal to the formula: state x leads to (2x + 1) % 6 and (2x + 2) % 6.
# Agreement with the environment about 1 in 6 pairs.
def step(x, a):
    return (2 * x + a + 1) % 6


def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = step(s, a)
        values = [u[step(successor, b)] for b in range(2)]
        scores.append(max(values))
    return int(scores[1] > scores[0])
