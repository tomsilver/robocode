# pylint: skip-file
# Tests: Python subset. The same two-step search as two_step, written with while loops.
# Prediction (before running): REFUSED, because `while` is not allowed syntax.
def policy(s, t, u, cache):
    scores = []
    a = 0
    while a < 2:
        successor = t[s][a]
        values = []
        b = 0
        while b < 2:
            values.append(u[t[successor][b]])
            b += 1
        scores.append(max(values))
        a += 1
    return int(scores[1] > scores[0])
