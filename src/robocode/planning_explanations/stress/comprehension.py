# pylint: skip-file
# Tests: Python subset. A two-step search written as one list comprehension.
# Prediction (before running): REFUSED, because comprehensions are not allowed.
def policy(s, t, u, cache):
    scores = [max([u[t[t[s][a]][b]] for b in range(2)]) for a in range(2)]
    return int(scores[1] > scores[0])
