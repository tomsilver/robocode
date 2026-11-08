# pylint: skip-file
# Tests: Python subset. A two-step search written as one list comprehension.
# Prediction (before running): REFUSED, because comprehensions are not allowed.
# Prediction for fix 4 (wider Python, before running): UNRESOLVED. `t[s][a]` is
# recomputed for each leaf, so no single location holds the first state for both
# of its leaves; each read moves one leaf only (like `left` in unrolled.py), and
# states_1 has no surviving location.
def policy(s, t, u, cache):
    scores = [max([u[t[t[s][a]][b]] for b in range(2)]) for a in range(2)]
    return int(scores[1] > scores[0])
