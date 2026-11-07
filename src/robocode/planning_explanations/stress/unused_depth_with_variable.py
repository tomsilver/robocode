# pylint: skip-file
# Tests: style sensitivity. Added AFTER the first run, as a follow-up to unused_depth.
# Same program as unused_depth.py, except the one-step value goes through a variable
# (`value = ...` then `greedy.append(value)`). Behaviour is identical.
# Prediction: SUPPORTED at depth 1 (checked once before saving this file).
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            values.append(u[t[successor][b]])
        scores.append(max(values))
    greedy = []
    for a in range(2):
        value = u[t[s][a]]
        greedy.append(value)
    return int(greedy[1] > greedy[0])
