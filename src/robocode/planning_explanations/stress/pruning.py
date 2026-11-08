# pylint: skip-file
# Tests: search shape. A two-step search that skips a leaf when the answer is already
# known: if action 1's first leaf beats action 0's score, action 1 wins whatever the
# last leaf is. It always returns the same action as the full search.
# Prediction (before running): NOT SUPPORTED.
# - Inputs where it skips a leaf: only 3 leaves are read -> UNRESOLVED.
# - Inputs where it reads all 4: a patch can trigger the skip, so a leaf the model
#   expects is never read -> REJECTED (intervention_mismatch).
# Prediction for fix 5 (search events with an optional cutoff, before running):
# SUPPORTED on both routes by the depth-2 leaf-utility model WITH cutoff: it skips
# exactly the leaf this program skips. Evaluated alternatives: 3 where it prunes,
# 4 where it does not. The same model without cutoff stays rejected.
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            value = u[t[successor][b]]
            values.append(value)
            if scores and value > scores[0]:
                break
        scores.append(max(values))
    return int(scores[1] > scores[0])
