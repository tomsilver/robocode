# pylint: skip-file
# Tests: how values are combined. A two-step search that scores a path as
# u[middle state] + u[leaf] instead of the leaf utility alone.
# Prediction (before running): UNRESOLVED. States and leaf utilities are all there,
# but no recorded value equals the model's branch score (max of leaf utilities).
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            leaf = t[successor][b]
            values.append(u[successor] + u[leaf])
        scores.append(max(values))
    return int(scores[1] > scores[0])
