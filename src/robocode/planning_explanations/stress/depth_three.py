# pylint: skip-file
# Tests: search shape. A full three-step search (8 leaves), combined with max.
# Prediction (before running): UNRESOLVED. Only depth 1 and 2 models exist, and the
# program never reads utilities at depth 1 or 2, so no shallower model fits either.
# Prediction for fix 3 (depth as a parameter, before running): SUPPORTED at depth 3
# with leaf values. The per-node maxima in `best` are extra values, not required.
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        first = t[s][a]
        best = []
        for b in range(2):
            second = t[first][b]
            values = []
            for c in range(2):
                values.append(u[t[second][c]])
            best.append(max(values))
        scores.append(max(best))
    return int(scores[1] > scores[0])
