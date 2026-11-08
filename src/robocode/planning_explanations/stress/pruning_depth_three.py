# pylint: skip-file
# Tests: whether the cutoff generalizes with depth. A full three-step search that stops
# expanding action 1 (both loops) as soon as one of its leaves beats action 0's score.
# Prediction (before running, fix 5): SUPPORTED by the depth-3 leaf-utility model with
# cutoff on every route seen in discovery. Where it stops depends on the input (after
# leaf 1, 2, 3 or 4 of action 1, or never), so there are several routes with few
# inputs each, and some validation inputs may fall on a route discovery never saw.
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        first = t[s][a]
        values = []
        stop = False
        for b in range(2):
            second = t[first][b]
            for c in range(2):
                value = u[t[second][c]]
                values.append(value)
                if scores and value > scores[0]:
                    stop = True
                    break
            if stop:
                break
        scores.append(max(values))
    return int(scores[1] > scores[0])
