# pylint: skip-file
# Tests: the common running-maximum idiom (`best = -inf`, then `if v > best: best = v`)
# with an import from math. The two-step search, keeping the best leaf this way.
# Prediction (before running, fix 4): UNRESOLVED, low confidence about which check.
# Each leaf utility is read twice (in the comparison and in the assignment), and
# neither read alone acts like the model's evaluation: patching the comparison read
# changes only the decision to replace, patching the assignment read skips the
# comparison. Patches that flip a comparison also change the route, so later
# visits shift.
from math import inf


def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        best = -inf
        for b in range(2):
            leaf = t[successor][b]
            if u[leaf] > best:
                best = u[leaf]
        scores.append(best)
    return int(scores[1] > scores[0])
