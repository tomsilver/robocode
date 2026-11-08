# pylint: skip-file
# Tests: whether "cutoff" means any pruning. This program gives up on action 1 as soon
# as one of its leaves is WORSE than action 0's score, assuming the other leaf will be
# worse too. That is unsound: when the skipped leaf is better, it picks action 0 while
# a full search picks action 1.
# Prediction (before running, fix 5): UNRESOLVED, rejected before matching by every
# model: the sound cutoff never changes a decision, so cutoff models decide like full
# search, and this program does not.
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            value = u[t[successor][b]]
            values.append(value)
            if scores and value < scores[0]:
                break
        scores.append(max(values))
    return int(scores[1] > scores[0])
