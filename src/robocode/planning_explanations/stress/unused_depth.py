# pylint: skip-file
# Tests: computed is not the same as used (negative control). It runs the full
# two-step search, throws the result away, and decides with a one-step look.
# Prediction (before running): SUPPORTED at depth 1 ONLY. Every depth-2 explanation is
# rejected because the output doesn't follow the two-step scores. The depth-1
# explanation must use the second loop: patching the first loop changes nothing.
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
        greedy.append(u[t[s][a]])
    return int(greedy[1] > greedy[0])
