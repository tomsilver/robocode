# pylint: skip-file
# Tests: the boundary of the definition. A two-step search over a hard-coded table,
# scored by a hard-coded preference (a higher state is better). It reads nothing but
# the start state, so its action is a fixed function of that state, like a lookup
# table.
# Prediction (before running, gap 2): NO LOOKAHEAD. Internally it searches, but the
# search never responds to the environment, so its behavior is reactive. No model
# fits either, since it reads no utilities.
def policy(s, t, u, cache):
    table = ((1, 2), (3, 4), (4, 5), (0, 5), (1, 3), (2, 0))
    scores = []
    for a in range(2):
        successor = table[s][a]
        scores.append(max(table[successor][0], table[successor][1]))
    return int(scores[1] > scores[0])
