# pylint: skip-file
# Tests: matching. A two-step search that remembers utilities it already looked up.
# With the fixed map, start states 0 and 3 reach one leaf state twice, so `u[leaf]`
# is read only 3 times there, but `memo[leaf]` is still read 4 times.
# Prediction (before running): SUPPORTED on both routes (repeat / no repeat), with
# the evaluations found at the `memo[leaf]` read, not at `u[leaf]`. Medium confidence.
def policy(s, t, u, cache):
    known = [False] * 6
    memo = [0] * 6
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            leaf = t[successor][b]
            if not known[leaf]:
                memo[leaf] = u[leaf]
                known[leaf] = True
            values.append(memo[leaf])
        scores.append(max(values))
    return int(scores[1] > scores[0])
