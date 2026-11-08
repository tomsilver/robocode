# pylint: skip-file
# Tests: whether the sum model depends on the order of operations. It makes the same
# decisions as sum_rewards (path value = u[middle] + u[leaf], then max), but adds
# u[middle] once AFTER taking the max, so no per-path sum is ever computed.
# Prediction (before running, after fix 3): UNRESOLVED. The depth-2 sum model needs a
# location for each path's sum (evaluations[0..3]), and none exists; its branch
# scores do exist. Every other model is rejected before matching.
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            values.append(u[t[successor][b]])
        scores.append(u[successor] + max(values))
    return int(scores[1] > scores[0])
