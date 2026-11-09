# pylint: skip-file
# Tests: the limit of a FIXED internal table. The two-step search reads the
# environment's table `t`, except that it believes state 0 is absorbing (both actions
# stay in 0). Its model is the environment with one row wrong.
# Prediction (before running, fix 7): UNRESOLVED. The environment model is rejected
# before matching (its decisions differ whenever state 0 is reached), and no fixed
# table fits either, because the other rows change with the input. Covering this
# needs a freer internal model ("the environment, except ...").
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = 0 if s == 0 else t[s][a]
        values = []
        for b in range(2):
            leaf = 0 if successor == 0 else t[successor][b]
            values.append(u[leaf])
        scores.append(max(values))
    return int(scores[1] > scores[0])
