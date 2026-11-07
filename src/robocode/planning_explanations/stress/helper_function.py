# pylint: skip-file
# Tests: Python subset. A two-step search that uses a small helper function.
# Prediction (before running): REFUSED, because nested functions are not allowed.
def policy(s, t, u, cache):
    def best_leaf(state):
        return max(u[t[state][0]], u[t[state][1]])

    left = best_leaf(t[s][0])
    right = best_leaf(t[s][1])
    return int(right > left)
