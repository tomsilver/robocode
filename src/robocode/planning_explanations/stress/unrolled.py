# pylint: skip-file
# Tests: matching. A correct two-step search with no loops: each leaf has its own line.
# Prediction (before running): UNRESOLVED. A mapping can only take a role's values
# from one expression (its repeated visits) or one list. Here the four leaf values
# come from four different expressions.
def policy(s, t, u, cache):
    left = t[s][0]
    right = t[s][1]
    left_score = max(u[t[left][0]], u[t[left][1]])
    right_score = max(u[t[right][0]], u[t[right][1]])
    return int(right_score > left_score)
