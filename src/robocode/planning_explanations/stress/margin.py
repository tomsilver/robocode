# pylint: skip-file
# Tests: whether patches probe the decision boundary. The two-step search, but action 1
# must win by MORE THAN 0.5. Every utility ends in .25, so score differences are whole
# numbers in ordinary runs and also under the low/high (±100) patches; there this rule
# always agrees with the model's plain comparison. It disagrees only when action 1
# leads by 0.5 or less, which no ordinary run or ±100 patch ever produces.
# Prediction (before running):
# - Low/high patches only (before fix 2): SUPPORTED (depth 2). Wrong: its decision
#   rule is not the model's.
# - Boundary patches (after fix 2): UNRESOLVED. A patch that puts action 1 just ahead
#   makes the model choose 1 while the program keeps 0.
def policy(s, t, u, cache):
    scores = []
    for a in range(2):
        successor = t[s][a]
        values = []
        for b in range(2):
            leaf = t[successor][b]
            value = u[leaf]
            values.append(value)
        scores.append(max(values))
    return int(scores[1] > scores[0] + 0.5)
