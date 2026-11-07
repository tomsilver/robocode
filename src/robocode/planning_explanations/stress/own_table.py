# pylint: skip-file
# Tests: whether the analyzer can tell "reads the environment's transitions" from
# "carries its own copy". The same two-step search as the two_step fixture, but it
# looks up successors in a hard-coded copy of the original fixed table, never in `t`.
# Prediction (before running):
# - One fixed table for every input (before fix 1): SUPPORTED (depth 2). The copy
#   equals `t` on every input, so this is indistinguishable from the two_step fixture.
# - A new table for every input (after fix 1): UNRESOLVED. Wherever the environment's
#   table differs from the copy, its decisions differ from the model's, so both depths
#   are rejected before matching. (It still plans, with its own model: fix 7.)
def policy(s, t, u, cache):
    table = ((1, 2), (3, 4), (4, 5), (0, 5), (1, 3), (2, 0))
    scores = []
    for a in range(2):
        successor = table[s][a]
        values = []
        for b in range(2):
            value = u[table[successor][b]]
            values.append(value)
        scores.append(max(values))
    return int(scores[1] > scores[0])
