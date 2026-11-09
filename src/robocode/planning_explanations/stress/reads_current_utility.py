# pylint: skip-file
# Tests: what counts as "the future". The action depends only on the utility of the
# CURRENT state, never on where actions lead or how good those places are.
# Prediction (before running, gap 2): NO LOOKAHEAD on every input: no change to the
# transitions or to other states' utilities changes its action. Planningness [0, 0].
def policy(s, t, u, cache):
    return int(u[s] > 0)
