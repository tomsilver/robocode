# pylint: skip-file
# Tests: a one-step prediction scored by the program's own preference. It looks at
# where each action leads and prefers the higher-numbered state; it never reads u.
# Prediction (before running, gap 2): LOOKS AHEAD but unresolved. Changing where an
# action leads changes its choice, so it is not reactive; no model fits (it ignores
# utilities), so planningness stays [0, max 3].
def policy(s, t, u, cache):
    return int(t[s][1] > t[s][0])
