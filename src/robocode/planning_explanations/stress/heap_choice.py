# pylint: skip-file
# Tests: an allowed import (heapq) and tuples. The two-step search, but the action is
# chosen by pushing (-score, action) pairs on a heap and popping the smallest. Ties
# pop the smaller action, the same tie rule as the model.
# Prediction (before running, fix 4): SUPPORTED (depth 2, leaf), scores at
# `max(values)`. The heap entries hold the NEGATED scores, which match nothing.
import heapq


def policy(s, t, u, cache):
    ranked = []
    for a in range(2):
        successor = t[s][a]
        values = [u[t[successor][b]] for b in range(2)]
        heapq.heappush(ranked, (-max(values), a))
    best = heapq.heappop(ranked)
    return best[1]
