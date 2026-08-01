# Environment patches

Reset-distribution variants of kinder environments, kept as patches because the
`third-party/kindergarden` remotes are not writable from here. Each patch is a
one-field change to a spawn range: the sampler code is untouched, so a run under a
patch differs from a standard run only in where objects are placed.

Apply and revert:

```bash
git -C third-party/kindergarden am ../../patches/<name>.patch   # or: apply
git -C third-party/kindergarden checkout <base-sha>             # to revert
```

| patch | effect |
| --- | --- |
| `kindergarden-stickbutton-reset-band.patch` | StickButton2D stick spawns in `[3.435, 3.450]` instead of `[0.0, 3.450]` |

## Why 3.435

That is the region where a bottom grasp is geometrically impossible for **every**
robot placement: bisecting against the environment's own collision and suction code
puts the threshold at stick x >= 3.4329, identical at two stick heights, and a
left-side grasp remains available so the tasks stay solvable. It is 0.43% of the
standard spawn range, so a standard-trained policy meets it roughly once per 230
episodes. See `outputs/reset_dist_a_2026-08-01/evidence/` for the measurements and
the oracle certifications of both suites.

## Keeping it invisible to the agent

The change lives in the kinder source itself rather than in a subclass or wrapper,
so the source the sandboxed agent reads matches the distribution it samples. Nothing
names the variant, and the generated environment description is byte-identical to
the standard one. A mismatch between source and behaviour would be worse than a
leak: it would manufacture the "the environment is broken" misattribution that this
experiment is trying to measure.

Note that `demos/` is excluded from the agent mount (see `_copy_kindergarden_without_tests`);
those recorded trajectories are from the standard distribution and would contradict
what the agent samples under a patch.
