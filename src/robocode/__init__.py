"""RoboCode: agents for robot physical reasoning.

Pins matplotlib to a non-interactive backend before anything imports it.

The env server renders states on ``socketserver`` worker threads, and
matplotlib's default macOS backend refuses to build a ``FigureManager`` off the
main thread, so ``render_state`` there dies with "Cannot create a GUI
FigureManager outside the main thread using the MacOS backend". Nothing in this
package renders to a screen, so Agg is always the right choice; it is set here,
in the package root, because this is the only module guaranteed to run before
the third-party matplotlib import in ``robocode.rendering``.

``setdefault`` leaves an explicit ``MPLBACKEND`` from the caller alone. This is
the matplotlib counterpart to
:func:`robocode.environments.mujoco_gl.configure_gl_backend`.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
