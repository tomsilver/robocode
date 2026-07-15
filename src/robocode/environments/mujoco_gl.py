"""Choose and lock the MuJoCo / PyOpenGL rendering backend for this process.

Importing ``mujoco`` binds PyOpenGL permanently to whatever GL platform the env
vars name at that moment, and anything that renders a kinder env pulls in mujoco.
So the platform is chosen once, here: default EGL, but honor a ``MUJOCO_GL`` the
caller set (the sandbox picks ``osmesa`` since headless EGL needs a GPU).
"""

import importlib
import os

# PyOpenGL's platform must match MUJOCO_GL. glfw is on-screen (GLX on Linux); the
# rest map to themselves.
_PYOPENGL_FOR_MUJOCO = {"egl": "egl", "osmesa": "osmesa", "glfw": "glx"}

# Import mujoco at most once, on first configuration (mutated in place, no global).
_mujoco_import = {"attempted": False}


def configure_gl_backend() -> tuple[str, str]:
    """Set and lock the GL backend once (idempotent); return the current pair.

    Callers that then run code overwriting those env vars (kinder registration
    forces osmesa on headless Linux) can restore the returned choice.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault(
        "PYOPENGL_PLATFORM",
        _PYOPENGL_FOR_MUJOCO.get(os.environ["MUJOCO_GL"], "egl"),
    )
    if not _mujoco_import["attempted"]:
        _mujoco_import["attempted"] = True
        try:
            # Binds PyOpenGL to MUJOCO_GL. Optional dep whose GL-touching import
            # can raise when runtime libs are missing; a failure just means no mujoco.
            importlib.import_module("mujoco")
        except Exception:  # pylint: disable=broad-except
            pass
    return os.environ["MUJOCO_GL"], os.environ["PYOPENGL_PLATFORM"]
