"""FFW-SH5 MuJoCo 텔레오퍼레이션 패키지.

하위 패키지는 application, visualization, kinematics, control 책임으로 분리한다.
최상위에서는 무거운 GUI·MuJoCo 모듈을 자동 import하지 않는다.
"""

import importlib.util
import os
import sys
from pathlib import Path


def _share_imgui_glfw_on_macos():
    """Make glfw and imgui-bundle use one Cocoa GLFW library."""
    if sys.platform != "darwin" or "PYGLFW_LIBRARY" in os.environ:
        return
    spec = importlib.util.find_spec("imgui_bundle")
    if spec is None or spec.origin is None:
        return
    bundled_glfw = Path(spec.origin).parent / "libglfw.3.dylib"
    if bundled_glfw.is_file():
        os.environ["PYGLFW_LIBRARY"] = str(bundled_glfw)


_share_imgui_glfw_on_macos()

__version__ = "3.1.0"

__all__ = ["__version__"]
