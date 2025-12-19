# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2
from enum import Enum

from .arguments import register_arguments, Argument
from .util import tuple_of


class FontFamily(Enum):
    SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
    PLAIN = cv2.FONT_HERSHEY_PLAIN
    DUPLEX = cv2.FONT_HERSHEY_DUPLEX
    COMPLEX = cv2.FONT_HERSHEY_COMPLEX
    TRIPLEX = cv2.FONT_HERSHEY_TRIPLEX
    COMPLEX_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL
    SCRIPT_SIMPLEX = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    SCRIPT_COMPLEX = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    @classmethod
    def id(cls, name: str):
        k = name.upper()
        if k in cls.__members__:
            return cls[k].value
        else:
            raise ValueError(f"Unsupported font family {name}")


register_arguments(
    view_scale=Argument(
        opt_name="scale",
        type=float,
        required=False,
        help="Scale of viewport, relative to internal occupancy grid",
    ),
    dpi_scale=Argument(
        type=float,
        required=False,
        help="DPI Scale for UI elements (lines, markers, text), does not affect output size",
    ),
    visualize=Argument(
        "-v",
        action="store_true",
        help="Enable visualization",
    ),
    no_wait=Argument(
        action="store_true",
        help="Do not wait for key stroke after simulation is complete, effective only with the -v flag",
    ),
    line_width_meters=Argument(
        opt_name="line_width",
        type=float,
        required=False,
        help="Line width for visualization in meters",
    ),
    line_color=Argument(
        type=tuple_of(float),
        required=False,
        help="Color of the trajectory line (b,g,r)",
    ),
    raw_slice=Argument(
        opt_name="slice",
        type=tuple_of(int),
        required=False,
        help="Output Image Slice in (x,y,w,h)",
    ),
    font_family=Argument(
        type=FontFamily.id,
        required=False,
        help="Font family for text rendering",
    ),
)

# ==============================================================================
# End argument injection
# ==============================================================================
import cv2, numpy as np
from dataclasses import dataclass
from pathlib import Path

from .util import ownAttributes
from .geometry import Point
from .world import World


@ownAttributes
@dataclass(frozen=False)
class Visualization:
    world: World
    debug: bool = False

    view_scale: float = 2.0
    dpi_scale: float = 2.0
    visualize: bool = False
    no_wait: bool = False
    line_width_meters: float = 0.05
    line_color: tuple[float, float, float] = (0, 0, 255)
    raw_slice: tuple[int, int, int, int] | None = None
    font_family: int = FontFamily.DUPLEX.value

    @property
    def scale(self):
        return self.view_scale * self.dpi_scale

    def slice(self, img: np.ndarray):
        if self.raw_slice is None:
            return img
        x, y, w, h = self.raw_slice
        h0, w0 = self.world.raw_shape
        h1, w1 = img.shape[:2]
        fx, fy = w1 / w0, h1 / h0
        _x, w = int(x * fx), int(w * fx)
        _y, h = int(y * fy), int(h * fy)
        x = max(int(x * fx), 0)
        y = max(int(y * fy), 0)
        if len(img.shape) == 2:
            return img[y : _y + h, x : _x + w]
        elif len(img.shape) == 3:
            return img[y : _y + h, x : _x + w, :]
        else:
            raise ValueError(f"Unsupported image shape {img.shape}")

    def saveImg(self, path: Path | str, img: np.ndarray):
        cv2.imwrite(str(path), self.slice(img))

    @property
    def marker_label_offset(self):
        dx = int(0.2 / self.world.res * self.scale)
        return Point(dx * 2, dx, type=int)

    @property
    def text_scale(self):
        return 0.02 / self.world.res * self.scale

    @property
    def font_weight(self):
        return self.line_width

    @property
    def line_width(self):
        return self.px(self.line_width_meters)

    def px(self, d: float):
        return max(int(d / self.world.res * self.scale), 1)

    def line(
        self,
        img: np.ndarray,
        src: Point[int],
        dst: Point[int],
        color: tuple[int, int, int] | int = 255,
    ):
        cv2.line(img, src, dst, color, self.line_width, cv2.LINE_AA)

    def text(
        self,
        img: np.ndarray,
        text: str,
        pos: Point[int],
        color=(255, 255, 255),
        weight=None,
        outline: int = 0,
        outline_color=(255, 255, 255),
        **kw,
    ):
        weight = weight or self.font_weight
        kw.update(
            img=img,
            text=text,
            org=pos,
            fontFace=self.font_family,
            fontScale=self.text_scale,
            color=color,
            thickness=weight,
            lineType=cv2.LINE_AA,
        )
        if outline > 0:
            t = (outline + 1) * weight
            cv2.putText(**kw | dict(thickness=t, color=outline_color))
        cv2.putText(**kw)

    def draw_src(self, img: np.ndarray, pos: Point[int], color=(255, 0, 0)):
        cv2.circle(img, pos, self.px(0.2), color, self.line_width)
        self.text(img, "SRC", pos + self.marker_label_offset, color, outline=2)

    def draw_dst(self, img: np.ndarray, pos: Point[int], color=(0, 192, 0)):
        cv2.drawMarker(
            img, pos, color, cv2.MARKER_SQUARE, self.px(0.4), self.line_width
        )
        self.text(img, "DST", pos + self.marker_label_offset, color, outline=2)

    def caption(
        self,
        img: np.ndarray,
        text: str,
        fg=(128, 128, 128),
        bg: tuple[int, int, int] | None = None,
    ):
        a = self.px(0.4)
        pos = (a, img.shape[0] - a)
        self.text(img, text, pos, fg, outline=3)

    @property
    def handle(self):
        return self.world.name

    def show(
        self,
        img: np.ndarray,
        desc: str | None = None,
        scale=None,
        wait_key: int | None = 1,
    ) -> tuple[str, np.ndarray, int]:
        if scale is not None:
            s = 1 / scale
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        elif self.dpi_scale is not None:
            s = 1 / self.dpi_scale
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if desc is None:
            title = self.handle
        else:
            title = f"{self.handle} ({desc})"
        if self.visualize:
            cv2.imshow(title, img)
            if wait_key is not None:
                key = cv2.waitKey(wait_key)
            else:
                key = -1
        else:
            key = -1
        return title, img, key

    @property
    def grayscale(self) -> np.ndarray:
        """
        Get the image of the world (MONO, uint8, grayscale)
        """
        s = self.scale
        img = self.world.img
        if s is not None:
            return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        else:
            return img.copy()

    @property
    def view(self) -> np.ndarray:
        """
        Get the image of the world (RGB, uint8, grayscale)
        """
        return np.stack([self.grayscale] * 3, axis=-1)

    def world_pos(self, p: Point[int]) -> Point[float]:
        """
        Get world location given the pixel coordinates on the view
        """
        return self.world.world_pos(p, scale=self.view_scale)

    def pixel_pos(self, p: Point[float]) -> Point[int]:
        """
        Get location on the dpi_scaled view given the world coordinates in meters
        """
        return self.world.pixel_pos(p, scale=self.scale)
