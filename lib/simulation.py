# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .arguments import register_arguments, Argument
from .util import tuple_of
from .geometry import Point


def point(s: str) -> Point[float]:
    return Point(*tuple_of(float)(s), type=float)


register_arguments(
    src=Argument(type=point, required=False, help="Simulation Starting Location"),
    dst=Argument(type=point, required=False, help="Simulation Destination"),
    radius=Argument(
        "-r",
        type=float,
        required=False,
        help="Collision radius for the robot, in meters",
    ),
    threshold=Argument(
        "-t",
        type=float,
        required=False,
        help="Threshold distance near the target",
    ),
    max_travel=Argument(
        "-M",
        type=float,
        required=False,
        help="Max travel distance before aborting the simulation",
    ),
    step_length=Argument(
        type=float, required=False, help="Simulation step length, in meters"
    ),
    task_offset=Argument(
        type=point,
        required=False,
        help="Offset both the src and dst coordinates, "
        "with +y being the unit vector pointing from src to dst",
    ),
    record=Argument(
        action="store_true",
        help="Record simulation output to a video file",
    ),
)

# ==============================================================================
# End argument injection
# ==============================================================================
from dataclasses import dataclass, field
from math import pi

from .util import ownAttributes, repeat
from .world import World
from .output import Output
from .visualization import Visualization


@ownAttributes
@dataclass(frozen=False)
class SimulationBase:
    world: World
    debug: bool = False

    src: Point[float] = None
    dst: Point[float] = None

    radius: float | None = None
    threshold: float | None = None
    max_travel: float | None = None
    step_length: float | None = None
    task_offset: Point[float] | None = None
    record: bool = False

    out: Output = field(init=False)
    vis: Visualization = field(init=False)

    def __post_init__(self):
        if getattr(self, "out", None) is None:
            self.out = Output(**self.world.meta)
        if getattr(self, "vis", None) is None:
            self.vis = Visualization(**self.world.meta)
        if self.radius is None:
            self.radius = self.world.res * 4
        if self.threshold is None:
            self.threshold = self.radius * 2
        if self.step_length is None:
            self.step_length = self.world.res * 4
        if self.src is None or self.dst is None:
            self.src, self.dst = self.prompt_src_dst
        # SRC-DST offset
        if self.task_offset is not None:
            offset = self.task_offset
            y = self.dst - self.src
            y /= y.norm
            x = Point.Angular(y.angle - 0.5 * pi)
            delta = x * offset.x + y * offset.y
            self.src += delta
            self.dst += delta

    @property
    def prompt_src_dst(self):
        import cv2, sys

        vis = self.vis
        src_pos, dst_pos = self.src, self.dst
        src_pixel = None if self.src is None else vis.pixel_pos(self.src)
        dst_pixel = None if self.dst is None else vis.pixel_pos(self.dst)

        direct_navigable: bool = False

        def render(cursor: Point[int] | None = None, wait_key: int | None = None):
            img = vis.view
            if cursor:
                if src_pixel is None:
                    vis.draw_src(img, cursor, (192, 192, 192))
                else:
                    vis.draw_dst(img, cursor, (192, 192, 192))
            if src_pixel:
                vis.draw_src(img, src_pixel)
            if dst_pixel:
                vis.draw_dst(img, dst_pixel)
            # Gray preview line
            if cursor and src_pixel:
                vis.line(img, src_pixel, cursor, (128, 128, 128))
            # Direct connection line
            if src_pixel and dst_pixel:
                color = (0, 192, 0) if direct_navigable else (0, 0, 255)
                vis.line(img, src_pixel, dst_pixel, color)
            vis.show(img, wait_key=wait_key)

        def onMouse(event, x, y, flags, _: None):
            nonlocal src_pixel, src_pos, dst_pixel, dst_pos, direct_navigable
            p = Point(x, y, type=int)
            cursor = p * vis.dpi_scale
            if event == cv2.EVENT_LBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
            ):
                if src_pixel is None:
                    src_pixel = cursor
                    src_pos = vis.world_pos(p)
                else:
                    dst_pixel = cursor
                    dst_pos = vis.world_pos(p)
                    direct_navigable = vis.world.checkLine(
                        src_pos, dst_pos, self.radius
                    )
            render(cursor, None)

        render()
        cv2.setMouseCallback(vis.handle, onMouse)

        for keycode in repeat(cv2.waitKey, 10):
            match keycode:
                case 8 | 127:  # Backspace
                    if dst_pixel is not None:
                        dst_pixel = None
                    elif src_pixel is not None:
                        src_pixel = None
                    render()
                case 13:  # Enter
                    if src_pixel is not None and dst_pixel is not None:
                        break
                case _:
                    if keycode == ord("q") or keycode == 27:
                        sys.exit(0)
                    elif keycode >= 0:
                        print(f"Unknown keycode: {keycode}")
        cv2.setMouseCallback(vis.handle, lambda *args: None)
        return src_pos, dst_pos

    @property
    def prompt_initial_heading(self):
        import cv2, sys, math

        vis = self.vis
        src_pos, dst_pos = self.src, self.dst
        src_pixel = None if self.src is None else vis.pixel_pos(self.src)
        dst_pixel = None if self.dst is None else vis.pixel_pos(self.dst)
        h, w = vis.grayscale.shape
        R = math.ceil(math.sqrt(h**2 + w**2))

        def render(cursor: Point[int] | None = None, navigable: bool = False):
            img = vis.view
            if cursor:
                if src_pixel is None:
                    vis.draw_src(img, cursor, (192, 192, 192))
                else:
                    vis.draw_dst(img, cursor, (192, 192, 192))
            if src_pixel:
                vis.draw_src(img, src_pixel)
            if dst_pixel:
                vis.draw_dst(img, dst_pixel)
            # Gray preview line
            if cursor and src_pixel:
                vis.line(img, src_pixel, cursor, (128, 128, 128))
            # Direct connection line
            if src_pixel and dst_pixel:
                color = (0, 192, 0) if navigable else (0, 0, 255)
                vis.line(img, src_pixel, dst_pixel, color)
            vis.show(img)

        def onMouse(event, x, y, flags, _: None):
            nonlocal src_pixel, src_pos, dst_pixel, dst_pos
            p = Point(x, y, type=int)
            cursor = p * vis.dpi_scale
            if event == cv2.EVENT_LBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
            ):
                navigable = False
                if src_pixel is None:
                    src_pixel = cursor
                    src_pos = vis.world_pos(p)
                else:
                    dst_pixel = cursor
                    dst_pos = vis.world_pos(p)
                    navigable = vis.world.checkLine(src_pos, dst_pos, self.radius)
            render(cursor, navigable)

        render()
        cv2.setMouseCallback(vis.handle, onMouse)

        for keycode in repeat(cv2.waitKey, 10):
            match keycode:
                case 8 | 127:  # Backspace
                    if dst_pixel is not None:
                        dst_pixel = None
                    elif src_pixel is not None:
                        src_pixel = None
                    render()
                case 13:  # Enter
                    if src_pixel is not None and dst_pixel is not None:
                        break
                case _:
                    if keycode == ord("q") or keycode == 27:
                        sys.exit(0)
                    elif keycode >= 0:
                        print(f"Unknown keycode: {keycode}")
        cv2.setMouseCallback(vis.handle, lambda *args: None)
        return src_pos, dst_pos


if __name__ == "__main__":
    from .arguments import parse

    SimulationBase(**parse())
