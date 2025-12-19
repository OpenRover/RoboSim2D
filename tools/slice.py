import sys, cv2, numpy as np

try:
    from lib.visualization import Visualization
    from lib.output import Output
    from lib.arguments import parse
    from lib.geometry import Point
except ImportError:
    from ..lib.visualization import Visualization
    from ..lib.output import Output
    from ..lib.arguments import parse
    from ..lib.geometry import Point

args = parse()
vis = Visualization(**args)
out = Output(**args)
img_path = out(suffix="png")
if vis.visualize:
    s = vis.scale
    if vis.raw_slice is not None:
        x, y, w, h = vis.raw_slice
    else:
        x, y = 0, 0
        w, h = Point(*(Point(vis.world.w, vis.world.h) / vis.world.res), type=int)

    def render():
        view = vis.grayscale.astype(np.float32) / 255.0
        mask = np.ones_like(view) * 0.5
        mask_slice = vis.slice(mask)
        mask_slice[:, :] = 1.0
        view = np.stack([view * mask, view * mask, view], axis=-1)
        caption = ", ".join([str(_).ljust(3) for _ in (x, y, w, h)])
        vis.caption(view, caption, fg=(0, 0, 0), bg=[255] * 3)
        return vis.show(view, scale=1.0)

    drag_origin: Point[int] | None = None

    def onMouse(event, cx, cy, flags, _: None):
        global x, y, w, h, drag_origin
        cx = int(cx / s)
        cy = int(cy / s)
        match event:
            case cv2.EVENT_LBUTTONDOWN:
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    drag_origin = Point(cx, cy)
                else:
                    x, y = cx, cy
                    w, h = 1, 1
            case cv2.EVENT_MBUTTONDOWN:
                drag_origin = Point(cx, cy)
            case cv2.EVENT_MOUSEMOVE:
                if flags == cv2.EVENT_FLAG_LBUTTON:
                    w = max(1, cx - x)
                    h = max(1, cy - y)
                if flags in (
                    cv2.EVENT_FLAG_MBUTTON,
                    cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_CTRLKEY,
                ):
                    if drag_origin is not None:
                        dx, dy = cx - drag_origin.x, cy - drag_origin.y
                        x, y = x + dx, y + dy
                    drag_origin = Point(cx, cy)
            case _:
                return
        vis.raw_slice = (x, y, w, h)
        render()

    handle, _, _ = render()
    cv2.setMouseCallback(handle, onMouse)
    while cv2.waitKey(1) < 0:
        pass
if vis.raw_slice is not None:
    print("--slice=", ",".join(str(_) for _ in vis.raw_slice), sep="", file=sys.stderr)
if img_path is not None:
    vis.saveImg(img_path, vis.grayscale)
