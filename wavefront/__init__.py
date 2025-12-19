# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from lib.arguments import register_arguments, Argument

register_arguments(
    vanish_threshold=Argument(
        type=float,
        required=False,
        help="Termination condition for wavefront simulation.",
    )
)

import builtins, numpy as np, cv2
from lib.geometry import Point
from lib.util import dup, RollingAverage, fig2img
from lib.simulation import SimulationBase
from lib.arguments import auto_parse
from lib.video import Video


def steeper(y: np.ndarray, lv: int = 1):
    for _ in range(lv):
        y = 0.5 * np.sin((y - 0.5) * np.pi) + 0.5
        y = np.clip(y, 0, 1)

    return y


@auto_parse()
class WaveFront(SimulationBase):
    dtype = np.float64
    # Termination threshold - the remaining probability
    vanish_threshold: float = 0.01  # 1%

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vanish_threshold = self.world.meta.get(
            "vanish_threshold", self.vanish_threshold
        )

    def __post_init__(self):
        super().__post_init__()
        self.micro_step = min(self.world.res / 4, self.step_length)
        occupancy = self.world.padMap(self.radius)
        h, w = occupancy.shape
        # The base field, 1.0 for free space, 0.0 for obstacles
        # `field * base` filters out probability distributions in obstacles
        self.base = 1.0 - occupancy.astype(self.dtype)
        # Discrete cartesian coordinates (per-cell)
        Y, X = np.mgrid[:h, :w].astype(self.dtype) * self.world.res
        ox, oy = self.world.world_pos(Point(0, 0))
        self.X = +X + ox
        self.Y = -Y + oy
        # source_field:
        #   A gaussian distribution around the source.
        #   It's sigma equals the radius of the robot.
        self.source_field = self.norm(self.gaussian(self.src, self.radius / 2))
        # drain_field:
        #   A circle mask centered at the destination.
        #   It's radius equals the threshold distance.
        x1, y1, r = self.X - self.dst.x, self.Y - self.dst.y, self.threshold
        r += self.world.res / 2
        self.drain_mask = np.array(x1**2 + y1**2 <= r**2, bool)
        self.drain_field = np.array(self.drain_mask, self.dtype)

    @property
    def mask(self):
        r = self.vis.scale
        return (
            cv2.resize(self.base, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
            > 0.5
        )

    @property
    def view(self) -> np.ndarray:
        diff = (self.base <= 0.5) ^ self.world.occupancy
        r = self.vis.scale
        diff = (
            cv2.resize(diff * 155, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
            >= 128
        )
        view = self.vis.view
        view[diff, :] = 192
        return view

    def norm(self, field: np.ndarray) -> np.ndarray:
        f = np.maximum(field * self.base, 0.0).astype(self.dtype)
        return (f / f.sum()).astype(self.dtype)

    def gaussian(self, origin: Point[float], sigma) -> np.ndarray:
        X, Y = self.X - origin.x, self.Y - origin.y
        return np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(self.dtype)

    def render(
        self,
        bg: np.ndarray,
        field: np.ndarray,
        color: list[float] = [0.0, 0.0, 1.0],
        invert=False,
    ) -> np.ndarray:
        # Requires float array of range [0.0, 1.0]
        if bg.max() > 1.0 or bg.min() < 0.0:
            raise ValueError(
                f"Background must be a float array of range [0.0, 1.0], got {[bg.min(), bg.max()]}"
            )
        s = self.vis.scale
        if invert:
            field = 1.0 - field
        field = np.clip(field, 0.0, 1.0)
        field = cv2.resize(field, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        field = np.stack([field] * 3, axis=-1)
        f0, f1 = 1.0 - field, field
        fg = np.ones_like(bg) * color
        return np.clip(bg * f0 + fg * f1, 0, 1)

    def heatmap(
        self,
        field: np.ndarray,
        c0: list[float] = [0.0, 0.0, 1.0],
        c1: list[float] = [1.0, 0.5, 0.0],
        invert: bool = False,
    ) -> np.ndarray:
        # Requires float array of range [0.0, 1.0]
        s = self.vis.scale
        if invert:
            field = 1.0 - field
        field = np.clip(field, 0.0, 1.0)
        field = cv2.resize(field, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        field = np.stack([field] * 3, axis=-1)
        f0, f1 = 1.0 - field, field
        bg = np.ones_like(field) * c1
        fg = np.ones_like(field) * c0
        return bg * f0 + fg * f1

    def tick(self, u0: np.ndarray, du: np.ndarray) -> np.ndarray:
        """
        input: u0 - the prob. field at time t_{n}
        output: u1 - the prob. field at time t_{n+1}
        """
        raise NotImplementedError

    class Session:

        started: bool = False
        finished: bool = False

        def __init__(self, wf: "WaveFront"):
            self.wf = wf
            self.u0: np.ndarray = wf.source_field.copy()
            self.p0: float = self.u0.sum()
            self.du = np.zeros_like(self.u0)

        def __next__(self):
            if self.finished:
                raise StopIteration
            if self.started:
                # Apply next tick
                u1 = self.wf.tick(self.u0, self.du) * self.wf.base
                # Normalize probability field
                u1: np.ndarray = np.clip(u1, 0.0, None)
                # u1 *= self.p0 / u1.sum()
                # Compute drain probability
                dp = np.sum(u1[self.wf.drain_mask])
                u1[self.wf.drain_mask] = 0.0
                # Compute probability shift
                p1 = self.p0 - dp
                # dp = self.p0 - p1
                du = u1 - self.u0
                # Constrain total probability
                # u1 *= p1 / u1.sum()
                # print(f"prob. shift = {(1.0 - p1 / u1.sum()) * 100:.4f}%")
                # Termination conditions
                if p1 < 0:
                    raise StopIteration
                if p1 < self.wf.vanish_threshold:
                    self.finished = True
                # dp = self.p0 - p1
                # Prepare for next rick
                self.u0, self.p0, self.du = u1, p1, du
            else:
                self.started = True
                dp = 0.0
            return self.u0, 1.0 - self.p0, dp

    def __iter__(self):
        return WaveFront.Session(self)

    @staticmethod
    def run(wf: "WaveFront"):
        name = wf.__class__.__name__
        out_list = wf.out(name, suffix="txt")
        out_img = wf.out(name, suffix="png")
        out_fig = wf.out(name, suffix="pdf")
        recording_wave = wf.out(name, suffix="mp4")
        recording_cumulative = wf.out(name + "-cumulative", suffix="mp4")
        recording_transient = wf.out(name + "-transient", suffix="mp4")

        if out_list is not None:
            trj_list_file = open(out_list, "w")
            print = dup(trj_list_file)
        else:
            print = builtins.print

        if wf.record:
            if None in (recording_wave, recording_cumulative, recording_transient):
                raise ValueError("Recording is enabled but prefix is not specified")
            else:
                video_wave = Video(recording_wave)
                video_cumulative = Video(recording_cumulative)
                video_transient = Video(recording_transient)
        else:
            video_wave = None
            video_cumulative = None
            video_transient = None

        U = np.zeros_like(wf.base)
        P = list[float]()
        DP = list[float]()
        VIS_DP = list[float]()
        T = list[float]()
        t = 0.0
        dt = wf.micro_step
        t_report = 0.0
        dp_accumulate = 0.0

        if wf.vis.visualize or wf.record:
            import matplotlib.pyplot as plt

            from matplotlib import rc

            rc("font", family="Times New Roman", size=16)

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig1.tight_layout(h_pad=6.0, w_pad=3.0)
            fig2.tight_layout(h_pad=6.0, w_pad=3.0)
            bkg = [183 / 255] * 3
            fig1.patch.set_facecolor(bkg)
            fig2.patch.set_facecolor(bkg)
            ax1.set_facecolor(bkg)
            ax2.set_facecolor(bkg)
            ax1.set_ylim(0, 1)
            # ax1.set_title("Cumulative Probability")
            ax1.set_xlabel("Travel Distance (m)")
            # ax2.set_title("Transient Probability")
            ax2.set_xlabel("Travel Distance (m)")
            (p_plot,) = ax1.plot([], [], "k-", lw=2)
            (dp_plot,) = ax2.plot([], [], "k-", lw=2)

            bg = wf.view.astype(wf.dtype) / 255.0

            roll: RollingAverage | None = None

            def render(u: np.ndarray, p: float):
                nonlocal roll
                if roll is None:
                    roll = RollingAverage(0.8, value=u.max())
                    k = 1 / roll.value
                else:
                    k = 1 / roll(u.max())
                m = wf.render(bg, u * k, color=[0.0, 0.0, 1.0])
                m = (m * 255.0).astype(np.uint8)
                wf.vis.draw_src(m, wf.vis.pixel_pos(wf.src))
                wf.vis.draw_dst(m, wf.vis.pixel_pos(wf.dst))
                caption = f"Travel {t:.2f}m | Progress {p * 100.0:.2f}%"
                wf.vis.caption(m, caption, fg=(0, 0, 0), bg=(255, 255, 255))
                _, frame, key = wf.vis.show(m)
                if video_wave is not None:
                    video_wave.write(frame)
                # T-P Plot
                nonlocal p_plot, dp_plot
                ax1.set_xlim(0, max(t, 10.0))
                ax2.set_xlim(0, max(t, 10.0))
                ax2.set_ylim(0, max(max(VIS_DP), 0.1))
                p_plot.set_data(T, P)
                dp_plot.set_data(T, VIS_DP)
                dw = min(1.0, len(VIS_DP) * wf.step_length / 200.0)
                dp_plot.set_linewidth(2.0 - dw)
                img_cumulative = fig2img(fig1)
                img_transient = fig2img(fig2)
                wf.vis.show(img_cumulative, "cumulative", 2.0)
                wf.vis.show(img_transient, "transient", 2.0)
                if video_cumulative is not None:
                    video_cumulative.write(img_cumulative)
                if video_transient is not None:
                    video_transient.write(img_transient)
                return key

        for u, p, dp in wf:
            t1 = t + dt
            if wf.max_travel is not None and t1 > wf.max_travel:
                break
            U += u
            if t1 >= t_report:
                T.append(t)
                P.append(p)
                DP.append(dp_accumulate)
                VIS_DP.append(100 * dp_accumulate / wf.step_length)
                print(f"{t1:.2f}, {p:.4f}, {dp_accumulate:.8f}", flush=True)
                dp_accumulate = 0.0
                t_report += wf.step_length
                # Visualization
                if wf.vis.visualize or wf.record:
                    key = render(u, p)
                    if key >= 0:
                        break
            else:
                dp_accumulate += dp
            t = t1
        else:
            print(f"{t:.2f}, {p:.4f}", dp_accumulate, sep=", ")
        x = np.array(T)
        y = np.array(DP)
        c = np.sum(y)
        print(f"# coverage: {c:.4f}")
        mean = np.sum(x * y) / c
        print(f"# travel  : {mean:.2f}")
        variance = np.sum((x - mean) ** 2 * y)
        std = np.sqrt(variance)
        print(f"# std     : {std:.2f}")
        print(f"# src     : {wf.src}")
        print(f"# dst     : {wf.dst}")
        print(f"# distance: {(wf.src - wf.dst).norm:.2f}")
        # u = U / float(n + 1)
        amp: float = 1.0
        bg = wf.vis.view.astype(wf.dtype) / 255.0

        def vis():
            u = U * amp / U.max()
            v = bg.copy()
            v[wf.mask] = wf.heatmap(u)[wf.mask]
            v = (v * 255.0).astype(np.uint8)
            caption = f"Expected Travel: {mean:.2f} +/- {std:.2f}m"
            wf.vis.draw_src(v, wf.vis.pixel_pos(wf.src))
            wf.vis.draw_dst(v, wf.vis.pixel_pos(wf.dst))
            wf.vis.caption(v, caption, fg=(0, 0, 0), bg=(255, 255, 255))
            return v

        def renderVis(_amp: float | None = None):
            nonlocal amp
            if _amp is not None and _amp >= 1.0:
                amp = _amp
            return wf.vis.show(vis(), "probability distribution")

        if wf.vis.visualize and not wf.vis.no_wait:
            render(u, p)
            handle, _, _ = renderVis()
            cv2.createTrackbar(
                "AMP", handle, int(amp * 10), 100, lambda v: renderVis(v / 10.0)
            )
            try:
                while True:
                    if cv2.waitKey(1) > 0:
                        break
            except:
                pass

        if out_img is not None:
            wf.vis.saveImg(out_img, vis())

        if out_fig is not None:

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(T, P, "r-", lw=2)
            ax.set_xlim(0, max(t, 1.0))
            ax.set_ylim(0, max(max(P), 1e-10))
            fig.savefig(out_fig, transparent=True)
