#!/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser
from pathlib import Path
from sys import stdin, stderr, exit
from yaml import safe_load
import numpy as np
from tqdm import tqdm

try:
    from regression import Regression
except ImportError:
    from .regression import Regression

meta = safe_load(stdin.read())
ours = list[float]()
ours_total: int = 0
ours_failure: int = 0

parser = ArgumentParser()
parser.add_argument("root", type=str, nargs="?", default="results")
parser.add_argument(
    "--load", type=str, default=None, help="Load regression model from file"
)
parser.add_argument("--latex", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()
root: Path = Path(args.root)
load_model = str(args.load) if args.load is not None else None
should_write_latex = bool(args.latex)
should_plot = bool(args.plot)
should_animate = bool(args.animate)

from dataclasses import dataclass


@dataclass
class AdditionalData:
    sr: float
    spl: float
    marker: str = "o"

    @property
    def pl(self):
        return self.spl / self.sr


additional = dict(
    VLnav0=AdditionalData(sr=0.504, spl=0.21, marker="^"),
    VLnav=AdditionalData(sr=0.332, spl=0.136),
    GOAT0=AdditionalData(sr=0.83, spl=0.64, marker="^"),
    GOAT=AdditionalData(sr=0.61, spl=0.19),
)


if not root.is_dir():
    print(f"{dir} is not a directory", file=stderr)
    exit(1)


def dirs(d: Path):
    return (x for x in d.iterdir() if x.is_dir())


class BugAlgorithm:
    def __init__(self):
        self.failures = 0
        self.data = []

    def __call__(self, p: Path, baseline: float):
        with p.open("rt") as f:
            content = (l[1:] for l in f if l.startswith("#"))
            meta = safe_load("\n".join(content))
        if "abort" in meta or "travel" not in meta:
            self.failures += 1
        else:
            self.data.append(meta["travel"] / baseline)

    def stat(self):
        v = 1 / np.array(sorted(self.data))
        x = len(v) / (len(v) + self.failures)
        return v.mean(), v.std(), x

    @property
    def result(self):
        v = 1 / np.array(sorted(self.data))
        x = len(v) / (len(v) + self.failures)
        avg = np.mean(v)
        std = np.std(v) / 2
        return tuple(map(float, (x, avg, std)))


class BatchSampler:
    def __init__(self):
        self.failures: int = 0
        self.data: list[float] = []

    def __call__(self, p: Path, baseline: float):
        with p.open("rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if "FAIL" in line.upper():
                    self.failures += 1
                    continue
                # Let ValueError propagate
                (d,) = list(map(float, line.split(",")))
                self.data.append(d / baseline)

    def stat(self, success_rate: float):
        i = int(len(self.data) * success_rate)
        v = 1 / np.array(list(sorted(self.data))[:i])
        return v.mean(), v.std(), success_rate

    @property
    def result(self):
        data = 1 / np.array(sorted(self.data))
        total = len(data) + self.failures
        x = list[float]()  # Success rate
        avg = list[float]()
        std = list[float]()
        for i in range(1, len(data) + 1):
            v = data[:i]
            success_rate = i / total
            x.append(success_rate)
            avg.append(np.mean(v))
            std.append(np.std(v * success_rate))
        return np.array(x), np.array(avg), np.array(std)


class WaveFront:
    def __init__(self):
        self.T = list[float]()  # Travel Distance
        self.P = list[float]()  # Cumulative Probability
        self.DP = list[float]()  # Transient Probability

        self.total: int = 0
        self.success_rate: float = 1.0

    def __call__(self, p: Path):
        baseline: float | None = None
        self.total += 1
        with p.open("rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                # Let ValueError propagate
                t, p, dp = list(map(float, line.split(",")))
                if p <= 1e-3:
                    continue
                elif baseline is None:
                    baseline = t
                self.T.append(t / baseline)
                self.P.append(p)
                self.DP.append(dp)
            self.success_rate = min(self.success_rate, p)
        return baseline

    def stat(self, success_rate: float):
        T, P, DP = (1 / np.array(self.T)), np.array(self.P), np.array(self.DP)
        indices = P <= success_rate
        T, DP = T[indices], DP[indices]
        return (
            (T * DP).sum() / DP.sum(),
            np.std(len(DP) * T * DP / DP.sum()),
            success_rate,
        )

    @property
    def result(self):
        T, P, DP = 1 / np.array(self.T), np.array(self.P), np.array(self.DP)
        DP *= self.total
        x = list[float]([0.0])  # Success rate
        avg = list[float]([1.0])
        std = list[float]([0.0])
        for l in np.linspace(0.01, 1, 100):
            sel = P <= l
            if len(sel) == 0:
                continue
            t, dp = T[sel], DP[sel]
            dp /= dp.sum()
            x.append(l)
            avg.append(np.sum(t * dp))
            std.append(np.std(t * dp))
        sel = P < 0.01
        t, dp = T[sel], DP[sel]
        return np.array(x), np.array(avg), np.array(std)


RW = BatchSampler()
WB = BatchSampler()
WF = WaveFront()
Bug0, Bug1, Bug2 = BugAlgorithm(), BugAlgorithm(), BugAlgorithm()

for d in dirs(root):
    baseline = WF(d / "wavefront.txt")
    if baseline is None:
        raise ValueError(f"Wavefront @{d.name} has no baseline")
    # print(f"# Baseline of {d} = {baseline:.4f}")
    RW(d / "RandomWalk.list", baseline)
    WB(d / "WallBounce.list", baseline)
    Bug0(d / "Bug0L.txt", baseline)
    Bug0(d / "Bug0R.txt", baseline)
    Bug1(d / "Bug1L.txt", baseline)
    Bug1(d / "Bug1R.txt", baseline)
    Bug2(d / "Bug2L.txt", baseline)
    Bug2(d / "Bug2R.txt", baseline)
    # Our results
    v = meta[d.name]
    ours_total += 3
    if len(v) < 3:
        ours_failure += 3 - len(v)
    ours.extend((baseline / t for t in meta[d.name]))


ours_sr = (ours_total - ours_failure) / ours_total

# Train Regression Model
from torch import tensor, from_numpy
from torch.nn import Parameter

model = Regression()

if load_model is None:
    with tqdm(
        total=40000, desc=f"Training Regression Model (loss={0:.8f})", leave=False
    ) as progress:
        R, L, _ = RW.result
        with open("rw-curve.local.txt", "w") as f:
            for x, y in zip(R, L):
                f.write(f"{x} {y}\n")
        E = tensor([0.0] * len(R) + [1.0], requires_grad=False)
        R = tensor(list(R) + [1.0], requires_grad=False)
        L = tensor(list(L) + [1.0], requires_grad=False)
        for i, loss in enumerate(model.train(L, R, E, learning_rate=1e-4)):
            if i % 100 == 0:
                progress.set_description(f"Training Regression Model (loss={loss:.8f})")
                progress.n = i
                progress.refresh()
            if i >= 40000:
                break
    with open("regression.local.yaml", "w") as f:
        from yaml import safe_dump

        safe_dump({k: v.item() for k, v in model.__dict__().items()}, f)
else:
    with open(load_model, "rt") as f:
        from yaml import safe_load

        params: dict[str, float] = safe_load(f)
        for k, v in params.items():
            p = getattr(model, k, None)
            if not isinstance(p, Parameter):
                raise ValueError(f"Invalid parameter key: {k}")
            p.data = tensor(v, dtype=p.dtype)


if not any([should_write_latex, should_plot, should_animate]):
    parser.print_usage()
    parser.print_help()
    exit(1)

# DATA REPORTING
if should_write_latex:

    def pz(v: str, l: int):
        if len(v) == l:
            return "  " + v
        if len(v) > l:
            return v
        return "\\z" + v.rjust(l)

    def report(name: str, pl: float, std: float, sr: float, epsilon: float):
        print(name, "&")
        s = pz(f"{sr * 100:.2f}", 6)
        a = pz(f"{pl:.2f}", 5)
        b = pz(f"{std:.2f}", 4) if std is not None else None
        SPL = pz(f"{sr * pl:.2f}", 4)
        print(f"${s}\\%$", "&")
        if b is not None:
            print(f"${a}$\\small{{$\\pm {b}$}}", "&")
        else:
            print(f"${a}$", "&")
        print(f"${SPL}$", "&")
        if epsilon is not None:
            EPS = pz(f"{epsilon:.2f}", 4)
            print(f"${EPS}$", "&")
        else:
            print("--", "&")
        print("--", "\\\\")

    pl = float(np.mean(ours))
    std = float(np.std(ours))
    sr = ours_sr
    report("\\textbf{ClipRover\\,\\ding{72}}", pl, std, sr, model.E(pl, sr))
    print("\\hline")

    R, L, _ = RW.result
    eps = model.E(from_numpy(L), from_numpy(R)).numpy().mean()
    report("\\multirow{2}{*}{Random\\,Walk}", *RW.stat(0.5), eps)
    print("\\cline{2-3}")
    report("", *RW.stat(0.8), eps)
    print("\\hline")

    R, L, _ = WB.result
    eps = model.E(from_numpy(L), from_numpy(R)).numpy().mean()
    report("\\multirow{2}{*}{Wall\\,Bounce}", *WB.stat(0.5), eps)
    print("\\cline{2-3}")
    report("", *WB.stat(0.8), eps)
    print("\\hline")

    R, L, _ = WF.result
    eps = model.E(from_numpy(L), from_numpy(R)).numpy().mean()
    report("\\multirow{2}{*}{Wave\\,Front}", *WF.stat(0.5), eps)
    print("\\cline{2-3}")
    report("", *WF.stat(0.8), eps)

    print("\\Xhline{2\\arrayrulewidth}")

    R, L, _ = Bug0.result
    eps = model.E(L, R)
    report("Bug 0", *Bug0.stat(), eps)
    print("\\hline")
    R, L, _ = Bug1.result
    eps = model.E(L, R)
    report("Bug 1", *Bug1.stat(), eps)
    print("\\hline")
    R, L, _ = Bug2.result
    eps = model.E(L, R)
    report("Bug 2", *Bug2.stat(), eps)

    print("\\Xhline{2\\arrayrulewidth}")

    for name, data in additional.items():
        report(name, data.pl, None, data.sr, None)

    print("\\Xhline{2\\arrayrulewidth}")

# PLOTTING
if not should_plot and not should_animate:
    exit(0)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rc

rc("font", family="Times New Roman", size=16)


def init_figure():
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in ax.spines.values():
        s.set_edgecolor("#CCC")
        s.set_linewidth(1)
    ax.set_xlabel("Success Rate")
    ax.set_xticks(np.linspace(0, 100, 6), minor=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, color="#CCC", linewidth=1)
    # ax.set_xscale("log")
    ax.set_xlim(0, 100)

    ax.set_ylabel("Efficiency")
    # ax.set_yscale("log")
    ax.set_ylim(0, 1)
    return fig, ax


def EPL(epsilon: float):
    x0 = model.R(1.0, epsilon)
    L = tensor(np.linspace(x0, 1.0, 100), requires_grad=False)
    R = model.R(L, epsilon)
    return R.detach().numpy(), L.detach().numpy()


if should_plot:
    fig, ax = init_figure()
    x, y, e = RW.result
    ax.plot(100 * x, y, color="gray", linestyle="--")

    ax.fill_between(100 * x, y - e, y + e, alpha=0.2, color="gray")

    x, y, e = WB.result
    ax.plot(100 * x, y, color="blue", linestyle="--")
    ax.fill_between(100 * x, y - e, y + e, alpha=0.2, color="blue")

    x, y, e = WF.result
    ax.plot(100 * x, y, label="Wavefront", color="red", linestyle="-")
    ax.fill_between(100 * x, y - e, y + e, alpha=0.2, color="pink")

    def ebar(
        ax: plt.Axes,
        x,
        y,
        e,
        /,
        color: str,
        label: str | None = None,
        marker="D",
        markersize: int = 8,
        **kwargs,
    ):
        kwargs = dict(capsize=4, clip_on=False, zorder=10, color=color) | kwargs
        data, cap, bars = ax.errorbar([100 * x], [y], [e], **kwargs).lines
        data.set_marker(marker)
        data.set_markersize(markersize)
        for l in cap:
            l.set_linewidth(8)
            l.set_alpha(0.25)
        for l in bars:
            l.set_linewidth(2)
            l.set_alpha(0.25)
        if label is not None:
            ax.text(
                100 * x - 2,
                y,
                label,
                ha="right",
                va="center",
                fontsize=16,
                color=color,
                fontweight="bold",
            )

    ebar(ax, *Bug0.result, label="Bug0", color="blue")
    ebar(ax, *Bug1.result, label="Bug1", color="green")
    ebar(ax, *Bug2.result, label="Bug2", color="red")

    for name, data in additional.items():
        x, y = data.sr, data.pl
        ax.plot(
            100 * x,
            y,
            marker=data.marker,
            markersize=6,
            color="gray",
        )
        ax.text(
            100 * x - 2,
            y,
            name,
            ha="right",
            va="center",
            fontsize=16,
            color="gray",
            fontweight="bold",
        )

    # Our results
    v = ours
    x = ours_sr
    avg = np.mean(v)
    std = np.std(v) / 2

    ebar(ax, x, avg, std, color="black", marker="*", markersize=24)
    ax.text(
        100 * x - 4,
        avg,
        "ClipRover",
        ha="right",
        va="center",
        fontsize=16,
        color="black",
        fontweight="bold",
    )

    def steps(a, b, s):
        while a <= b:
            yield a
            a += s

    # Equipotential lines
    for epsilon in steps(0.0, 0.9, 0.1):
        X, Y = EPL(epsilon)
        ax.plot(100 * X, Y, color="gray", linestyle="--", linewidth=0.8)

    fig.savefig("summary.local.pdf")

    fig.show()
    plt.show()

# ========== ANIMATION ==========


def interpolate(X, Y, x0):
    for x1, x2, y1, y2 in zip(X, X[1:], Y, Y[1:]):
        if x1 <= x0 <= x2:
            return y1 + (y2 - y1) * (x0 - x1) / (x2 - x1)


def search(X: np.ndarray, Y: np.ndarray, y0):
    if min(Y) > y0:
        return X[0]
    if max(Y) < y0:
        return X[-1]
    for x1, x2, y1, y2 in zip(X, X[1:], Y, Y[1:]):
        if (y1 - y0) * (y2 - y1) > 0:
            return x1 + (x2 - x1) * (y0 - y1) / (y2 - y1)
    raise ValueError("No solution found")


if should_animate:
    import sys
    from pathlib import Path

    root = Path(__file__).parent.parent
    sys.path.append(str(root))

    from lib.video import Video
    from lib.util import fig2img

    outfile = Path("EQP.local.mp4")
    outfile.unlink(missing_ok=True)
    video = Video(outfile, 120)

    rc("text", usetex=True)

    fig, ax = init_figure()
    ax.patch.set_alpha(0)
    ax.grid(False)

    (line,) = ax.plot([], [], color="black", linestyle="--", linewidth=1)
    (dot,) = ax.plot([], [], color="black", marker="v", markersize=6, clip_on=False)
    label = None

    # Equipotential lines
    for epsilon in np.linspace(0, 1, 1200):
        X, Y = EPL(epsilon)
        line.set_data(100 * X, Y)
        x0 = max(X[0], 0.0)
        dot.set_data([100 * x0], [1.03])
        if label is not None:
            label.remove()
        label = ax.text(
            100 * x0 - 0.6,
            1.05,
            # "hello world",
            f"$\\varepsilon = {epsilon:.2f}$",
            ha="left",
            va="bottom",
            fontsize=16,
            color="black",
            clip_on=False,
        )
        # Find X value
        frame = fig2img(fig)
        video.write(frame)
        import cv2

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    video.release()
