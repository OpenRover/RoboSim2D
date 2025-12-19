#!/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Iterable
from argparse import ArgumentParser
from pathlib import Path
from sys import stdin, stderr, exit
from yaml import safe_load
from json import dump

manifest = safe_load(stdin.read())
SRC: dict[str, tuple[float, float]] = manifest["SRC"]
DST: dict[str, tuple[float, float]] = manifest["DST"]

parser = ArgumentParser()
parser.add_argument("root", type=Path, nargs=1)
parser.add_argument(
    "--base", type=str, default="base", help="Base name for the output files"
)
root: Path = parser.parse_args().root[0]
base = str(parser.parse_args().base)

if not root.is_dir():
    print(f"{dir} is not a directory", file=stderr)
    exit(1)


def dirs(d: Path):
    return (x for x in d.iterdir() if x.is_dir())


def files(d: Path):
    return (x for x in d.iterdir() if x.is_file())


from lib.geometry import Point
from simulation import OffsetPath
from itertools import pairwise


class Trajectory(list[Point[float]]):

    def __init__(
        self,
        data: Iterable[Point[float]] = [],
        /,
        meta: dict[str, any] = {},
    ):
        super().__init__(data)
        self.meta: dict[str, any] = meta

    @classmethod
    def parse(cls, lines: Iterable[str], **kwargs):
        data = list[Point[float]]()
        meta = list[str]()
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                meta.append(line[1:].strip())
            else:
                x, y, *_ = map(float, line.split(","))
                data.append(Point(x, y, type=float))
        meta = safe_load("\n".join(meta)) | kwargs.get("meta", {})
        return cls(data, meta=meta, **kwargs)

    def u_turn_index(self, delta=0.01):
        skip = 0
        for i in range(1, len(self) - 1):
            if skip > 0:
                skip -= 1
                continue
            if (self[i - 1] - self[i + 1]).norm < delta:
                skip = 1
                yield i
                continue

    def __getitem__(self, s: slice | int):
        if type(s) is slice:
            return Trajectory(super().__getitem__(s), meta=self.meta)
        else:
            return super().__getitem__(s)

    def __str__(self):
        import numpy as np
        from base64 import b64encode

        data = [(a.x, a.y, (b - a).angle) for a, b in pairwise(self)]
        if len(data) > 0:
            end = (*self[-1], data[-1][2])
            array = np.array([*data, end], dtype=np.float32)
        else:
            array = np.array([], dtype=np.float32)
        return b64encode(array.tobytes()).decode("ascii")


offset = dict(x=0, y=0, r=0)
o: float = 0.1

watch_list = [
    # List of runs to visualize
    # "C-NE/Bug2R"
    # "SE-NE/Bug1R"
    # "C-SE/Bug1L"
]

offset_path_config = dict(
    look_back=1.0, look_ahead=1.0, step_length=0.1, resolution=0.02
)


def debug_offset(t, o):
    if not OffsetPath.run(
        OffsetPath(
            t,
            o,
            # debug=True,
            visualize=True,
            no_wait=False,
            **offset_path_config,
        )
    ):
        exit(1)


def aggregate(*runs: Path):
    trajectories = {}
    for run in reversed(runs):
        try:
            ID = run.stem
            prefix = f"{run.parent.name}/{ID}"
            print(f"Aggregating {prefix} (results/{prefix}.txt)", file=stderr)
            t = Trajectory.parse(run.open("rt"))
            visualize = prefix in watch_list
            if len(watch_list) and not visualize:
                continue
            elif visualize:
                print(len(t))
            t1 = []
            if ID.startswith("Bug1"):
                idx, *_ = list(t.u_turn_index()) + [None]
                if idx is not None:
                    t, t1 = t[: idx - 2], t[idx + 4 :]
            match ID:
                case "Bug0L" | "Bug1R":
                    if visualize:
                        debug_offset(t, +o)
                    t = Trajectory(OffsetPath(t, +o, **offset_path_config), meta=t.meta)
                case "Bug0R" | "Bug1L":
                    if visualize:
                        debug_offset(t, -o)
                    t = Trajectory(OffsetPath(t, -o, **offset_path_config), meta=t.meta)
            t.extend(t1)
            success = "abort" not in t.meta
            if run.stem[-5:-1] == "Bug2" and not success:
                r = len(t) - int(0.6 / offset_path_config["step_length"])
            else:
                r = len(t)
            stride = [0.1, 0, 0.1][int(ID[-2])]
            trajectories[run.stem] = dict(
                id=run.stem,
                success=success,
                range=r,
                stride=stride,
                offset=offset,
                color=["red", "green", "blue"][int(run.stem[-2])],
                buffer=str(t),
            )
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print(f"Error processing {run}: {e}", file=stderr)
            raise e
    return trajectories


for d in dirs(root):
    runs = [f for f in files(d) if f.suffix == ".txt" and f.stem.startswith("Bug")]
    runs.sort(key=lambda p: p.stem)
    S, T = d.name.split("-")
    meta = dict(base=base, type="sim", start_pos=SRC[S], target_pos=DST[T])
    with (d.parent / (d.name + "-Bug-L.json")).open("wt") as f:
        trj = aggregate(*(r for r in runs if r.stem.endswith("L")))
        dump(
            meta | dict(trajectories=trj),
            f,
            indent="\t",
        )
    with (d.parent / (d.name + "-Bug-R.json")).open("wt") as f:
        trj = aggregate(*(r for r in runs if r.stem.endswith("R")))
        dump(
            meta | dict(trajectories=trj),
            f,
            indent="\t",
        )
