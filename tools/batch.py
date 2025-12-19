#!/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from os import cpu_count
from sys import stdin, stdout, stderr, executable
from yaml import safe_load as parse
from typing import Iterable, Callable, Any
from functools import wraps
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Process, Pool, Manager
from subprocess import Popen, PIPE
from time import time
from tqdm import tqdm

WATCH_LIST: list[tuple[str, str, str]] = [
    # ("SW", "C", "Bug1L"),
    # ("SW", "NE", "Bug1L"),
]

# SLICE = {"slice": "19,7,320,360"}


def cmdlineFlag(s: str) -> str:
    import re

    return re.sub(r"([a-z])([A-Z])", r"\1-\2", s).lower().replace("_", "-")


def kw2cmdline(**kwargs):
    for k, v in kwargs.items():
        if type(v) is bool:
            if v:
                yield f"--{cmdlineFlag(k)}"
        else:
            yield f"--{cmdlineFlag(k)}={v}"


class Python:
    def __init__(self, module: str, **env):
        self.module = module
        self.env = env
        self.arguments = tuple[str]()

    def __str__(self):
        return " ".join((Path(executable).name, *self.arguments))

    def __repr__(self):
        return str(self)

    def __call__(self, *args, **kwargs):
        args = self.arguments = "-m", self.module, *args, *kw2cmdline(**kwargs)
        self.proc = Popen((executable,) + args, stdout=PIPE, env=self.env)
        self.stdout = self.proc.stdout
        self.pid = self.proc.pid
        return self

    def wait(self):
        return self.proc.wait()


def parse_outputs(stream: Iterable[bytes]) -> dict:
    from yaml import safe_load

    data = []
    for line in stream:
        try:
            line = line.decode("utf-8").strip()
            if line.startswith("#"):
                data.append(line[1:].strip())
        except:
            pass
    return safe_load("\n".join(data))


def format_message(desc: tuple[str, str, str], dt: float, meta: dict) -> str:
    src, dst, module = desc
    msg = [
        f"{src.rjust(2)}-{dst.ljust(2)}",
        module.ljust(10),
        f"{dt:.2f} s".rjust(7),
    ]
    try:
        if "travel" in meta:
            travel = f"{meta['travel']:.2f}".rjust(6)
        else:
            travel = "--.--".rjust(6)
        t_msg = f"travel {travel}"
        if "std" in meta:
            std = f"{meta['std']:.2f}".rjust(6)
            t_msg += f" +/- {std}"
        t_msg += " m"
        msg.append(t_msg)
        if "success_rate" in meta:
            val = meta["success_rate"]
            if val is None:
                val = "N/A"
            else:
                val = f"{val:.4f}"
            msg.append(f"success rate: {val}")
        if "abort" in meta:
            msg.append("aborted: " + meta["abort"])
    except Exception as e:
        msg.append(str(e))
    return " | ".join(msg)


def register(
    proc: Python | None = None, status: str | None = None, log=Path("exec.log")
):
    if proc is None:
        with log.open("wt") as f:
            f.write("")
        return
    entry = f"{str(proc.pid).rjust(8)} [{status}] {proc}\n"
    if not log.exists():
        with open("exec.log", "wt") as f:
            f.write(entry)
        return
    with log.open("rt+") as f:
        # Scan for previous registration
        lines = f.readlines()
        f.seek(0)
        flag = False
        for l in lines:
            try:
                pid, _ = l.strip().split(" ", 1)
                if int(pid) == proc.pid:
                    flag = True
                    f.write(entry)
                    continue
            except:
                pass
            f.write(l)
        if not flag:
            f.write(entry)
        f.truncate()


def exec(
    world: str,
    src: str,
    dst: str,
    module: str,
    kw: dict,
    filter: Callable | None = None,
    *filter_args,
):
    args = []
    triplet = src, dst, module.split(".")[-1]
    if triplet in WATCH_LIST:
        args += ["-v"]
    t0 = time()
    proc = Python(module)(world, *args, **kw)
    register(proc, " " * 6)
    print(proc, file=stderr)
    if filter:
        output = filter(proc.stdout, *filter_args)
    else:
        output = proc.stdout
    try:
        meta = parse_outputs(output)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        exit(1)

    t1 = time()
    dt = t1 - t0
    register(proc, f"{dt:.2f}".rjust(6))
    return triplet, dt, meta


def unpack_exec(args):
    return exec(*args)


type PList = dict[str, list[float]]
type Combs = list[tuple[str, str]]


def combs_preset(SRC: PList, DST: PList, combinations: Combs):
    for src, dst in combinations:
        if src not in SRC or dst not in DST:
            raise ValueError(f"Missing {src} or {dst} in batch configuration")
        yield src, SRC[src], dst, DST[dst]


def combs_shuffle(SRC: PList, DST: PList):
    for src, p0 in SRC.items():
        for dst, p1 in DST.items():
            if src == dst:
                continue
            yield src, p0, dst, p1


def combinations(
    world: str,
    SRC: PList,
    DST: PList,
    combinations: Combs | None = None,
    save: bool = False,
    **kw: dict[str, Any],
):
    if SRC is None or DST is None:
        raise ValueError("Missing SRC or DST in batch configuration")

    world_name = Path(world).name
    combs = (
        combs_preset(SRC, DST, combinations)
        if combinations
        else combs_shuffle(SRC, DST)
    )

    for src, p0, dst, p1 in combs:
        local_kw = dict(
            **kw,
            src=",".join(map(str, p0)),
            dst=",".join(map(str, p1)),
        )
        if save:
            local_kw = dict(prefix=f"results/{src}-{dst}/", **local_kw)
        yield world, src, dst, local_kw


def factory(f):
    @wraps(f)
    def outer(*args, **kwargs):

        def inner(world: str, src: str, dst: str, kw: dict):
            invoke_kw = kw | kwargs | dict(src=src, dst=dst)
            for module, _kw, *extras in f(*args, **invoke_kw):
                yield world, src, dst, module, {**kw, **_kw}, *extras

        inner.__name__ = f.__name__
        return inner

    return outer


@factory
def Bug(**kw):
    for n in (1, 2, 0):
        for d in "LR":
            kw = dict(radius=0.25)
            yield f"simulation.Bug{n}{d}", kw


@factory
def RandomWalk(N: int = 100, **kw):
    for n in range(N):
        kw = dict(radius=0.25, seed=n)
        yield f"simulation.RandomWalk", kw


@factory
def WallBounce(N: int = 180, /, **kw):
    from numpy import linspace

    for hdg in linspace(0, 360, N, endpoint=False):
        kw = dict(radius=0.25, heading=hdg)
        yield f"simulation.WallBounce", kw


@factory
def WaveFront(queue, prog, /, src: str, dst: str, maxTravel: float, **kw):
    kw = dict(radius=0.25, resolution=0.05, threshold=1, stepLength=1.0)
    filter = WaveFrontFilter
    desc = f"{src.rjust(2)}-{dst.ljust(2)}"
    yield f"wavefront.WaveFront", kw, filter, desc, maxTravel, queue, prog


class RollingAverage:
    def __init__(self, decay: float = 0.5, *, value: float = 0.0):
        self.decay = decay
        self.value = value

    def __call__(self, value: float) -> float:
        self.value = self.decay * self.value + (1 - self.decay) * value
        return self.value


def WaveFrontFilter(
    lines: Iterable[bytes], desc: str, max_travel: float, queue=None, prog=None
):
    slot: int = queue.get() if queue is not None else 1
    desc = desc.rjust(5)

    speed = RollingAverage(0.5, value=0.0)

    def get_desc(c: float = 0.0, s: float = 0.0) -> str:
        c = f"{c:.2f}".rjust(6)
        s = f"{s:.2f}".rjust(5)
        return f"{desc} (coverage {c}% | speed {s} m/s)"

    t0 = time()
    prev_travel = 0.0

    if queue is None:
        prog = tqdm(total=max_travel, desc=get_desc(), position=slot + 1, **ProgOpts)
    else:
        prog[slot] = get_desc(), max_travel, 0.0
    for b in lines:
        line = b.decode("utf-8").strip()
        if line.startswith("#"):
            yield b
        else:
            try:
                travel, p, dp = map(float, line.split(","))
                t1 = time()
                t0, dt = t1, t1 - t0
                _desc = get_desc(p, speed((travel - prev_travel) / dt))
                prev_travel = travel
                if queue is None:
                    prog.desc = _desc
                    prog.n = travel
                    prog.refresh()
                else:
                    prog[slot] = _desc, max_travel, travel
            except Exception as e:
                prog.write(str(e))
    if queue is None:
        prog.close()
    if queue is not None:
        queue.put(slot)


Combs = Iterable[tuple[str, str, str, dict]]
ProgOpts = dict(
    bar_format="{l_bar}{bar}| {n:.3f}/{total_fmt} [{elapsed}]",
    leave=False,
    dynamic_ncols=True,
    file=stdout,
)


def runBugAlgorithms(combs: Combs, META: dict = {}):
    # Bug algorithms
    tasks = list(chain(*map(Bug(), *zip(*combs))))
    progress = tqdm(total=len(tasks), desc="Bug Algorithms", **ProgOpts)
    with Pool(max(1, cpu_count() or 1)) as pool:
        try:
            for triplet, dt, meta in pool.imap_unordered(unpack_exec, tasks):
                META["-".join(triplet)] = meta
                progress.write(format_message(triplet, dt, meta))
                progress.update(1)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            exit(1)
    progress.clear()
    return META


def runBatchSampler(combs: Combs, factory: "RandomWalk | WallBounce", META: dict = {}):
    # Random Walk algorithms
    module = factory.__name__
    prog1 = tqdm(combs, desc=module, position=0, **ProgOpts)
    for world, src, dst, kw in prog1:
        tasks = list(factory(world, src, dst, kw))
        desc = f"{src.rjust(2)}-{dst.ljust(2)}".rjust(10)
        prog2 = tqdm(total=len(tasks), desc=desc, position=1, **ProgOpts)
        with Pool(max(1, cpu_count() or 1)) as pool:
            t = list[int]()
            data = list[float]()
            try:
                for _, dt, meta in pool.imap_unordered(unpack_exec, tasks):
                    t.append(dt)
                    if "travel" in meta and "abort" not in meta:
                        data.append(float(meta["travel"]))
                    else:
                        data.append(None)
                    prog2.update(1)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                exit(1)
            with open(f"results/{src}-{dst}/{module}.list", "w") as f:
                for d in data:
                    if d is not None:
                        f.write(f"{d:.4f}\n")
                    else:
                        f.write("FAIL\n")
            triplet = src, dst, module
            D = [d for d in data if d is not None]
            mean = sum(D) / len(D)
            std = sum((x - mean) ** 2 for x in D) ** 0.5 / len(D)
            meta = dict(travel=mean, std=std, success_rate=len(D) / len(data))
            META["-".join(triplet)] = meta
            prog1.write(format_message(triplet, sum(t) / len(t), meta))
    return META


def monitor_prog(prog: list[None | tuple[str, int | float, int | float]]):
    bars: list[tqdm | None] = [None] * len(prog)
    caches = [None] * len(prog)
    while True:
        for i, (bar, desc, cache) in enumerate(zip(bars, prog, caches)):
            if cache == desc:
                continue
            caches[i] = desc
            if desc is None:
                if bar is not None:
                    bar.close()
                    bars[i] = None
                continue
            desc, total, n = desc
            if bar is None:
                bar = bars[i] = tqdm(desc=desc, total=total, position=i + 1, **ProgOpts)
            bar.desc = desc
            bar.total = total
            bar.n = n
            bar.refresh()


def runWaveFrontPool(combs: Combs, META: dict = {}):
    # WaveFront algorithms
    with Manager() as manager:
        slots = max(1, int((cpu_count() or 1) // 2))
        queue = manager.Queue()
        prog = manager.list([None] * (slots))
        tasks = list(chain(*map(WaveFront(queue, prog), *zip(*combs))))
        for n in range(slots):
            queue.put(n)
        progress = tqdm(total=len(tasks), desc="WaveFront", position=0, **ProgOpts)
        monitor = Process(target=monitor_prog, args=(prog,))
        monitor.start()
        with Pool(slots) as pool:
            for triplet, dt, meta in pool.imap_unordered(unpack_exec, tasks):
                meta["success_rate"] = meta.get("coverage", None)
                META["-".join(triplet)] = meta
                progress.write(format_message(triplet, dt, meta))
                progress.update(1)
        monitor.terminate()
        progress.clear()
    return META


def runWaveFrontSync(combs: Combs, META: dict = {}):
    # WaveFront algorithms
    tasks = list(chain(*map(WaveFront(None, None), *zip(*combs))))
    progress = tqdm(total=len(tasks), desc="WaveFront", position=0, **ProgOpts)
    for triplet, dt, meta in map(unpack_exec, tasks):
        meta["success_rate"] = meta.get("coverage", None)
        META["-".join(triplet)] = meta
        progress.write(format_message(triplet, dt, meta))
        progress.update(1)
    progress.clear()
    return META


if __name__ == "__main__":
    register()
    parser = ArgumentParser(prog="python3 -m batch")
    parser.add_argument("--world", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-travel", type=float, default=1000)
    args = parser.parse_args()
    demo = bool(args.demo)
    KW = dict(
        radius=0.255,
        resolution=0.025,
        scale=2.0,
        maxTravel=float(args.max_travel),
        # **SLICE,
    )
    META = dict[str, any]()
    # Generate world view
    # Python("tools.slice")(world, prefix="results/world", scale=4, **SLICE).wait()
    config = parse(stdin)
    runBugAlgorithms(combinations(**config, **KW, save=True), META=META)
    if demo:
        exit(0)
    runBatchSampler(combinations(**config, **KW), RandomWalk(), META=META)
    runBatchSampler(combinations(**config, **KW), WallBounce(), META=META)
    runWaveFrontPool(combinations(**config, **KW, save=True), META=META)
    # Save meta
    meta_path = Path("results/meta.yaml")
    with meta_path.open("w") as f:
        from yaml import dump

        dump(META, f)
