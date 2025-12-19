# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from lib.arguments import parse
from lib.simulation import SimulationBase

if __name__ == "__main__":
    kw = parse()
    if "src" in kw:
        del kw["src"]
    if "dst" in kw:
        del kw["dst"]
    kw["visualize"] = True
    try:
        while True:
            print("=" * 20)
            sim = SimulationBase(**kw)
            print("SRC:", sim.src)
            print("DST:", sim.dst)
    except KeyboardInterrupt:
        print("")
    except:
        pass
