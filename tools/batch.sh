#!/bin/bash
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
cat manifest-PERC.yaml | python3 tools/batch.py $@ 2> batch.log
