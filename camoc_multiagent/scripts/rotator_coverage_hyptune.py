import sys
import os
import signal
import json
from envs.rotator_coverage import rotator_coverage_v0
from rl_agent.tune_hyp import study

# 7m38s
# TRAIN_STEPS = int(1e6)
# OPT_STEPS = 4

TRAIN_STEPS = int(2_000_000)
OPT_STEPS = 10

# ~15 hr
# TRAIN_STEPS = int(1e7)
# OPT_STEPS = 50

os.setpgrp()

env_train = rotator_coverage_v0.env_train()
env_eval = rotator_coverage_v0.env_train()  # use another copy of the same env stack

hyp = study(
    env_train, env_eval, "rotator_coverage_v0_" + sys.argv[1], TRAIN_STEPS, OPT_STEPS
)

json = json.dumps(hyp)
with open("./rl_agent/hyperparams/rotator_coverage_v0_hyp.json", "w") as f:
    f.write(json)

os.killpg(0, signal.SIGQUIT)
