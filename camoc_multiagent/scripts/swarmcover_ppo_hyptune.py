import os
import signal
import json
from envs.swarmcover import swarm_cover_v1
from rl_agent.tune_hyp import study

# 7m38s
# TRAIN_STEPS = int(1e6)
# OPT_STEPS = 4

# ~15 hr
TRAIN_STEPS = int(1e7)
OPT_STEPS = 50

os.setpgrp()

env_train = swarm_cover_v1.train_env()
env_eval = swarm_cover_v1.train_env()  # use another copy of the same env stack

hyp = study(env_train, env_eval, "swarmcover_ppo", TRAIN_STEPS, OPT_STEPS)

json = json.dumps(hyp)
with open("./rl_agent/hyperparams/swarmcover_ppo_hyp.json", "w") as f:
    f.write(json)

os.killpg(0, signal.SIGQUIT)
