import os
import signal
import json
from envs.swarmcover import swarm_cover_v1
from rl_agent.tune_hyp import study

TRAIN_STEPS = int(1e5)
OPT_STEPS = 4

os.setpgrp()

env = swarm_cover_v1.train_env()

hyp = study(env, "swarmcover_ppo", TRAIN_STEPS, OPT_STEPS)

json = json.dumps(hyp)
with open("./rl_agent/hyperparams/swarmcover_ppo_hyp.json", "w") as f:
    f.write(json)

os.killpg(0, signal.SIGQUIT)
