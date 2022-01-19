# Computationally Approximated Manifold ODE Control applied to Geometrically Complex Multiagent Environments

Using conda, create the environment with

```
conda env create -f environment.yml
conda activate camoc
```

To run the tests on built-in PPO on the built in PettingZoo environments, run 

```
python -m scripts.pistonball_ppo_test
python -m scripts.simplespread_ppo_test
```

To train a PPO agent on the Rotator Coverage custom environment, run

```
python -m scripts.rotator_coverage_train
```

And to evaluate and render the trained policy, run

```
python -m scripts.rotator_coverage_eval
```
