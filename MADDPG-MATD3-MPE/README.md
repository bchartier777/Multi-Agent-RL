# Modified implementation of MAPPO
This is a modified implementation of Multi-agent Proximal Policy Optimization (MAPPO). [This](https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main) is the reference repo, developed by Lizhi Sjtu.<br/>

## Summary of changes
The following changes were implemented:
 - Implemented alternate versions of the MLP Actor/Critic networks, Actor_MLP_v2, Critic_MLP_v2.  These implement additional stacked linear layers with an optional BatchNorm1d (BatchNorm1d is not yet working)
 - Implemented second recurrent Actor/Critic network, Actor_RNN_v2 and Critic_RNN_v2, replacing the GRU with an LSTM
 - Added an optional execution of user-defined number of random actions from the action space prior to choosing actions from the network, for exploration
 - Added option for addition of Gaussian noise to the action for exploration
 - Enabled AdamW optimizer
 - Fixed tensor/Numpy array conversion warning for optimization, although it did not improve throughput, added reward plotting, additional minor refactoring

Some source changes are required to enable a subset of these.

## Testing environments
The return has been validated on Simple Spread for all changes but not all combinations of changes.

## Performance
In the process of quantifying the return for all updates.

## Environment requirements
Requires [this](https://github.com/openai/multiagent-particle-envs) version of Multi-agent Particle Environment (MPE).  See the notes in the original repo in the reference section for the minor changes which are required to make_env.py and environment.py in MPE to enable support for discrete environments.

## Python version and Conda environment
This has been tested with Python 3.7.9 on Win 10.  Use of a virtual environment is recommended.  Following is a Conda implementation:

```
conda create --name marl_env python==3.7.9 pip
conda activate
pip install -r requirements.txt
```

The requirements.txt file was generated with pipreqs, not the environment configuration from the reference repo, but should be correct.

## Usage  WORKING HERE
All execution parameters are implemented in MAPPO_MPE_main.py.  Following is an example for the simple spread environment.

```
python MAPPO_MPE_main.py --env_name simple_spread  1>ExecOut\stdOutMAPPO_SimpSpread.txt  2>ExecOut\stdErrMAPPO_SimpSpread.txt
```

## Outputs
The return at each timestep is output to the terminal and a plot of the normalized episode reward is generated prior to termination.

## Results
See next steps

## Next steps
 - Quantify the normalized return with all enhancements enabled

## References
[1] [This](https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main) is the reference repo, developed by Lizhi Sjtu.

[2] The paper: @article{title={The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games}, author={Yu, Velu, Vinitsky,  Gao, Wang, Bayen and Wu}, journal={arXiv preprint arXiv:2103.01955}, year={2022}
}

