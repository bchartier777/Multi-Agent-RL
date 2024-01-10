# Modified implementations of MADDPG and MATD3
This is a modified implementation of Multi-agent Deep Deterministic Policy Gradient (MADDPG) and Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3).  [This](https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main) is the reference repo, developed by Lizhi Sjtu.  This is a well written, succinct implementation of multiple MARL algorithms.<br/>

## Summary of changes
The primary purpose of this implementation was to evaluate the benefit of the following changes to average return for both MADDPG and MATD3:
 - Added second Actor/Critic network, Actor_v2, Critic_v2, implemeting additional stacked linear layers with an optional BatchNorm1d (BatchNorm1d is not yet working).  Also implements user-defined fc1 and fc2 unit size
 - Added third recurrent Actor/Critic network, Actor_v3 and Critic_v3, implementing an LSTM.  The algorithm is executing but the return does not converge since additional modifications are required to the replay buffer and Q calculations.
 - Added an optional Gumbel softmax to the action selection from the actor_target as well as action reslection for the actor loss. For MADDPG, this only works with the original Actor network.
 - Added optional noise from an Ornstein–Uhlenbeck process, in addition to the Gaussian noise.
 - Added an optional execution of user-defined number of random samples from the action space prior to choosing actions from the network
 - Enabled AdamW optimizer
 - Minor changes to enable execution of additional MPE scenarios as well as limited refactoring to enable diagnostics

For MATD3 only:
 - Implemented a second critic network, Critic_MATD3_v2, which has 5 hidden layers (512, 256, ...).  Critic_MATD3_v2 works with Actor, Actor_v2 and Actor_v3.
 - For optimization (trick) 1, added option to the clipped double-Q learning to use the maximum Q value at a user-defined interval

Some source changes are required to enable a subset of these.

## Testing environments
The return has been validated on simple_speaker_listener.  Also tested for execution but not averge return on simple_tag, simple_spread, simple_push, simple_crypto and simple_adversary.  Not yet working on simple_world_comm, simple_reference and simple.

## Performance
The return for the simple_speaker_listener environment is similar to the reference repo.  One next step is to quantify performance on the other Petting Zoo environments.

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

## Usage
All execution parameters are implemented in parse_args in utils.py.  Following is an example for the simple speaker listener environment.

```
python MADDPG_MATD3_main.py --env_name simple_speaker_listener  1>ExecOut\stdOutMADDPG_SimpSpeak.txt  2>ExecOut\stdErrMADDPG_SimpSpeak.txt
```

## Outputs
The return at each timestep is output to the terminal.  Have not yet integrated Tensorboard or average return graph generation.

## Results
Executing with Actor_v2, Critic_v2, the average returns for simple speaker listener are similar to the Actor / Critic networks in the original implementation.  One next step is to quantify return with other enhancements enabled.

## Next steps
 - Scope integration of [Multi-agent Mujoco](https://github.com/Farama-Foundation/Gymnasium-Robotics). This may require Linux, since the docs state that Windows is not supported (my current dev environment is Windows).
 - Quantify performance of subset of combination of new options with multiple Petting Zoo environments
 - Scope completion of the recurrent Actor and Critic classes, requires multiple mods to the two primary training methods and the replay buffer.

## References
[1] [This](https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main) is the reference repo, developed by Lizhi Sjtu.
[2] MADDPG: Lowe R, Wu Y I, Tamar A, et al. Multi-agent actor-critic for mixed cooperative-competitive environments[J]. Advances in neural information processing systems, 2017, 30.<br />
[3] MATD3: Ackermann, J., et al.  Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics, 2019.  arXiv:1910.01465v2.
