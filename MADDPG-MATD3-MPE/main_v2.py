import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mpe.make_env1 import make_env
import argparse
from utils import ReplayBuffer, parse_args, create_res_dir
from maddpg import MADDPG
from matd3 import MATD3
import copy
# from multiagent_mujoco_master.multiagent_mujoco.mujoco_multi import MujocoMulti
import time
# from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

class Trainer:
    def __init__(self, args, env_name, model_num):
        self.args = args
        self.env_name = env_name
        self.model_num = model_num
        self.seed = args.seed
        # print(gym.__version__)
        if args.env_lib == 'petting_zoo': # This version is not currently supported
            print ("Not supporting newer version of Petting Zoo with current version")
        #self.new_env.reset()
        elif args.env_lib == 'mpe':
            self.env = make_env(env_name, discrete=False)  # Continuous action space
            self.env_evaluate = make_env(env_name, discrete=False)
            self.args.N = self.env.n  # The number of agents
            # TODO: Convert to a for loop
            self.args.obs_dim_n = []; self.args.action_dim_n = []
            for i in range(self.args.N):  # obs dimensions of N agents
                self.args.obs_dim_n.append(self.env.observation_space[i].shape[0])
            for i in range(self.args.N):  # actions dimensions of N agents
                self.args.action_dim_n.append(self.env.action_space[i].shape[0])
        elif args.env_lib == 'multi_mujoco':
            # Started the integration of MuJoCo, on hold for now
            print ("Not supporting MuJoCo with current version of this repo.")
            env_args = {"scenario": "HalfCheetah-v2",
                  "agent_conf": "2x3",
                  "agent_obsk": 0,
                  "episode_limit": 1000}
            env = MujocoMulti(env_args=env_args)
            self.env_evaluate = MujocoMulti(env_args=env_args)
            env_info = env.get_env_info()
            n_actions = env_info["n_actions"]
            self.args.N = env_info["n_agents"]
            self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]  # obs dimensions of N agents
            self.args.action_dim_n = [n_actions for i in range(self.args.N)]  # actions dimensions of N agents

        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        res_dir = create_res_dir(args)
        # Create N agents
        if self.args.algorithm == "MADDPG" and args.env_lib == 'mpe':
            print("Algorithm: MADDPG, MPE")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif self.args.algorithm == "MADDPG" and args.env_lib == 'petting_zoo':
            print ("Not supporting newer version of Petting Zoo with current version of this repo.")
            #self.agent_n = [MADDPG_PZ(args, agent_id,self.args.episode_limit,
            #    args.buffer_size, args.batch_size, args.lr_a, args.lr_c, res_dir) for agent_id in range(args.N)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        else:
            print("Wrong!!!")

        if args.env_lib == 'mpe':
            self.replay_buffer = ReplayBuffer(self.args)
        else:
            print ("Only MPE is supported with the current version of this repo.")
            # self.replay_buffer = Buffer_PZ(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/{}/{}_env_{}_model_num_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.model_num, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
        self.evaluate_policy()

        while self.total_steps < self.args.max_train_steps:
            obs_n = self.env.reset()
            for _ in range(self.args.episode_limit):
                # Each agent selects actions based on its own local observations(add noise for exploration)
                # agent_id: env.action_space(agent_id).sample() for agent_id in env.agents
                if self.total_steps < self.args.random_steps:
                    a_n = [self.env.action_space[i].sample() for i in range(self.args.N)]
                else:
                    a_n = [agent.choose_action(self.args, obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))
                # Store the transition
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.total_steps += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size:
                    # Train each agent individually
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.args, self.replay_buffer, self.agent_n)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()

                if all(done_n):
                    break

        self.env.close()
        self.env_evaluate.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            obs_n = self.env_evaluate.reset()
            episode_reward = 0
            for _ in range(self.args.episode_limit):
                a_n = []
                if (args.env_lib == 'mpe'):
                    # a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
                     for agent, obs in zip(self.agent_n, obs_n):
                        a_n.append(agent.choose_action(self.args, obs, noise_std=0))
                else:
                    print ("Not supporting other env libraries with current version")
                if (args.env_lib == 'mpe'):
                    obs_next_n, r_n, done_n, _ = self.env_evaluate.step(copy.deepcopy(a_n))
                    episode_reward += r_n[0]
                    obs_n = obs_next_n
                    if all(done_n):
                        break
                else:
                    print ("Not supporting other env libraries for now")
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward, self.noise_std))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/{}_env_{}_model_num_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.model_num, self.seed), np.array(self.evaluate_rewards))
        for agent_id in range(self.args.N):
            self.agent_n[agent_id].save_model(self.env_name, self.args.algorithm, self.model_num, self.total_steps, agent_id)

if __name__ == '__main__':
    args = parse_args() # Parse arguments from command line

    runner = Trainer(args, env_name=args.env_name, model_num=1)
    runner.run()
