import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from mpe.make_env1 import make_env
from collections import namedtuple
from utils import parse_args
import matplotlib.pyplot as plt

RewardAgg = namedtuple('RewardAgg', ['ep', 'reward'])
class Runner_MAPPO_MPE:
    def __init__(self, args, number):
        self.args = args
        self.env_name = args.env_name
        self.number = number
        self.seed = args.seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = make_env(self.env_name, discrete=True) # Discrete action space
        self.args.N = self.env.n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space[i].n for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))
        self.rew_agg = []

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        plt.plot([r.ep for r in self.rew_agg], [r.reward for r in self.rew_agg])
        plt.title('MAPPO')
        plt.xlabel('Training step')
        plt.ylabel('Normalized episode reward')
        plt.savefig('./Plots/MAPPO_env_{}_number_{}_seed_{}.png'.format(self.env_name, self.number, self.seed))
        plt.show()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward), flush=True)
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        # Update list of rewards for plots
        self.rew_agg.append(RewardAgg(self.total_steps, evaluate_reward))

    def run_episode_mpe(self, evaluate=False):
        noise_std = 0.05
        episode_reward = 0
        obs_n = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        # Execute user-defined number of random steps for exploration, using
        # random noise for the log prob
        for episode_step in range(self.args.episode_limit):
            if self.total_steps < self.args.random_steps:
                a_n = [self.env.action_space[i].sample() for i in range(self.args.N)] # Sample random action
                a_n = np.asarray(a_n)
                noise = torch.normal(mean=0,std=noise_std, size=(3,5))
                noise_sm = torch.nn.functional.softmax(noise)
                dist = Categorical(probs=noise_sm)
                a_n_samp = dist.sample()
                a_logprob_n = dist.log_prob(a_n_samp)
                a_logprob_n = a_logprob_n.numpy()
            else:
                # a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
                a_n, a_logprob_n = self.agent_n.choose_action(args, obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            episode_reward += r_n[0]
            # Rendering exception: pyglet 2.0.8 requires Python 3.8 or newer.
            # self.env.render()

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1


if __name__ == '__main__':
    args = parse_args()

    runner = Runner_MAPPO_MPE(args, number=1)
    runner.run()
