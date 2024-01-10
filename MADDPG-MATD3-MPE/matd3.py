import torch
import torch.nn.functional as F
import numpy as np
import copy
from utils import OUNoise
from networks import Actor,Critic_MATD3, Critic_MATD3_v2, Actor_v2, Critic_v2, Actor_v3, Critic_v3


class MATD3(object):
    def __init__(self, args, agent_id):
        self.N = args.N
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.doubleq_minmax_update_freq = args.doubleq_minmax_update_freq
        self.actor_pointer = 0
        # Create an individual actor and critic for each agent according to the 'agent_id'
        # self.actor = Actor(args, agent_id)
        # self.critic = Critic_MATD3(args)
        self.critic = Critic_MATD3_v2(args)
        self.actor = Actor(args, agent_id)
        # self.critic = Critic_v2(args, agent_id)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_c)
        self.ou_noise = OUNoise(size=self.action_dim, seed=args.seed)

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, args, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        if args.gumb_softmax_action == True:
            a, logits = self.actor(obs)
            a = F.gumbel_softmax(logits, hard=True)
            a = a.data.numpy().flatten()
        else:
            a = self.actor(obs).data.numpy().flatten()
        if (args.noise == "gaussian"):
            a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        else: # else OU Noise
            noise = self.ou_noise.sample()
            a = (a + noise).clip(-self.max_action, self.max_action)

        return a

    def train(self, args, replay_buffer, agent_n):
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            batch_a_next_n = []
            for i in range(self.N):
                if args.gumb_softmax_action == True: # Only works with Actor, Critic_MADDPG and Critic_MADDPG_v2
                    batch_a_next, logits = agent_n[i].actor_target(batch_obs_next_n[i])
                    a_softmax = F.gumbel_softmax(logits, hard=True)
                    batch_a_next_n.append(a_softmax.squeeze(0).detach())
                else:
                    batch_a_next = agent_n[i].actor_target(batch_obs_next_n[i]) # torch.Size([1024, 2])
                    noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)
                    batch_a_next_n.append(batch_a_next)

            # Trick 2:clipped double Q-learning
            Q1_next, Q2_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            # Optimization: Use the maximum Q value at user-defined intervals
            if self.actor_pointer % self.doubleq_minmax_update_freq == 0:
                target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)
            else:
                target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.max(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            if args.gumb_softmax_action == True:
                a, logits = self.actor(batch_obs_n[self.agent_id])
                batch_a_n[self.agent_id] = F.gumbel_softmax(logits, hard=True)
            else:
                batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_n, batch_a_n).mean()  # Only use Q1
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(), "./model/{}/{}_actor_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, number, int(total_steps / 1000), agent_id))
