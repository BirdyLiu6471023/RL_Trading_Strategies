# environment and rl related:
import gym
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np




class PolicyNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, modified_state=False):
        super(PolicyNet, self).__init__()
        self.modified_state = modified_state
        state_dim = state_dim - 2 if self.modified_state else state_dim
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self.modified_state:
            if len(x.shape) == 1:
                x = x[2:]
            else:
                x = x[:, 2:]
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return torch.softmax(x, dim=-1)


class ValueNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, modified_state=False):
        super(ValueNet, self).__init__()
        self.modified_state = modified_state
        state_dim = state_dim - 2 if self.modified_state else state_dim
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.modified_state:
            if len(x.shape) == 1:
                x = x[2:]
            else:
                x = x[:, 2:]
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, \
                 lmbda, epochs, eps, gamma, device, modified_state=False):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, modified_state=modified_state).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, modified_state=modified_state).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        actions_index = torch.tensor(transition_dict['actions_index']).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions_index)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions_index))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage

            actor_loss = torch.mean(-torch.min(surr1))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'actions_index': [], \
                                   'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    # (cur_state, valid_action, reward, next_state, done), _ = env.step(action)
                    action_idx = agent.take_action(state)
                    action = action_idx - env.max_action
                    (cur_state, valid_action, reward, next_state, done), _ = env.step(action)
                    transition_dict['states'].append(cur_state)
                    transition_dict['actions_index'].append(action_idx)
                    transition_dict['actions'].append(valid_action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list