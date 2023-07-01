# environment and rl related:
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Q_Value_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, modified_state=False):
        super().__init__()
        self.modified_state = modified_state
        state_dim = state_dim - 2 if self.modified_state else state_dim
        self.linear_stack = nn.Sequential(
            # nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, action_dim)
        )

    # this is our target:
    def forward_Q_value(self, state):
        if self.modified_state:
            if len(state.shape) == 1:
                state = state[2:]
            else:
                state = state[:, 2:]
        state_action_value = self.linear_stack(state)
        return state_action_value

    # The Boltzmann Policy Prob, which is not our target:
    def forward_action_prob(self, state, temperature=1.0):
        state_action_value = self.forward_Q_value(state)
        transformed_values = state_action_value / temperature
        action_probabilities = F.softmax(transformed_values, dim=-1)
        return action_probabilities


class Double_Deep_Q_Learning:
    def __init__(self, environment, learning_rate, gamma, tau, hidden_dim=128, batch_size=5, modified_state=False, \
                 temperature=1, loss="MSE", theta=50):

        # trading environment
        self.env = environment
        self.env.reset()

        # Q Value Net:
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.policy_net = Q_Value_Net(self.state_dim, hidden_dim, self.action_dim, modified_state=modified_state)
        self.target_net = Q_Value_Net(self.state_dim, hidden_dim, self.action_dim, modified_state=modified_state)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # update and optimizer:
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.temperature = temperature
        self.loss = loss
        self.theta = theta

        # store trajectory:
        self.memory = deque([], maxlen=self.batch_size * 2)

        self.gamma = gamma

    def boltzmann_policy(self, state):
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        actions_prob = self.target_net.forward_action_prob(torch_state, self.temperature)
        act_idx = torch.multinomial(actions_prob, 1).item()
        act = self.env.action_range[act_idx]
        # while act not in self.env.get_valid_action(state):
        #    act = np.random.choice(self.env.action_range, p=actions_prob)
        return act

    def random_policy(self, state):
        act = np.random.choice(self.env.action_range)
        return act

    def train(self, episodes, cut, checkpoint=100):
        self.total_profit_training = []
        for e in range(episodes):
            # getting a episode:
            # reset trading environment, getting the first state:
            cur_policy = self.boltzmann_policy if e >= cut else self.random_policy

            cur_state = self.env.reset()
            self.first_state = cur_state

            done = self.env.done
            self.memory = []
            while not done:
                act = cur_policy(cur_state)
                self.memory.append(self.env.step(act))
                # update parameters:
                self.replay_optimizer()
                cur_state = self.env.cur_state
                done = self.env.done

            # soft update of the target network's weights
            # theta <- tau*theta + (1- tau)*theta
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + \
                                             target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

            self.total_profit_training.append(self.env.total_profit)
            if (e + 1) % checkpoint == 0:
                print(f"episode {e + 1}: total_profit {self.env.total_profit}")

    def replay_optimizer(self):

        if len(self.memory) < self.batch_size:
            return

            # resample the experience from the self.memory
        samples = random.sample(self.memory, self.batch_size)

        train_state = torch.zeros((self.batch_size, self.state_dim))
        target_state_action_values = torch.zeros((self.batch_size, self.action_dim))
        action_batch = []

        for i in range(self.batch_size):

            (cur_state, cur_action, reward, next_state, done), _ = samples[i]

            cur_action_index = cur_action + self.env.max_action

            action_batch.append([cur_action_index])

            # transform into torch format;
            cur_torch_state = torch.tensor(cur_state, dtype=torch.float).to(self.device)
            next_torch_state = torch.tensor(next_state, dtype=torch.float).to(self.device)

            # calculate Q value:
            Q = self.policy_net.forward_Q_value(cur_torch_state)
            Q_new = self.target_net.forward_Q_value(next_torch_state)

            target = Q
            target[cur_action_index] = reward
            if not done:
                target[cur_action_index] += self.gamma * torch.max(Q_new, dim=0)[0].item()

            # put into the training space:
            train_state[i] = cur_torch_state
            target_state_action_values[i] = target

            # train_state =  torch.tensor(train_state,dtype=torch.float).to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(self.device)

        action_batch = torch.tensor(action_batch)
        state_action_values = self.policy_net.forward_Q_value(train_state)
        state_action_values_gather = state_action_values.gather(1, action_batch)

        # Huber Loss
        if self.loss == "Huber":
            criterion = nn.HuberLoss(delta=self.theta)
            loss = criterion(state_action_values_gather, target_state_action_values.gather(1, action_batch))

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(self.Q_Net.parameters(), 1000)
            self.optimizer.step()

        elif self.loss == "MSE":
            loss_func = nn.MSELoss()
            loss = loss_func(state_action_values_gather, target_state_action_values.gather(1, action_batch))

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()


# test:
def test(trained_policy, episodes=100, ticker='GOOG', start_date="2019-07-01", end_date="2019-12-01"):


    env_test = env_market(window_size=30, skip=1, initial_money=initial_money, state_type="value", max_action=1, \
                          ticker='GOOG', start_date=start_date, end_date=end_date)

    total_rewards = []
    for e in range(episodes):
        cur_state = env_test.reset()
        # simulation:

        done = False
        valid_actions = []
        rewards = []
        while not done:
            # state:
            state = torch.tensor(cur_state, dtype=torch.float).to(device)

            # select action:
            action = trained_policy(state)

            # get result and go to next state:
            (_, valid_action, reward, next_state, done), total_reward = env_test.step(action)
            cur_state = next_state

        total_rewards.append(total_reward)

    return total_rewards