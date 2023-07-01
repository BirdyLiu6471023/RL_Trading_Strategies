## environment for market
from dataloader import YahooDownloader
import pandas as pd
from yfinance as yf
import numpy as np

class env_market():
    def __init__(self, window_size, skip, initial_money, state_type="value", max_action=20, \
                 ticker='GOOG', start_date="2018-07-01", end_date="2019-07-01"):
        '''
        window_size: the days of stock value would be taken consideration into the state, which is (state_size + 1)
        trend: data to train;
        skip: the trading frequency;
        initial_money: initial amount of money in balance;
        '''

        # variables related to state:
        self.state_type = state_type
        self.window_size = window_size
        # self.half_window = self.window_size //2

        # data:
        self.trend = self.get_data(start_date, end_date, ticker)

        # trading frequency
        # self.skip = skip

        # change in account:
        self.initial_money = initial_money
        self.cash = self.initial_money
        self.inventory = []
        self.stock_shares = 0

        # trajectory:
        self.cur_market_value = self.cash
        self.reward_trajectory = []
        self.total_profit = 0

        # some restrictions: e.g if max_action = 20,
        # which means the max volumn you could buy is 20, and the max volumn you could sell is also 20
        self.max_action = max_action
        self.action_dim = 2 * self.max_action + 1
        self.action_range = list(range(-int(max_action), int(max_action) + 1))

    def get_data(self, start_date, end_date, ticker):
        downloader = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[ticker])
        data = downloader.fetch_data()
        return np.array(data.close)

    def get_state(self, t):
        '''
        state = [remaining_cash, stock_shares, change_1, change_2, change_3,..change_w]
        calculate state, could be change in value or percentage of change;
        '''
        # if there is no enough days of value:
        d = t - self.window_size
        block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + list(self.trend[0: t + 1])

        state = [self.cash, self.stock_shares]
        for i in range(self.window_size - 1):
            if self.state_type == "value":
                state.append(block[i + 1] - block[i])
            else:
                state.append((block[i + 1] - block[i]) / (block[i] + 0.0001))

        return np.array(state)

    def get_valid_action(self, state, t):
        '''
        getting valid actions according to remaining balance and shares in stock,
        and the max_action limitation, before putting into step, please check if action is valid
        '''
        sell_limit = state[1] if state[1] <= self.max_action else self.max_action
        buy_limit = state[0] // self.trend[t] if state[0] // self.trend[t] <= self.max_action else self.max_action
        valid_actions = list(range(-int(sell_limit), int(buy_limit) + 1))

        return valid_actions

    def trading(self, volumn):
        '''
        cash, stock shares, market value would change in every trade day;
        calculate reward daily
        '''

        # go to next state
        self.cash -= volumn * self.trend[self.t]
        self.stock_shares += volumn

        # calculate rewards after taking this action and going to next state:
        self.t += 1
        market_value = self.stock_shares * self.trend[self.t] + self.cash
        reward = market_value - self.cur_market_value
        self.reward_trajectory.append(reward)
        self.total_profit += reward
        self.cur_market_value = market_value
        return reward

    def valid_action(self, action):
        valid_actions = self.get_valid_action(self.cur_state, self.t)
        if action in valid_actions:
            return action

        elif action < 0 and action < min(valid_actions):
            return min(valid_actions)

        elif action > 0 and action > max(valid_actions):
            return max(valid_actions)

    def step(self, action):
        '''
        negative action, sell; positive action, buy
        the sell/buy price is based on the closed price in trend of that day
        '''

        valid_action = self.valid_action(action)

        assert self.done == False

        cur_state = self.cur_state
        # in this step, self.cash, self.stock_shares,
        # self.cur_market_value, self.reward_trajectory would change:
        reward = self.trading(valid_action)

        # update self.cur_state:
        self.cur_state = self.get_state(self.t)
        next_state = self.cur_state

        # output accumulated rewards:
        total_reward = self.total_profit

        # if the state is terminal:
        if self.t == len(self.trend) - 1:
            self.done = True

        return (cur_state, valid_action, reward, next_state, self.done), total_reward

    def reset(self):
        '''
        When you begin a new training, please remember to reset the environment, only run this could cur_state in history;
        '''
        # change in account:
        self.cash = self.initial_money
        self.inventory = []
        self.stock_shares = 0

        # trajectory:
        self.cur_market_value = self.cash
        self.reward_trajectory = []
        self.total_profit = 0
        self.done = False

        self.t = 0
        self.cur_state = self.get_state(self.t)
        self.state_dim = len(self.cur_state)

        return self.cur_state