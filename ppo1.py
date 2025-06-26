import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from gym import spaces

# 下載股價資料
def get_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    prices = data['Close'].dropna()
    prices = prices.astype(float).values.astype(np.float32)
    return prices, data['Close'].dropna()

# 計算技術指標
def compute_indicators(prices):
    ma5 = np.convolve(prices, np.ones(5)/5, mode='same')
    ma10 = np.convolve(prices, np.ones(10)/10, mode='same')
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return ma5, ma10, rsi

# 交易環境
class TradingEnv(gym.Env):
    def __init__(self, price_series, initial_balance=100000, transaction_fee_percent=0.002, tax_percent=0.003, risk_penalty_factor=0.0002):
        super(TradingEnv, self).__init__()
        self.price_series = price_series
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.tax_percent = tax_percent
        self.risk_penalty_factor = risk_penalty_factor
        self.ma5, self.ma10, self.rsi = compute_indicators(price_series)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.last_action = None
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.balance,
            self.holdings,
            self.price_series[self.current_step],
            self.ma5[self.current_step],
            self.ma10[self.current_step],
            self.rsi[self.current_step]
        ], dtype=np.float32)
        return obs

    def step(self, action):
        price = self.price_series[self.current_step]
        previous_net_worth = self.balance + self.holdings * price

        shares_bought = 0
        shares_sold = 0

        if action == 0:  # Buy
            can_buy = self.balance // price
            self.holdings += can_buy
            cost = can_buy * price * (1 + self.transaction_fee_percent)
            self.balance -= cost
            shares_bought = can_buy

        elif action == 1:  # Sell
            shares_sold = self.holdings
            self.balance += self.holdings * price * (1 - self.transaction_fee_percent - self.tax_percent)
            self.holdings = 0

        self.current_step += 1
        done = self.current_step >= len(self.price_series) - 1

        price = self.price_series[self.current_step]
        self.net_worth = self.balance + self.holdings * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        switch_penalty = 0
        if self.last_action is not None and self.last_action != action:
            switch_penalty = 10
        self.last_action = action

        reward = (self.net_worth - previous_net_worth 
                  - self.risk_penalty_factor * (self.max_net_worth - self.net_worth)
                  - switch_penalty)

        info = {
            'shares_bought': shares_bought,
            'shares_sold': shares_sold
        }

        return self._next_observation(), reward, done, info

# 改良版神經網路
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.value = ValueNetwork(input_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs_old, returns):
        states = np.array(states, dtype=np.float32)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        returns = torch.FloatTensor(returns)

        for _ in range(5):
            values = self.value(states).squeeze()
            advantages = returns - values.detach()

            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            values = self.value(states).squeeze()
            value_loss = nn.MSELoss()(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

# 訓練部分
symbol = "NVDA"
train_start = "2016-01-01"
train_end = "2023-12-31"
test_start = "2024-01-01"
test_end = "2024-12-31"

train_prices, _ = get_stock_data(symbol, train_start, train_end)
env = TradingEnv(train_prices)
agent = PPOAgent(input_dim=6, output_dim=3)
episode_rewards = []
portfolio_values = []

for episode in range(200):
    state = env.reset()
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    done = False
    total_reward = 0

    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        state = next_state
        total_reward += reward
    returns = agent.compute_returns(rewards, dones)
    agent.update(states, actions, log_probs, returns)
    episode_rewards.append(total_reward)

    final_price = env.price_series[env.current_step]
    portfolio_value = env.balance + env.holdings * final_price
    portfolio_values.append(portfolio_value)

    if (episode+1) % 10 == 0:
        print(f"Episode {episode+1} finished")

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode during Training")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_old_reward.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(portfolio_values)
plt.xlabel("Episode")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Value per Episode during Training")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_old_PV.png")
plt.show()

# 測試部分
test_prices, test_price_series = get_stock_data(symbol, test_start, test_end)
test_env = TradingEnv(test_prices)
state = test_env.reset()
done = False

net_worths = []
actions_taken = []
prices_taken = []
shares_boughts = []
shares_solds = []

while not done:
    prices_taken.append(test_env.price_series[test_env.current_step])
    action, _ = agent.select_action(state)
    next_state, reward, done, info = test_env.step(action)
    actions_taken.append(action)
    net_worths.append(test_env.net_worth)
    shares_boughts.append(info['shares_bought'])
    shares_solds.append(info['shares_sold'])
    state = next_state

print(f"Final test net worth: {test_env.net_worth}")

# 顯示交易記錄
# for date, price, act, buy, sell in zip(test_price_series.index, prices_taken, actions_taken, shares_boughts, shares_solds):
#     if act == 0:
#         print(f"{date.date()} Action: BUY, Tried to buy {buy} shares at price {price}")
#     elif act == 1:
#         print(f"{date.date()} Action: SELL, Tried to sell {sell} shares at price {price}")
#     else:
#         print(f"{date.date()} Action: HOLD at price {price}")

for date, price, act, buy, sell in zip(test_price_series.index, prices_taken, actions_taken, shares_boughts, shares_solds):
    if act == 0 and buy > 0:
        print(f"{date.date()} Buy {buy} shares at price {price}")
    elif act == 1 and sell > 0:
        print(f"{date.date()} Sell {sell} shares at price {price}")

# 畫圖
plt.figure(figsize=(14,6))
plt.plot(test_price_series.index, test_prices, label='Stock Price')

buy_plotted = False
sell_plotted = False

for date, price, act in zip(test_price_series.index, prices_taken, actions_taken):
     if act == 0:
         plt.scatter(date, price, marker='^', color='green', label='Buy' if not buy_plotted else "", s=30)
         buy_plotted = True
     elif act == 1:
         plt.scatter(date, price, marker='v', color='red', label='Sell' if not sell_plotted else "", s=30)
         sell_plotted = True


plt.title(f"{symbol} PPO Trading Result")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# 今日收盤價與建議
today = datetime.datetime.now().date()
last_trade_date = today - datetime.timedelta(days=1)
recent_start = (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
recent_end = today.strftime("%Y-%m-%d")

recent_prices_raw = yf.download(symbol, start=recent_start, end=recent_end, auto_adjust=True)
if isinstance(recent_prices_raw.columns, pd.MultiIndex):
    recent_prices_raw.columns = recent_prices_raw.columns.get_level_values(0)

recent_prices_series = recent_prices_raw['Close'].dropna()
recent_prices = recent_prices_series.values.astype(np.float32)

if len(recent_prices) < 15:
    print("資料不足，無法提供建議")
else:
    ma5, ma10, rsi = compute_indicators(recent_prices)
    latest_price = recent_prices[-1]
    latest_ma5 = ma5[-1]
    latest_ma10 = ma10[-1]
    latest_rsi = rsi[-1]

    balance = 100000  # 你目前預設的資金
    holdings = 0

    observation = np.array([
        balance,
        holdings,
        latest_price,
        latest_ma5,
        latest_ma10,
        latest_rsi
    ], dtype=np.float32)

    action, _ = agent.select_action(observation)

    if action == 0:
        can_buy = int(balance // latest_price)
        print(f"建議：買入 (Buy) 當前股價: {latest_price}，建議買入 {can_buy} 股")
    elif action == 1:
        print(f"建議：賣出 (Sell) 當前股價: {latest_price}，建議賣出全部持股")
    else:
        print(f"建議：持有 (Hold) 當前股價: {latest_price}")
