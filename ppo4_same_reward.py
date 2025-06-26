import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

# === 資料與技術指標 ===
def fetch_0050_data(start_date, end_date):
    ticker = "NVDA"
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if data.empty:
        raise ValueError("無法抓取股價數據，請檢查網路或日期範圍")
    data = data.ffill()
    prices = np.array(data['Close'].values, dtype=np.float64).flatten()
    volumes = np.array(data['Volume'].values, dtype=np.float64).flatten()
    dates = pd.to_datetime(data.index).values
    mask = ~np.isnan(prices) & ~np.isnan(volumes)
    valid_idx = np.where(mask)[0]
    return prices[valid_idx], volumes[valid_idx], dates[valid_idx]

def calculate_technical_indicators(prices, volumes, window=5):
    prices = np.asarray(prices, dtype=np.float64).flatten()
    volumes = np.asarray(volumes, dtype=np.float64).flatten()
    ma = np.convolve(prices, np.ones(window)/window, mode='valid')
    ma = np.concatenate([np.full(window-1, ma[0]), ma])
    diff = np.diff(prices, prepend=prices[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
    avg_gain = np.concatenate([np.full(13, avg_gain[0]), avg_gain])
    avg_loss = np.concatenate([np.full(13, avg_loss[0]), avg_loss])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    volume_norm = volumes / np.max(volumes)
    return ma, rsi / 100, volume_norm

# === 統一環境 ===
class StockTradingEnv:
    def __init__(self, prices, volumes, dates,
                 window_size=5, initial_balance=100000, trading_cost=0.001):
        self.prices = np.array(prices, dtype=np.float64).flatten()
        self.ma, self.rsi, self.volume = calculate_technical_indicators(
            prices, volumes, window_size)
        self.dates = dates
        self.window_size = window_size
        self.n = len(prices)
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0
        self.max_steps = self.n - window_size - 1
        self.action_log = []
        self.hold_streak = 0
        self.last_buy_price = None
        self.trading_cost = trading_cost
        self.trade_cooldown = 0

    def reset(self):
        self.current_step = 0
        self.balance = 100000
        self.shares = 0
        self.action_log = []
        self.hold_streak = 0
        self.last_buy_price = None
        self.trade_cooldown = 0
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        idx = self.current_step + self.window_size - 1
        current_price = self.prices[idx]
        current_ma = self.ma[idx]
        current_date = self.dates[idx]

        reward = 0
        self.trade_cooldown = max(0, self.trade_cooldown - 1)
        sold_shares = 0
        trade_profit = 0

        if action == 1 and self.trade_cooldown == 0:
            if self.balance >= current_price:
                shares_to_buy = int(self.balance // current_price) // 2
                cost = shares_to_buy * current_price * (1 + self.trading_cost)
                if self.balance >= cost:
                    self.shares += shares_to_buy
                    self.balance -= cost
                    self.last_buy_price = current_price
                    self.last_buy_step = self.current_step
                    self.action_log.append(f"Buy {shares_to_buy} shares at {current_price:.2f}")
                    reward += 0.5
                    if current_price > current_ma:
                        reward += 0.3
                    self.hold_streak = 0
                    self.trade_cooldown = 2
        elif action == 2 and self.shares > 0 and self.trade_cooldown == 0:
            sold_shares = self.shares / 2
            buy_price = self.last_buy_price
            holding_days = self.current_step - getattr(self, 'last_buy_step', self.current_step)
            revenue = self.shares * current_price * (1 - self.trading_cost)
            self.balance += revenue
            if buy_price and current_price > buy_price:
                trade_profit = (current_price - buy_price) * sold_shares
                reward += 5.0
                reward += trade_profit / (buy_price * sold_shares) * 200
            if holding_days >= 10:
                reward += 2.0
            if current_price < current_ma:
                reward += 0.3
            self.action_log.append(f"Sell {self.shares} shares at {current_price:.2f}")
            self.shares = 0
            self.last_buy_price = None
            self.hold_streak = 0
            self.trade_cooldown = 2
        else:
            self.action_log.append("Hold")
            self.hold_streak += 1
            if self.hold_streak >= 5:
                reward -= 0.05
            else:
                reward += 0.1

        next_price = (self.prices[self.current_step + self.window_size]
                    if self.current_step < self.max_steps else current_price)
        pv_now = self.balance + self.shares * current_price
        pv_next = self.balance + self.shares * next_price
        reward += (pv_next - pv_now) / pv_now * 500

        done = self.current_step >= self.max_steps
        if done:
            total_return = (pv_now - 100000) / 100000
            reward += total_return * 300

        return self._get_state(), reward, done, {}

    def _get_state(self):
        s = self.current_step
        e = s + self.window_size
        price_window = self.prices[s:e] / np.max(self.prices)
        ma_window = self.ma[s:e] / np.max(self.ma)
        rsi_window = self.rsi[s:e]
        volume_window = self.volume[s:e]
        return np.concatenate([price_window, ma_window, rsi_window, volume_window,
                               [self.shares / 1000, self.balance / 100000]])

# === PPO Agent ===
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
        states = torch.FloatTensor(np.array(states))
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

            value_loss = nn.MSELoss()(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

# === 訓練 ===
end_date = datetime(2024, 5, 9)
train_start = (end_date - timedelta(days=8*365)).strftime('%Y-%m-%d')
test_start = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

prices, volumes, dates = fetch_0050_data(train_start, end_str)
train_mask = dates < np.datetime64(test_start)
test_mask = ~train_mask

train_env = StockTradingEnv(prices[train_mask], volumes[train_mask], dates[train_mask], window_size=5)
state_size = train_env.window_size * 4 + 2
action_size = 3
agent = PPOAgent(state_size, action_size)

episodes = 200
reward_history = []
pv_history = []

for e in range(episodes):
    state = train_env.reset()
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    total_reward = 0

    for _ in range(train_env.max_steps):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = train_env.step(action)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        state = next_state
        total_reward += reward
        if done:
            break

    returns = agent.compute_returns(rewards, dones)
    agent.update(states, actions, log_probs, returns)
    pv = train_env.balance + train_env.shares * train_env.prices[-1]
    reward_history.append(reward)
    pv_history.append(pv)

    #print(f"Ep {e+1}/{episodes} | Reward {total_reward:.2f} | PV {pv:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), reward_history)
plt.title("Reward per Episode during Training")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_new_reward.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), pv_history)
plt.title("Portfolio Value per Episode by PPO")
plt.xlabel("Episode")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_new_pv.png")
plt.show()

# === 今日股價建議 ===
today = datetime.now().strftime("%Y-%m-%d")
recent_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

prices, volumes, _ = fetch_0050_data(recent_start, today)
ma, rsi, volume = calculate_technical_indicators(prices, volumes)

current_price = prices[-1]
current_ma = ma[-1]
current_rsi = rsi[-1]
current_volume = volume[-1]

state = np.concatenate([
    prices[-5:] / np.max(prices),
    ma[-5:] / np.max(ma),
    rsi[-5:],
    volume[-5:],
    [0.0, 1.0]  # 假設初始無持股、初始資金 100%
])

state_tensor = torch.FloatTensor(state).unsqueeze(0)
probs = agent.policy(state_tensor).detach().numpy()[0]
action = np.argmax(probs)
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

print(f"\n{today} share: {current_price:.2f}")
print(f"Today's suggested action: {action_dict[action]}")
