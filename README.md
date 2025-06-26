# Stock Trading Agents with PPO – Final Project Report

This repository presents a series of Proximal Policy Optimization (PPO) agents developed for stock trading tasks. Key variations include single-agent vs. multi-agent designs, soft vs. weighted voting strategies, and ablation experiments involving feature engineering (e.g., moving average, RSI).

We benchmark these agents against a Dueling DQN baseline to evaluate reward shaping and performance stability.  
The full report is available [here](./final_report.pdf).

Project submitted for *MA5041: Modern Artificial Intelligence* — National Central University, Spring 2025.

# Code Description
## First edition
### ppo1.py
- Single-agent implementation
- **Reward function:**
```bash
reward = (self.net_worth - previous_net_worth) 
         - self.risk_penalty_factor * (self.max_net_worth - self.net_worth)
         - switch_penalty
```
- Different reward design from [Lee's](https://github.com/harrylee1971/2025-spring-Modern-AI) Dueling DQN agent.

### ppo2.py
- muti-agent(e.g. 5)
- Uses **soft voting** to decide action.
- Reward function is the same as in `ppo1.py`

### ppo3.py
- muti-agent(e.g. 5).
- Uses **Weighted Soft Voting** to decide action.
- Reward function is the same as in `ppo1.py`.

## Second edition
### ppo4.py
- Used in demo (2025/6/13); YouTube videos: [NVDA](https://youtu.be/CFLTScgIPdY)/[TSMC](https://youtu.be/1DbVWUNMbnc).
- muti-agent(e.g. 5) and uses Weighted Soft Voting to determine actions, based on validation-set performance weights.
- Reward function is the same as in `ppo1.py`.

### ppo4_noMA.py
- Used in demo (2025/6/13); YouTube videos: [NVDA](https://youtu.be/ZMQrV69abBM).
- Same implementation as  `ppo4.py`.
- **Difference**: Excludes **moving average (MA)** as a feature.
  
### ppo4_noRSI.py
- Used in demo (2025/6/13); YouTube videos: [NVDA](https://youtu.be/QBf5hbadcoY).
- Same implementation as  `ppo4.py`.
- **Difference**: Excludes **Relative Strength Index (RSI)** as a feature.

## For final report
### ppo1_same_reward.py
- Used in final report (2025/6/27).
- Same implementation as `ppo1.py` (single-agent)
- **Reward function** is the same as in `dql_3.py` from [Lee's Github](https://github.com/harrylee1971/2025-spring-Modern-AI).

## Appendix
- 📄 [Final Report (PDF)](./final_report.pdf)  
- 📂 Source Code: See all `ppo*.py` files in this repository

# Thanks
- Classmate - *[Harry Lee](https://github.com/harrylee1971)*
- Teacher - *[Kuo-Shih Tseng](https://sites.google.com/site/kuoshihtseng/)*
- National Central University Class MA5041 — *Modern Artificial Intelligence*


