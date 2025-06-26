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
- Used in demo(6/13), and demo video on Youtube [NVDA](https://youtu.be/CFLTScgIPdY)/[TSMC](https://youtu.be/1DbVWUNMbnc).
- muti-agent(e.g. 5) and uses Weighted Soft Voting to determine actions, based on validation-set performance weights.
- Reward function is the same as in `ppo1.py`.

### ppo4_noMA.py
- Used in demo(6/13), and demo video on Youtube [NVDA](https://youtu.be/ZMQrV69abBM).
- Same implementation as  `ppo4.py`.
- **Difference**: Excludes **moving average (MA)** as a feature.
  
### ppo4_noRSI.py
- Used in demo(6/13), and demo video on Youtube [NVDA](https://youtu.be/QBf5hbadcoY).
- Same implementation as  `ppo4.py`.
- **Difference**: Excludes **Relative Strength Index (RSI)** as a feature.

## For final report
### ppo1_same_reward.py
- Used in report(6/27).
- Same implementation as `ppo1.py`, only one agent to make decision.
- **Reward function** is the same as in `dql_3.py` from [Lee's Github](https://github.com/harrylee1971/2025-spring-Modern-AI).

# Partners
- [Harry Lee](https://github.com/harrylee1971)
- National Central University Class MA5041 — *Modern Artificial Intelligence*

# Thanks
- Classmate - *Harry Lee*
- Teacher - *[Kuo-Shih Tseng](https://sites.google.com/site/kuoshihtseng/)* 
