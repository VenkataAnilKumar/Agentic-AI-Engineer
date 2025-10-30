# 🎯 Benchmarks & Datasets

## 📋 Overview

Standard benchmarks and datasets are essential for comparing agent performance and measuring progress. This guide covers key benchmarks across reinforcement learning, multi-agent systems, and LLM agents.

## 🎮 Atari Benchmark

### Overview

The Arcade Learning Environment (ALE) provides 57 Atari 2600 games for RL benchmarking.

```python
import gymnasium as gym

# Create Atari environment
env = gym.make('ALE/Breakout-v5', render_mode='human')

# Observation: (210, 160, 3) RGB image
# Actions: Discrete(4) - NOOP, FIRE, RIGHT, LEFT
# Reward: Score delta

state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

print(f"Episode reward: {total_reward}")
```

### Key Games

| Game | Description | Challenge |
|------|-------------|-----------|
| **Pong** | Classic paddle game | Continuous control, opponent modeling |
| **Breakout** | Brick breaking | Long-term planning, pixel-level control |
| **Space Invaders** | Shoot aliens | Timing, spatial reasoning |
| **Montezuma's Revenge** | Exploration puzzle | Sparse rewards, exploration |
| **Ms. Pac-Man** | Maze navigation | Multi-objective, dynamic enemies |

### Performance Standards

```python
# Human-normalized scores
def normalize_score(agent_score, random_score, human_score):
    """Normalize to human baseline"""
    return (agent_score - random_score) / (human_score - random_score)

# DQN (2015): 75% of human performance
# Rainbow (2017): 230% of human performance
# Agent57 (2020): >100% on all 57 games

human_scores = {
    'Pong': 14.6,
    'Breakout': 30.5,
    'SpaceInvaders': 1668.7,
    'MsPacman': 6951.6
}
```

## 🤖 MuJoCo Benchmark

### Continuous Control Tasks

```python
import gymnasium as gym

# Create MuJoCo environment
env = gym.make('HalfCheetah-v4')

# Observation: Joint positions and velocities
# Actions: Continuous torques for each joint
# Reward: Forward velocity minus control cost

# Popular environments:
mujoco_envs = [
    'Ant-v4',           # 8-DOF quadruped
    'HalfCheetah-v4',   # 6-DOF bipedal runner
    'Hopper-v4',        # 3-DOF hopper
    'Walker2d-v4',      # 6-DOF bipedal walker
    'Humanoid-v4',      # 17-DOF humanoid
    'Swimmer-v4',       # 5-DOF swimmer
]
```

### Performance Baselines

```python
# Expected returns (1M timesteps)
baselines = {
    'HalfCheetah-v4': {
        'TRPO': 2000,
        'PPO': 3000,
        'SAC': 10000,
        'TD3': 11000
    },
    'Humanoid-v4': {
        'TRPO': 1200,
        'PPO': 1500,
        'SAC': 5000,
        'TD3': 5500
    }
}

def evaluate_agent(agent, env_name, episodes=10):
    """Standard evaluation protocol"""
    env = gym.make(env_name)
    rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, deterministic=True)  # No exploration
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
    
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards)
    }
```

## ⚔️ StarCraft II Benchmark

### SMAC (StarCraft Multi-Agent Challenge)

```python
from smac.env import StarCraft2Env

# Create SMAC environment
env = StarCraft2Env(map_name="3m")  # 3 Marines vs 3 Marines

# Multi-agent cooperative task
n_agents = env.n_agents
n_actions = env.n_actions

# Observation: Local observations per agent
# Actions: Move, attack, stop
# Reward: Shared team reward

state = env.reset()
done = False

while not done:
    actions = []
    
    for agent_id in range(n_agents):
        obs = env.get_obs_agent(agent_id)
        action = agent.select_action(obs, agent_id)
        actions.append(action)
    
    reward, done, _ = env.step(actions)

env.close()
```

### Map Difficulties

| Map | Agents | Difficulty | Description |
|-----|--------|------------|-------------|
| **3m** | 3 | Easy | 3 Marines vs 3 Marines |
| **8m** | 8 | Easy | 8 Marines vs 8 Marines |
| **2s3z** | 5 | Medium | 2 Stalkers + 3 Zealots |
| **3s5z** | 8 | Hard | 3 Stalkers + 5 Zealots |
| **corridor** | 6 | Super Hard | 6 Zealots in corridor |

### State-of-the-Art Results

```python
# Win rates on SMAC maps
sota_performance = {
    'QMIX': {'3m': 100, '8m': 100, '3s5z': 96},
    'QPLEX': {'3m': 100, '8m': 100, '3s5z': 98},
    'MAVEN': {'3m': 100, '8m': 100, '3s5z': 97}
}
```

## 🧠 AgentBench (LLM Agents)

### Comprehensive LLM Agent Evaluation

```python
# AgentBench evaluates LLM agents across 8 distinct scenarios

scenarios = {
    'os_interaction': 'Operating system tasks',
    'database': 'SQL query generation',
    'knowledge_graph': 'KG reasoning',
    'digital_card_game': 'Strategic game playing',
    'lateral_thinking_puzzles': 'Creative problem solving',
    'house_holding': 'Household tasks (ALFWorld)',
    'web_shopping': 'WebShop navigation',
    'web_browsing': 'General web tasks'
}

# Example: OS Interaction
from agentbench import OSInteractionBench

bench = OSInteractionBench()
task = bench.get_task(0)

# Task: Create directory and file
prompt = """
Task: Create a directory called 'test' and a file 'hello.txt' with content 'Hello World'
"""

agent_response = llm_agent.execute(prompt)
success = bench.evaluate(task, agent_response)
```

### LLM Agent Performance

```python
# Performance comparison (Success Rate %)
llm_performance = {
    'GPT-4': {
        'os_interaction': 72.4,
        'database': 54.3,
        'knowledge_graph': 51.2,
        'web_shopping': 36.2
    },
    'GPT-3.5': {
        'os_interaction': 42.5,
        'database': 35.1,
        'knowledge_graph': 28.3,
        'web_shopping': 18.7
    },
    'Claude-2': {
        'os_interaction': 65.3,
        'database': 48.2,
        'knowledge_graph': 45.1,
        'web_shopping': 32.4
    }
}
```

## 👥 Multi-Agent Benchmarks

### Multi-Agent Particle Environments (MPE)

```python
from pettingzoo.mpe import simple_spread_v3

# Create environment
env = simple_spread_v3.parallel_env(N=3)

observations, infos = env.reset()

while env.agents:
    actions = {
        agent: policy(observations[agent])
        for agent in env.agents
    }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

### Available Scenarios

| Scenario | Agents | Type | Description |
|----------|--------|------|-------------|
| **Simple Spread** | 3 | Cooperative | Cover landmarks |
| **Simple Adversary** | 3 | Competitive | Catch target |
| **Simple Crypto** | 2 | Communication | Secret messaging |
| **Simple Push** | 2 | Physical | Push ball to goal |

### Level-Based Foraging (LBF)

```python
import lbforaging

# Create foraging environment
env = lbforaging.ForagingEnv(
    players=4,
    max_food=8,
    sight=2,
    max_episode_steps=50
)

# Cooperative foraging task
# Agents must coordinate to collect food
```

## 🛡️ Safety Benchmarks

### AI Safety Gridworlds

```python
from ai_safety_gridworlds.environments import boat_race

# Safety-focused environments
env = boat_race.BoatRaceEnvironment()

# Challenges:
safety_challenges = [
    'boat_race',          # Side effects
    'conveyor_belt',      # Reward hacking
    'distributional_shift', # Distribution shift
    'friend_foe',         # Robustness
    'safe_interruptibility', # Interruptibility
    'tomato_watering'     # Reward gaming
]
```

### SafeLife

```python
from safelife import SafeLifeEnv

# Dynamic environment with safety constraints
env = SafeLifeEnv('append-still')

# Objectives:
# 1. Accomplish goal (green cells)
# 2. Preserve life (don't destroy red cells)
# 3. Side effect penalty

state = env.reset()
for _ in range(1000):
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    
    # Evaluate safety
    safety_score = info['safety_performance']
```

## 📊 Domain-Specific Benchmarks

### Robotics

```python
# RoboSuite
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False
)

# Tasks: Lift, Stack, PickPlace, Door, Wipe
```

### Autonomous Driving

```python
# CARLA Simulator
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

# Scenarios: Urban navigation, highway, weather conditions
```

### Finance

```python
# FinRL
from finrl import config
from finrl.env.env_stocktrading import StockTradingEnv

# Stock trading environment
df = load_stock_data()
env = StockTradingEnv(df)

# Tasks: Portfolio optimization, risk management
```

## 🎲 Procedural Generation Benchmarks

### Procgen

```python
from procgen import ProcgenEnv

# 16 procedurally generated games
env = ProcgenEnv(
    num_envs=1,
    env_name="coinrun",
    num_levels=0,  # 0 = unlimited levels
    start_level=0,
    distribution_mode="hard"
)

# Tests generalization to unseen levels
```

### NetHack Learning Environment

```python
import gym
import nle

env = gym.make('NetHackChallenge-v0')

# Complex roguelike game
# Tests:
# - Long-horizon planning
# - Partial observability
# - Stochasticity
# - Combinatorial action space
```

## 📚 Resources

### Benchmark Suites
- **[Gymnasium](https://gymnasium.farama.org/)** - Standard RL environments
- **[PettingZoo](https://pettingzoo.farama.org/)** - Multi-agent environments
- **[MiniGrid](https://minigrid.farama.org/)** - Simple gridworld environments
- **[Behaviour Suite](https://github.com/deepmind/bsuite)** - DeepMind testing suite

### Leaderboards
- **[Papers With Code](https://paperswithcode.com/)**
- **[OpenAI Gym Leaderboard](https://github.com/openai/gym)**
- **[SMAC Leaderboard](https://github.com/oxwhirl/smac)**

## 🔗 Related Topics

- [Metrics & Methods](./Metrics-Methods.md)
- [Evaluation Frameworks](./Evaluation-Frameworks.md)
- [Open-Source Libraries](../Frameworks-Tools/Open-Source-Libraries.md)
- [Real-World Applications](../Applications-Case-Studies/Real-World-Applications.md)

---

*This guide covers standard benchmarks for agent evaluation. For evaluation metrics, see Metrics & Methods.*