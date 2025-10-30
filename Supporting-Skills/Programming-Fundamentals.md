# 💻 Programming Fundamentals

## 📋 Overview

Strong programming fundamentals are essential for building robust, maintainable agent systems. This guide covers Python best practices, design patterns, and software engineering principles for AI development.

## 🐍 Python Best Practices

### Type Hints & Documentation

```python
from typing import List, Dict, Optional, Union, Callable
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Configuration for agent system"""
    name: str
    learning_rate: float = 0.001
    max_episodes: int = 1000
    device: str = "cpu"

def train_agent(
    agent: 'Agent',
    env: 'Environment',
    episodes: int,
    callbacks: Optional[List[Callable]] = None
) -> Dict[str, float]:
    """
    Train agent in environment.
    
    Args:
        agent: The agent to train
        env: Training environment
        episodes: Number of training episodes
        callbacks: Optional callbacks for monitoring
    
    Returns:
        Dictionary of training metrics
    """
    metrics = {'total_reward': 0.0, 'success_rate': 0.0}
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward)
        
        if callbacks:
            for callback in callbacks:
                callback(episode, metrics)
    
    return metrics
```

### Context Managers

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Time a code block"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        print(f"{name} took {duration:.2f}s")

# Usage
with timer("Agent training"):
    train_agent(agent, env, episodes=100)

class ResourceManager:
    """Manage agent resources"""
    
    def __enter__(self):
        self.connection = self.connect_to_server()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        return False

# Usage
with ResourceManager() as conn:
    conn.send_data(data)
```

### Decorators

```python
import functools
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for agent actions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2.0)
def call_external_api(endpoint: str) -> dict:
    """Call API with retry logic"""
    response = requests.get(endpoint)
    response.raise_for_status()
    return response.json()

def log_performance(func):
    """Log function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} executed in {duration:.4f}s")
        return result
    return wrapper

@log_performance
def complex_computation(n: int) -> int:
    return sum(i**2 for i in range(n))
```

## 🎨 Design Patterns

### Strategy Pattern (Agent Behaviors)

```python
from abc import ABC, abstractmethod

class ExplorationStrategy(ABC):
    """Base class for exploration strategies"""
    
    @abstractmethod
    def select_action(self, q_values: np.ndarray) -> int:
        pass

class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
    
    def select_action(self, q_values: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)

class BoltzmannExploration(ExplorationStrategy):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def select_action(self, q_values: np.ndarray) -> int:
        exp_q = np.exp(q_values / self.temperature)
        probs = exp_q / exp_q.sum()
        return np.random.choice(len(q_values), p=probs)

class Agent:
    def __init__(self, strategy: ExplorationStrategy):
        self.strategy = strategy
        self.q_values = np.zeros(10)
    
    def act(self) -> int:
        return self.strategy.select_action(self.q_values)

# Usage
agent1 = Agent(EpsilonGreedy(epsilon=0.1))
agent2 = Agent(BoltzmannExploration(temperature=0.5))
```

### Observer Pattern (Event System)

```python
from typing import List, Callable

class Observable:
    """Observable for event notifications"""
    
    def __init__(self):
        self._observers: List[Callable] = []
    
    def attach(self, observer: Callable):
        self._observers.append(observer)
    
    def detach(self, observer: Callable):
        self._observers.remove(observer)
    
    def notify(self, event: dict):
        for observer in self._observers:
            observer(event)

class Agent(Observable):
    def __init__(self):
        super().__init__()
        self.state = "idle"
    
    def perform_action(self, action: str):
        self.state = "acting"
        self.notify({'type': 'action_started', 'action': action})
        
        # Perform action...
        result = self._execute(action)
        
        self.state = "idle"
        self.notify({'type': 'action_completed', 'result': result})

# Observers
def log_event(event: dict):
    print(f"Event: {event}")

def save_to_db(event: dict):
    # Save event to database
    pass

agent = Agent()
agent.attach(log_event)
agent.attach(save_to_db)
agent.perform_action("move_forward")
```

### Factory Pattern (Agent Creation)

```python
class AgentFactory:
    """Factory for creating different agent types"""
    
    @staticmethod
    def create_agent(agent_type: str, **kwargs):
        if agent_type == "dqn":
            return DQNAgent(**kwargs)
        elif agent_type == "ppo":
            return PPOAgent(**kwargs)
        elif agent_type == "sac":
            return SACAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

# Usage
agent = AgentFactory.create_agent("dqn", learning_rate=0.001)
```

## ⚡ Async & Concurrency

### Asyncio for Agent Systems

```python
import asyncio
from typing import List

async def agent_task(agent_id: int, duration: float):
    """Simulated agent task"""
    print(f"Agent {agent_id} starting")
    await asyncio.sleep(duration)
    print(f"Agent {agent_id} completed")
    return f"Result from agent {agent_id}"

async def run_multi_agent_system(num_agents: int):
    """Run multiple agents concurrently"""
    tasks = [
        agent_task(i, duration=1.0 + i * 0.5)
        for i in range(num_agents)
    ]
    
    # Run all concurrently
    results = await asyncio.gather(*tasks)
    return results

# Execute
results = asyncio.run(run_multi_agent_system(5))
```

### Threading & Multiprocessing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def train_single_agent(agent_id: int, config: dict) -> dict:
    """Train one agent (CPU-bound)"""
    # Training logic
    return {'agent_id': agent_id, 'reward': 100.0}

def train_agents_parallel(num_agents: int, config: dict):
    """Train multiple agents in parallel"""
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [
            executor.submit(train_single_agent, i, config)
            for i in range(num_agents)
        ]
        
        results = [future.result() for future in futures]
    
    return results

# For I/O-bound tasks (API calls, etc.)
def fetch_data_async(urls: List[str]):
    """Fetch data using threads"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_url, urls))
    return results
```

## 🧪 Testing

### Unit Tests

```python
import unittest
import numpy as np

class TestAgent(unittest.TestCase):
    """Test cases for Agent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = Agent(state_dim=4, action_dim=2)
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        self.assertEqual(self.agent.state_dim, 4)
        self.assertEqual(self.agent.action_dim, 2)
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.array([1, 2, 3, 4])
        action = self.agent.select_action(state)
        
        self.assertIn(action, [0, 1])
        self.assertIsInstance(action, int)
    
    def test_learning(self):
        """Test agent learns from experience"""
        initial_q = self.agent.q_values.copy()
        
        # Simulate learning
        for _ in range(100):
            state = np.random.rand(4)
            action = self.agent.select_action(state)
            reward = 1.0
            self.agent.learn(state, action, reward)
        
        # Q-values should change
        self.assertFalse(np.array_equal(initial_q, self.agent.q_values))
    
    def tearDown(self):
        """Clean up after tests"""
        self.agent = None

if __name__ == '__main__':
    unittest.main()
```

### Pytest Examples

```python
import pytest

@pytest.fixture
def agent():
    """Fixture for agent instance"""
    return Agent(state_dim=4, action_dim=2)

@pytest.fixture
def env():
    """Fixture for environment"""
    return Environment()

def test_agent_acts(agent, env):
    """Test agent can act in environment"""
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    
    assert next_state is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)

@pytest.mark.parametrize("learning_rate", [0.001, 0.01, 0.1])
def test_different_learning_rates(learning_rate):
    """Test agent with different learning rates"""
    agent = Agent(learning_rate=learning_rate)
    assert agent.learning_rate == learning_rate

@pytest.mark.slow
def test_long_training():
    """Test that takes a long time"""
    # Long-running test
    pass
```

## 📊 Data Structures

### Priority Queue for Task Scheduling

```python
import heapq

class PriorityQueue:
    """Min-heap priority queue"""
    
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    
    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    def is_empty(self):
        return len(self._queue) == 0

# Usage
task_queue = PriorityQueue()
task_queue.push("low_priority_task", priority=10)
task_queue.push("high_priority_task", priority=1)
task_queue.push("medium_priority_task", priority=5)

while not task_queue.is_empty():
    task = task_queue.pop()
    print(f"Processing: {task}")
```

### LRU Cache for Memoization

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def compute_value_function(state: tuple) -> float:
    """Cached value function computation"""
    # Expensive computation
    return complex_calculation(state)

# Manual implementation
class LRUCache:
    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

## 📚 Resources

### Books
- **"Fluent Python"** by Luciano Ramalho
- **"Effective Python"** by Brett Slatkin
- **"Design Patterns"** by Gang of Four
- **"Clean Code"** by Robert Martin

### Tools
- [Black](https://github.com/psf/black) - Code formatter
- [Pylint](https://pylint.org/) - Linter
- [MyPy](http://mypy-lang.org/) - Type checker
- [Pytest](https://pytest.org/) - Testing framework

## 🔗 Related Topics

- [System Design](./System-Design.md)
- [Math & Algorithms](./Math-Algorithms.md)
- [Orchestration Frameworks](../Frameworks-Tools/Orchestration-Frameworks.md)
- [APIs & Pipelines](../Integration-Deployment/APIs-Pipelines.md)

---

*This guide covers programming fundamentals for agent development. For system-level design, see System Design.*