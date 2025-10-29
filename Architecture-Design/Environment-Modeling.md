# 🌍 Environment Modeling

## 📋 Overview

Environment modeling is the process of creating representations of the world in which agents operate. Accurate models enable agents to predict outcomes, plan actions, and make informed decisions under uncertainty.

## 🧠 World Models

### What is a World Model?

A world model is an agent's internal representation of:
- **State space**: Possible configurations of the environment
- **Dynamics**: How the environment changes over time
- **Observations**: What the agent can perceive
- **Rewards/Goals**: What the agent aims to achieve

### Types of World Models

```python
from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

class WorldModel(ABC):
    """Base class for world models"""
    
    @abstractmethod
    def predict(self, state, action):
        """Predict next state given current state and action"""
        pass
    
    @abstractmethod
    def update(self, state, action, next_state, reward):
        """Update model based on experience"""
        pass

class DeterministicModel(WorldModel):
    """Deterministic state transition model"""
    
    def __init__(self):
        self.transitions = {}  # (state, action) -> next_state
    
    def predict(self, state, action):
        """Predict next state deterministically"""
        key = (state, action)
        return self.transitions.get(key, state)
    
    def update(self, state, action, next_state, reward):
        """Learn transition"""
        self.transitions[(state, action)] = next_state

class ProbabilisticModel(WorldModel):
    """Probabilistic state transition model"""
    
    def __init__(self):
        self.transitions = {}  # (state, action) -> {next_state: probability}
    
    def predict(self, state, action):
        """Sample next state from distribution"""
        key = (state, action)
        if key not in self.transitions:
            return state
        
        next_states = list(self.transitions[key].keys())
        probabilities = list(self.transitions[key].values())
        return np.random.choice(next_states, p=probabilities)
    
    def update(self, state, action, next_state, reward):
        """Update transition probabilities"""
        key = (state, action)
        if key not in self.transitions:
            self.transitions[key] = {}
        
        # Increment count for this transition
        if next_state not in self.transitions[key]:
            self.transitions[key][next_state] = 0
        self.transitions[key][next_state] += 1
        
        # Normalize to probabilities
        total = sum(self.transitions[key].values())
        for s in self.transitions[key]:
            self.transitions[key][s] /= total
```

## 🗺️ State Representation

### Feature-Based Representation

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class FeatureVector:
    """Continuous feature representation"""
    features: np.ndarray
    feature_names: List[str]
    
    def normalize(self):
        """Normalize features to [0, 1]"""
        min_vals = self.features.min(axis=0)
        max_vals = self.features.max(axis=0)
        self.features = (self.features - min_vals) / (max_vals - min_vals + 1e-8)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return dict(zip(self.feature_names, self.features))

class StateEncoder:
    """Encode environment states as feature vectors"""
    
    def __init__(self, feature_extractors: Dict[str, callable]):
        self.extractors = feature_extractors
    
    def encode(self, raw_state) -> FeatureVector:
        """Extract features from raw state"""
        features = []
        names = []
        
        for name, extractor in self.extractors.items():
            value = extractor(raw_state)
            if isinstance(value, (list, np.ndarray)):
                features.extend(value)
                names.extend([f"{name}_{i}" for i in range(len(value))])
            else:
                features.append(value)
                names.append(name)
        
        return FeatureVector(
            features=np.array(features),
            feature_names=names
        )

# Example usage
encoder = StateEncoder({
    'position_x': lambda s: s['robot']['x'],
    'position_y': lambda s: s['robot']['y'],
    'velocity': lambda s: np.linalg.norm(s['robot']['velocity']),
    'battery': lambda s: s['robot']['battery_level'],
    'obstacles': lambda s: len(s['obstacles'])
})
```

### Graph-Based Representation

```python
import networkx as nx

class SpatialGraph:
    """Graph representation of spatial environment"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_location(self, location_id, properties=None):
        """Add a location node"""
        self.graph.add_node(location_id, **(properties or {}))
    
    def add_connection(self, loc1, loc2, distance=None, traversable=True):
        """Add connection between locations"""
        self.graph.add_edge(
            loc1, loc2,
            distance=distance,
            traversable=traversable
        )
    
    def shortest_path(self, start, goal):
        """Find shortest path between locations"""
        try:
            return nx.shortest_path(
                self.graph, start, goal,
                weight='distance'
            )
        except nx.NetworkXNoPath:
            return None
    
    def get_neighbors(self, location):
        """Get accessible neighbors"""
        return [
            n for n in self.graph.neighbors(location)
            if self.graph[location][n].get('traversable', True)
        ]

# Example: Building layout
building = SpatialGraph()
building.add_location('room_101', {'type': 'office'})
building.add_location('room_102', {'type': 'office'})
building.add_location('hallway', {'type': 'corridor'})
building.add_connection('room_101', 'hallway', distance=5)
building.add_connection('room_102', 'hallway', distance=7)
```

## 🎲 Handling Uncertainty

### Partially Observable Markov Decision Process (POMDP)

```python
import numpy as np
from typing import List

class POMDP:
    """Partially Observable Markov Decision Process"""
    
    def __init__(self, states, actions, observations):
        self.states = states
        self.actions = actions
        self.observations = observations
        
        # Belief state (probability distribution over states)
        self.belief = np.ones(len(states)) / len(states)
        
        # Model components
        self.transition_prob = {}  # T(s, a, s')
        self.observation_prob = {}  # O(s', a, o)
        self.reward_func = {}  # R(s, a)
    
    def update_belief(self, action, observation):
        """Bayesian belief update"""
        new_belief = np.zeros(len(self.states))
        
        for i, s_prime in enumerate(self.states):
            # Observation probability
            obs_prob = self.observation_prob.get(
                (s_prime, action, observation), 0
            )
            
            # Transition probability sum
            trans_sum = 0
            for j, s in enumerate(self.states):
                trans_prob = self.transition_prob.get(
                    (s, action, s_prime), 0
                )
                trans_sum += self.belief[j] * trans_prob
            
            new_belief[i] = obs_prob * trans_sum
        
        # Normalize
        if new_belief.sum() > 0:
            new_belief /= new_belief.sum()
            self.belief = new_belief
    
    def get_most_likely_state(self):
        """Get state with highest probability"""
        return self.states[np.argmax(self.belief)]
```

### Occupancy Grid Mapping

```python
class OccupancyGrid:
    """2D occupancy grid for spatial uncertainty"""
    
    def __init__(self, width, height, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Probability grid (0.5 = unknown, 0 = free, 1 = occupied)
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return grid_x, grid_y
    
    def update_cell(self, x, y, is_occupied, confidence=0.1):
        """Update cell probability using log-odds"""
        grid_x, grid_y = self.world_to_grid(x, y)
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            current_prob = self.grid[grid_y, grid_x]
            
            # Log-odds update
            current_odds = current_prob / (1 - current_prob)
            update_odds = confidence if is_occupied else (1 - confidence)
            new_odds = current_odds * (update_odds / (1 - update_odds))
            
            # Convert back to probability
            self.grid[grid_y, grid_x] = new_odds / (1 + new_odds)
    
    def is_occupied(self, x, y, threshold=0.7):
        """Check if cell is occupied"""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.grid[grid_y, grid_x] > threshold
        return False
```

## ⏰ Temporal Modeling

### Temporal State Sequences

```python
from collections import deque
from typing import Optional

class TemporalBuffer:
    """Buffer for temporal state sequences"""
    
    def __init__(self, max_length=100):
        self.buffer = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
    
    def add(self, state, timestamp):
        """Add state with timestamp"""
        self.buffer.append(state)
        self.timestamps.append(timestamp)
    
    def get_window(self, duration):
        """Get states within time window"""
        if not self.timestamps:
            return []
        
        current_time = self.timestamps[-1]
        cutoff_time = current_time - duration
        
        states = []
        for i in range(len(self.timestamps) - 1, -1, -1):
            if self.timestamps[i] >= cutoff_time:
                states.insert(0, self.buffer[i])
            else:
                break
        
        return states
    
    def get_velocity(self, state_key):
        """Estimate velocity from state changes"""
        if len(self.buffer) < 2:
            return 0
        
        s1 = self.buffer[-2][state_key]
        s2 = self.buffer[-1][state_key]
        dt = self.timestamps[-1] - self.timestamps[-2]
        
        return (s2 - s1) / dt if dt > 0 else 0
```

## 🔮 Predictive Models

### Learned Dynamics Model

```python
import torch
import torch.nn as nn

class NeuralDynamicsModel(nn.Module):
    """Neural network world model"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        """Predict next state"""
        x = torch.cat([state, action], dim=-1)
        delta_state = self.network(x)
        return state + delta_state  # Residual prediction
    
    def predict_sequence(self, initial_state, actions):
        """Predict sequence of future states"""
        states = [initial_state]
        state = initial_state
        
        for action in actions:
            state = self.forward(state, action)
            states.append(state)
        
        return states

# Training example
def train_dynamics_model(model, replay_buffer, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        states, actions, next_states = replay_buffer.sample(batch_size=64)
        
        predicted_states = model(states, actions)
        loss = criterion(predicted_states, next_states)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🛠️ Libraries & Tools

### Simulation Environments

| Library | Purpose | Repository |
|---------|---------|-----------|
| [PyBullet](https://pybullet.org/) | Physics simulation | [GitHub](https://github.com/bulletphysics/bullet3) |
| [MuJoCo](https://mujoco.org/) | Multi-joint dynamics | [Website](https://mujoco.org/) |
| [CARLA](https://carla.org/) | Autonomous driving | [Website](https://carla.org/) |
| [Habitat](https://aihabitat.org/) | Embodied AI | [GitHub](https://github.com/facebookresearch/habitat-sim) |

### Spatial Reasoning

| Library | Purpose | Repository |
|---------|---------|-----------|
| [Shapely](https://shapely.readthedocs.io/) | Geometric operations | [GitHub](https://github.com/shapely/shapely) |
| [Scikit-geometry](https://scikit-geometry.github.io/) | Computational geometry | [GitHub](https://github.com/scikit-geometry/scikit-geometry) |
| [NetworkX](https://networkx.org/) | Graph structures | [GitHub](https://github.com/networkx/networkx) |

### Probabilistic Modeling

| Library | Purpose | Repository |
|---------|---------|-----------|
| [PyMC](https://www.pymc.io/) | Bayesian inference | [GitHub](https://github.com/pymc-devs/pymc) |
| [PyStan](https://pystan.readthedocs.io/) | Probabilistic programming | [GitHub](https://github.com/stan-dev/pystan) |
| [Pomegranate](https://pomegranate.readthedocs.io/) | Probabilistic models | [GitHub](https://github.com/jmschrei/pomegranate) |

## 📚 Learning Resources

### Books
- **"Probabilistic Robotics"** by Thrun, Burgard, Fox
- **"Artificial Intelligence: A Modern Approach"** - Environment representation
- **"Planning Algorithms"** by LaValle

### Papers
- "World Models" - Ha & Schmidhuber (2018)
- "Learning Latent Dynamics for Planning" - Hafner et al. (2019)
- "Mastering Atari with Discrete World Models" - Hafner et al. (2021)

### Courses
- **Probabilistic Robotics** - University of Freiburg
- **Robot Mapping** - Cyrill Stachniss

## 🔗 Related Topics

- [Decision Making & Planning](../Core-Concepts/Decision-Making-Planning.md)
- [Simulation Environments](./Simulation-Environments.md)
- [Reasoning & Problem Solving](../Core-Concepts/Reasoning-Problem-Solving.md)
- [Agent Architectures](./Agent-Architectures.md)

---

*This document covers approaches to modeling environments for autonomous agents. For implementation examples, refer to simulation frameworks and the code samples provided.*