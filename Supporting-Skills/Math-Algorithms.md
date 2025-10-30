# 📐 Math & Algorithms

## 📋 Overview

Mathematical foundations are essential for understanding and implementing agent systems. This guide covers key mathematical concepts and algorithmic techniques used in AI agent development.

## 🔢 Linear Algebra

### Essential Concepts

**Vectors & Matrices** - State representations, transformations

```python
import numpy as np

# State vector
state = np.array([1.0, 2.0, 3.0, 4.0])

# Transition matrix (Markov chain)
P = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.3, 0.4, 0.2, 0.1],
    [0.0, 0.3, 0.5, 0.2],
    [0.0, 0.0, 0.4, 0.6]
])

# Next state distribution
next_state = P @ state
print(f"Next state: {next_state}")

# Steady state (eigenvalue problem)
eigenvalues, eigenvectors = np.linalg.eig(P.T)
steady_state = eigenvectors[:, np.argmax(eigenvalues)]
steady_state = steady_state / steady_state.sum()
print(f"Steady state: {steady_state}")
```

**Applications in Agents**:
- State space representation
- Policy matrices
- Value function approximation
- Dimensionality reduction (PCA, SVD)

## 📊 Probability & Statistics

### Probability Distributions

```python
import numpy as np
from scipy import stats

# Gaussian distribution for continuous actions
action_mean = 0.0
action_std = 1.0
action = np.random.normal(action_mean, action_std)

# Categorical distribution for discrete actions
action_probs = np.array([0.2, 0.5, 0.3])  # 3 actions
action = np.random.choice(len(action_probs), p=action_probs)

# Beta distribution for exploration
alpha, beta = 2, 5
exploration_rate = stats.beta.rvs(alpha, beta)
```

### Bayesian Inference

```python
class BayesianAgent:
    """Agent with Bayesian belief updating"""
    
    def __init__(self, prior_alpha=1, prior_beta=1):
        # Beta distribution for binary outcomes
        self.alpha = prior_alpha
        self.beta = prior_beta
    
    def update_belief(self, success: bool):
        """Update posterior after observation"""
        if success:
            self.alpha += 1
        else:
            self.beta += 1
    
    def get_estimate(self):
        """Get current estimate (mean of Beta)"""
        return self.alpha / (self.alpha + self.beta)
    
    def sample_action(self):
        """Thompson sampling"""
        theta = np.random.beta(self.alpha, self.beta)
        return theta > 0.5

# Multi-armed bandit with Thompson sampling
agents = [BayesianAgent() for _ in range(3)]

for trial in range(100):
    # Sample from each arm
    samples = [np.random.beta(a.alpha, a.beta) for a in agents]
    
    # Choose arm with highest sample
    chosen_arm = np.argmax(samples)
    
    # Simulate reward
    true_probs = [0.3, 0.5, 0.7]
    reward = np.random.random() < true_probs[chosen_arm]
    
    # Update belief
    agents[chosen_arm].update_belief(reward)
```

## 🎯 Optimization

### Gradient Descent

```python
def gradient_descent(f, grad_f, x0, lr=0.01, max_iters=1000):
    """Basic gradient descent"""
    x = x0
    history = [x]
    
    for i in range(max_iters):
        gradient = grad_f(x)
        x = x - lr * gradient
        history.append(x)
        
        if np.linalg.norm(gradient) < 1e-6:
            break
    
    return x, history

# Example: minimize f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1

def grad_f(x):
    return 2*x + 2

optimal_x, _ = gradient_descent(f, grad_f, x0=5.0)
print(f"Optimal x: {optimal_x}")  # Should be -1.0
```

### Convex Optimization

```python
from scipy.optimize import minimize

def objective(x):
    """Quadratic objective"""
    return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

def constraint1(x):
    """Constraint: x0 + x1 >= 1"""
    return x[0] + x[1] - 1

# Optimization with constraints
result = minimize(
    objective,
    x0=[0, 0],
    method='SLSQP',
    constraints={'type': 'ineq', 'fun': constraint1}
)

print(f"Optimal solution: {result.x}")
print(f"Optimal value: {result.fun}")
```

## 📈 Calculus

### Automatic Differentiation

```python
import torch

# Define function
def f(x):
    return x**3 + 2*x**2 - 5*x + 1

# Compute gradient using autograd
x = torch.tensor(2.0, requires_grad=True)
y = f(x)
y.backward()

print(f"f'(2) = {x.grad}")  # Should be 3*4 + 4*2 - 5 = 15
```

### Policy Gradient

```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)

def compute_policy_gradient(policy, states, actions, rewards):
    """REINFORCE algorithm"""
    log_probs = []
    
    for state, action in zip(states, actions):
        probs = policy(state)
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob)
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    
    # Policy gradient
    loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns))
    
    return loss
```

## 🕸️ Graph Theory

### Graph Algorithms

```python
import networkx as nx

# Create graph
G = nx.Graph()
G.add_edges_from([
    ('A', 'B', {'weight': 4}),
    ('A', 'C', {'weight': 2}),
    ('B', 'C', {'weight': 1}),
    ('B', 'D', {'weight': 5}),
    ('C', 'D', {'weight': 3})
])

# Shortest path (Dijkstra)
path = nx.shortest_path(G, 'A', 'D', weight='weight')
print(f"Shortest path: {path}")

# Minimum spanning tree
mst = nx.minimum_spanning_tree(G)
print(f"MST edges: {list(mst.edges())}")

# PageRank (for agent importance)
pagerank = nx.pagerank(G)
print(f"PageRank scores: {pagerank}")
```

### Multi-Agent Communication Graph

```python
def analyze_communication_topology(agents):
    """Analyze agent communication structure"""
    G = nx.DiGraph()
    
    # Build communication graph
    for agent in agents:
        for target, messages in agent.message_history.items():
            if messages:
                G.add_edge(agent.id, target, weight=len(messages))
    
    # Analyze structure
    metrics = {
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G.to_undirected()),
        'diameter': nx.diameter(G) if nx.is_strongly_connected(G) else float('inf'),
        'central_agents': nx.betweenness_centrality(G)
    }
    
    return metrics
```

## 📏 Algorithm Analysis

### Time Complexity

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| **Binary Search** | O(1) | O(log n) | O(log n) | O(1) |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) |
| **A* Search** | O(b^d) | O(b^d) | O(b^d) | O(b^d) |
| **Dijkstra** | O(E + V log V) | O(E + V log V) | O(E + V log V) | O(V) |

### Dynamic Programming

```python
def optimal_policy_dp(mdp, gamma=0.99, theta=1e-6):
    """Value iteration for MDP"""
    V = {s: 0 for s in mdp.states}
    
    while True:
        delta = 0
        
        for s in mdp.states:
            v = V[s]
            
            # Bellman update
            action_values = []
            for a in mdp.actions(s):
                q = sum(
                    p * (r + gamma * V[s_next])
                    for s_next, p, r in mdp.transitions(s, a)
                )
                action_values.append(q)
            
            V[s] = max(action_values) if action_values else 0
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract policy
    policy = {}
    for s in mdp.states:
        action_values = []
        for a in mdp.actions(s):
            q = sum(
                p * (r + gamma * V[s_next])
                for s_next, p, r in mdp.transitions(s, a)
            )
            action_values.append((q, a))
        
        policy[s] = max(action_values)[1] if action_values else None
    
    return policy, V
```

## 🔢 Information Theory

### Entropy & KL Divergence

```python
import numpy as np

def entropy(p):
    """Shannon entropy H(P)"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

def kl_divergence(p, q):
    """KL(P||Q)"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log2(p / q))

# Example: action distribution
p = np.array([0.7, 0.2, 0.1])
q = np.array([0.4, 0.4, 0.2])

print(f"H(P) = {entropy(p):.3f} bits")
print(f"KL(P||Q) = {kl_divergence(p, q):.3f}")
```

### Mutual Information

```python
def mutual_information(X, Y):
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(X, Y)

# Agent state-action dependency
states = [0, 0, 1, 1, 2, 2]
actions = [0, 1, 1, 2, 2, 2]

mi = mutual_information(states, actions)
print(f"I(State;Action) = {mi:.3f}")
```

## 📚 Resources

### Books
- **"Mathematics for Machine Learning"** by Deisenroth et al.
- **"Introduction to Algorithms"** by CLRS
- **"Convex Optimization"** by Boyd & Vandenberghe

### Online Courses
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Stanford CS229 Math Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)

### Tools
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing
- [SymPy](https://www.sympy.org/) - Symbolic mathematics

## 🔗 Related Topics

- [Reinforcement Learning](../Core-Concepts/Reinforcement-Learning.md)
- [Decision Making & Planning](../Core-Concepts/Decision-Making-Planning.md)
- [Programming Fundamentals](./Programming-Fundamentals.md)
- [Agent Evaluation](../Agent-Evaluation-Benchmarking/Metrics-Methods.md)

---

*This guide covers mathematical foundations for agent systems. For implementation details, see Programming Fundamentals.*