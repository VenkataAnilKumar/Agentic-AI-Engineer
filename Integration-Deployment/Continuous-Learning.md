# 🔄 Continuous Learning

## 📋 Overview

Continuous learning enables agents to adapt and improve over time through ongoing interaction with their environment. This guide covers online learning algorithms, model updating strategies, and adaptive systems for production agents.

## 📖 Online Learning

### Online Learning vs Batch Learning

| Aspect | Online Learning | Batch Learning |
|--------|----------------|----------------|
| **Data** | Streams, one at a time | Fixed dataset |
| **Updates** | Incremental | Periodic retraining |
| **Memory** | Constant | Grows with data |
| **Adaptation** | Real-time | Delayed |

### Online Gradient Descent

```python
import numpy as np
from typing import Callable

class OnlineLearner:
    """Online learning with SGD"""
    
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate
        self.n_updates = 0
    
    def predict(self, x: np.ndarray) -> float:
        """Make prediction"""
        return np.dot(x, self.weights) + self.bias
    
    def update(self, x: np.ndarray, y_true: float):
        """Update model with single example"""
        # Prediction
        y_pred = self.predict(x)
        
        # Compute error
        error = y_true - y_pred
        
        # Update weights (gradient descent)
        self.weights += self.lr * error * x
        self.bias += self.lr * error
        
        self.n_updates += 1
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Update with mini-batch"""
        for x_i, y_i in zip(X, y):
            self.update(x_i, y_i)

# Usage
learner = OnlineLearner(n_features=4)

# Stream data
for x, y in data_stream:
    prediction = learner.predict(x)
    learner.update(x, y)
```

### Experience Replay for Stability

```python
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for online RL"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample(self, batch_size: int):
        """Sample random batch"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Online RL with replay
replay_buffer = ReplayBuffer(capacity=100000)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Agent acts
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Store experience
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Learn from replay buffer
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.update(*batch)
        
        state = next_state
```

## 🔄 Incremental Learning

### Incremental Naive Bayes

```python
from collections import defaultdict
import numpy as np

class IncrementalNaiveBayes:
    """Incremental Naive Bayes classifier"""
    
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0
    
    def partial_fit(self, X, y):
        """Update model incrementally"""
        for features, label in zip(X, y):
            self.class_counts[label] += 1
            
            for feature_idx, feature_value in enumerate(features):
                self.feature_counts[label][feature_idx] += feature_value
            
            self.total_samples += 1
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        probas = []
        
        for features in X:
            class_probas = {}
            
            for class_label in self.class_counts:
                # Prior probability
                prior = self.class_counts[class_label] / self.total_samples
                
                # Likelihood
                likelihood = 1.0
                for idx, value in enumerate(features):
                    feature_prob = (
                        self.feature_counts[class_label][idx] / 
                        self.class_counts[class_label]
                    )
                    likelihood *= feature_prob ** value
                
                class_probas[class_label] = prior * likelihood
            
            # Normalize
            total = sum(class_probas.values())
            class_probas = {k: v/total for k, v in class_probas.items()}
            
            probas.append(class_probas)
        
        return probas
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return [max(p.items(), key=lambda x: x[1])[0] for p in probas]
```

## 🎯 Transfer Learning

### Fine-tuning Pre-trained Models

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TransferLearningAgent:
    """Agent with transfer learning capabilities"""
    
    def __init__(self, base_model_name: str, num_classes: int):
        # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add task-specific head
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Task-specific classification
        logits = self.classifier(pooled_output)
        return logits
    
    def unfreeze_base_model(self, num_layers: int = 2):
        """Unfreeze last N layers for fine-tuning"""
        # Unfreeze last N encoder layers
        for layer in self.base_model.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

# Progressive fine-tuning
agent = TransferLearningAgent("bert-base-uncased", num_classes=5)

# Stage 1: Train only classifier head
train(agent, epochs=5, lr=1e-3)

# Stage 2: Fine-tune last 2 layers
agent.unfreeze_base_model(num_layers=2)
train(agent, epochs=3, lr=1e-4)

# Stage 3: Fine-tune entire model
agent.unfreeze_base_model(num_layers=12)
train(agent, epochs=2, lr=1e-5)
```

## 🧠 Meta-Learning

### Model-Agnostic Meta-Learning (MAML)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MAMLAgent(nn.Module):
    """MAML for fast adaptation"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def adapt(self, support_x, support_y, steps=5, lr=0.01):
        """Adapt to new task using support set"""
        # Clone parameters
        adapted_params = {
            name: param.clone() 
            for name, param in self.named_parameters()
        }
        
        # Inner loop: adapt on support set
        for step in range(steps):
            # Forward pass with adapted parameters
            pred = self.forward(support_x)
            loss = nn.functional.mse_loss(pred, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, 
                self.parameters(),
                create_graph=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - lr * grad
        
        return adapted_params

# Meta-training loop
meta_model = MAMLAgent(input_dim=10, hidden_dim=64, output_dim=1)
meta_optimizer = optim.Adam(meta_model.parameters(), lr=1e-3)

for meta_iteration in range(1000):
    # Sample batch of tasks
    tasks = sample_tasks(batch_size=32)
    
    meta_loss = 0
    for task in tasks:
        # Get support and query sets
        support_x, support_y = task.support_set()
        query_x, query_y = task.query_set()
        
        # Adapt to task
        adapted_params = meta_model.adapt(support_x, support_y)
        
        # Evaluate on query set
        query_pred = meta_model(query_x)
        task_loss = nn.functional.mse_loss(query_pred, query_y)
        
        meta_loss += task_loss
    
    # Meta-update
    meta_loss /= len(tasks)
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
```

## 📚 Curriculum Learning

### Curriculum Learning Strategy

```python
class CurriculumLearning:
    """Curriculum learning for agent training"""
    
    def __init__(self, tasks, difficulty_scorer):
        self.tasks = tasks
        self.difficulty_scorer = difficulty_scorer
        self.performance_history = []
    
    def get_next_task(self, agent_performance: float):
        """Select next task based on current performance"""
        # Sort tasks by difficulty
        sorted_tasks = sorted(
            self.tasks,
            key=lambda t: self.difficulty_scorer(t)
        )
        
        # Progressive difficulty
        if agent_performance < 0.3:
            # Start with easy tasks
            return sorted_tasks[0]
        elif agent_performance < 0.6:
            # Medium difficulty
            mid_idx = len(sorted_tasks) // 2
            return sorted_tasks[mid_idx]
        else:
            # Hard tasks
            return sorted_tasks[-1]
    
    def update_curriculum(self, performance: float):
        """Update curriculum based on performance"""
        self.performance_history.append(performance)
        
        # Adapt curriculum based on learning curve
        if len(self.performance_history) > 10:
            recent_perf = np.mean(self.performance_history[-10:])
            
            if recent_perf > 0.8:
                # Increase difficulty
                self.tasks = self.add_harder_tasks(self.tasks)
            elif recent_perf < 0.3:
                # Add more easier tasks
                self.tasks = self.add_easier_tasks(self.tasks)

# Usage
curriculum = CurriculumLearning(tasks, difficulty_scorer)

for episode in range(1000):
    # Get appropriate task
    current_performance = agent.evaluate()
    task = curriculum.get_next_task(current_performance)
    
    # Train on task
    agent.train_on_task(task)
    
    # Update curriculum
    curriculum.update_curriculum(current_performance)
```

## 🎲 Active Learning

### Uncertainty Sampling

```python
class ActiveLearningAgent:
    """Agent with active learning for data efficiency"""
    
    def __init__(self, model, uncertainty_threshold=0.5):
        self.model = model
        self.threshold = uncertainty_threshold
        self.labeled_data = []
        self.unlabeled_pool = []
    
    def uncertainty_score(self, x):
        """Compute prediction uncertainty"""
        # Get model predictions
        probs = self.model.predict_proba([x])[0]
        
        # Entropy-based uncertainty
        entropy = -sum(p * np.log(p + 1e-10) for p in probs.values())
        
        return entropy
    
    def query_strategy(self, n_samples=10):
        """Select most informative samples to label"""
        # Compute uncertainty for unlabeled data
        uncertainties = [
            (x, self.uncertainty_score(x))
            for x in self.unlabeled_pool
        ]
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(key=lambda item: item[1], reverse=True)
        
        # Return top uncertain samples
        return [x for x, _ in uncertainties[:n_samples]]
    
    def train_with_active_learning(self, oracle, budget=100):
        """Train with active learning"""
        labeled_count = 0
        
        while labeled_count < budget:
            # Train on current labeled data
            if self.labeled_data:
                X = [x for x, y in self.labeled_data]
                y = [y for x, y in self.labeled_data]
                self.model.partial_fit(X, y)
            
            # Query most uncertain samples
            queries = self.query_strategy(n_samples=10)
            
            # Get labels from oracle
            for x in queries:
                if labeled_count >= budget:
                    break
                
                y = oracle.label(x)
                self.labeled_data.append((x, y))
                self.unlabeled_pool.remove(x)
                labeled_count += 1
```

## 🛠️ Model Updating Strategies

### Safe Model Updates

```python
class SafeModelUpdater:
    """Safely update production models"""
    
    def __init__(self, current_model, validation_data):
        self.current_model = current_model
        self.validation_data = validation_data
        self.performance_history = []
    
    def should_update(self, new_model, min_improvement=0.05):
        """Decide whether to update model"""
        # Evaluate current model
        current_perf = self.evaluate(self.current_model)
        
        # Evaluate new model
        new_perf = self.evaluate(new_model)
        
        # Check improvement
        improvement = (new_perf - current_perf) / current_perf
        
        return improvement > min_improvement
    
    def canary_deployment(self, new_model, traffic_split=0.1):
        """Gradual rollout with canary deployment"""
        # Route small traffic to new model
        for x, y in self.validation_data:
            if random.random() < traffic_split:
                pred = new_model.predict(x)
            else:
                pred = self.current_model.predict(x)
            
            # Monitor performance
            self.log_prediction(pred, y)
        
        # Analyze canary metrics
        canary_success = self.analyze_canary_metrics()
        
        if canary_success:
            # Increase traffic gradually
            self.gradual_rollout(new_model)
        else:
            # Rollback
            self.rollback()
    
    def evaluate(self, model):
        """Evaluate model performance"""
        correct = 0
        for x, y in self.validation_data:
            pred = model.predict(x)
            if pred == y:
                correct += 1
        return correct / len(self.validation_data)
```

## 📚 Learning Resources

### Books
- **"Online Learning and Online Convex Optimization"** by Shalev-Shwartz
- **"Meta-Learning in Neural Networks"** by Hospedales et al.
- **"Active Learning"** by Settles

### Papers
- "Model-Agnostic Meta-Learning (MAML)" - Finn et al.
- "Curriculum Learning" - Bengio et al.
- "Active Learning Literature Survey" - Settles

## 🔗 Related Topics

- [Reinforcement Learning](../Core-Concepts/Reinforcement-Learning.md)
- [Agent Evaluation](../Agent-Evaluation-Benchmarking/Metrics-Methods.md)
- [Cloud Platforms](./Cloud-Platforms.md)
- [Automation & Scheduling](./Automation-Scheduling.md)

---

*This guide covers continuous learning strategies for adaptive agent systems. For evaluation metrics, see the Agent Evaluation guide.*