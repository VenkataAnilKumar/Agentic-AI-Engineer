# 📊 Metrics & Methods

## 📋 Overview

Effective evaluation of agent systems requires comprehensive metrics covering performance, efficiency, robustness, and generalization. This guide provides frameworks for measuring agent capabilities.

## 🎯 Success Rate Metrics

### Task Success Rate

```python
import numpy as np
from typing import List, Dict

class SuccessRateMetrics:
    """Measure task completion success"""
    
    def __init__(self):
        self.results = []
    
    def record_episode(self, success: bool, steps: int, reward: float):
        """Record episode outcome"""
        self.results.append({
            'success': success,
            'steps': steps,
            'reward': reward
        })
    
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.results:
            return 0.0
        
        successes = sum(1 for r in self.results if r['success'])
        return successes / len(self.results)
    
    def average_steps_to_success(self) -> float:
        """Average steps for successful episodes"""
        successful = [r['steps'] for r in self.results if r['success']]
        return np.mean(successful) if successful else float('inf')
    
    def success_rate_over_time(self, window_size: int = 100) -> List[float]:
        """Rolling success rate"""
        rates = []
        
        for i in range(len(self.results) - window_size + 1):
            window = self.results[i:i + window_size]
            successes = sum(1 for r in window if r['success'])
            rates.append(successes / window_size)
        
        return rates
    
    def first_success_episode(self) -> int:
        """Episode number of first success"""
        for i, result in enumerate(self.results):
            if result['success']:
                return i
        return -1

# Usage
metrics = SuccessRateMetrics()

for episode in range(1000):
    success, steps, reward = run_episode(agent, env)
    metrics.record_episode(success, steps, reward)

print(f"Success Rate: {metrics.success_rate():.2%}")
print(f"Avg Steps to Success: {metrics.average_steps_to_success():.1f}")
```

### Partial Credit Metrics

```python
class PartialCreditEvaluator:
    """Evaluate with partial credit"""
    
    def evaluate_task_completion(self, agent_output: dict, target: dict) -> float:
        """
        Evaluate with partial credit for multi-step tasks
        Returns score between 0 and 1
        """
        subtask_scores = []
        
        for subtask_id, subtask_target in target.items():
            if subtask_id in agent_output:
                score = self._evaluate_subtask(
                    agent_output[subtask_id],
                    subtask_target
                )
                subtask_scores.append(score)
            else:
                subtask_scores.append(0.0)
        
        return np.mean(subtask_scores)
    
    def _evaluate_subtask(self, output, target) -> float:
        """Evaluate individual subtask"""
        if isinstance(target, dict):
            # Structured output
            correct_keys = sum(
                1 for k in target.keys()
                if k in output and output[k] == target[k]
            )
            return correct_keys / len(target)
        else:
            # Binary correct/incorrect
            return 1.0 if output == target else 0.0
```

## ⚡ Efficiency Metrics

### Sample Efficiency

```python
class SampleEfficiencyMetrics:
    """Measure learning efficiency"""
    
    def __init__(self):
        self.episode_rewards = []
        self.cumulative_samples = []
    
    def record_training_step(self, reward: float, num_samples: int):
        """Record training data"""
        self.episode_rewards.append(reward)
        
        total_samples = (self.cumulative_samples[-1] if self.cumulative_samples else 0) + num_samples
        self.cumulative_samples.append(total_samples)
    
    def samples_to_threshold(self, threshold: float) -> int:
        """Samples needed to reach performance threshold"""
        for i, reward in enumerate(self.episode_rewards):
            if reward >= threshold:
                return self.cumulative_samples[i]
        return -1
    
    def area_under_curve(self) -> float:
        """AUC of reward vs samples curve"""
        return np.trapz(self.episode_rewards, self.cumulative_samples)
    
    def regret(self, optimal_reward: float) -> float:
        """Cumulative regret"""
        return sum(optimal_reward - r for r in self.episode_rewards)

# Compare algorithms
results = {
    'DQN': SampleEfficiencyMetrics(),
    'PPO': SampleEfficiencyMetrics(),
    'SAC': SampleEfficiencyMetrics()
}

# Train and compare
for algo_name, metrics in results.items():
    agent = create_agent(algo_name)
    for episode in range(1000):
        reward, samples = train_episode(agent)
        metrics.record_training_step(reward, samples)

# Report
threshold = 200
for algo, metrics in results.items():
    samples = metrics.samples_to_threshold(threshold)
    print(f"{algo}: {samples} samples to reach reward {threshold}")
```

### Computational Efficiency

```python
import time
import psutil
import torch

class ComputationalMetrics:
    """Measure computational resources"""
    
    def __init__(self):
        self.measurements = []
    
    def measure_inference(self, agent, state, num_runs: int = 100):
        """Measure inference time and memory"""
        process = psutil.Process()
        
        # Warmup
        for _ in range(10):
            agent.act(state)
        
        # Measure
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        for _ in range(num_runs):
            action = agent.act(state)
        
        duration = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        
        return {
            'avg_inference_time': duration / num_runs * 1000,  # ms
            'throughput': num_runs / duration,  # actions/sec
            'memory_used': end_memory - start_memory,  # MB
            'peak_memory': end_memory
        }
    
    def measure_training(self, agent, env, episodes: int):
        """Measure training computational cost"""
        start_time = time.time()
        initial_memory = psutil.virtual_memory().used / 1024 / 1024 / 1024  # GB
        
        for episode in range(episodes):
            agent.train_episode(env)
        
        training_time = time.time() - start_time
        final_memory = psutil.virtual_memory().used / 1024 / 1024 / 1024
        
        return {
            'total_training_time': training_time,
            'time_per_episode': training_time / episodes,
            'memory_used': final_memory - initial_memory,
            'gpu_utilization': self._get_gpu_utilization()
        }
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization if available"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0
```

## 📈 Scalability Metrics

### Multi-Agent Scalability

```python
class ScalabilityMetrics:
    """Measure system scalability"""
    
    def test_agent_scaling(self, num_agents_list: List[int], env_factory):
        """Test performance as agent count increases"""
        results = []
        
        for num_agents in num_agents_list:
            env = env_factory(num_agents=num_agents)
            agents = [create_agent() for _ in range(num_agents)]
            
            start = time.time()
            episode_reward = self._run_multi_agent_episode(agents, env)
            duration = time.time() - start
            
            results.append({
                'num_agents': num_agents,
                'reward': episode_reward,
                'time': duration,
                'agents_per_second': num_agents / duration
            })
        
        return results
    
    def _run_multi_agent_episode(self, agents: List, env) -> float:
        """Run single multi-agent episode"""
        states = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            states, rewards, done, _ = env.step(actions)
            total_reward += sum(rewards)
        
        return total_reward

# Test scalability
scalability = ScalabilityMetrics()
results = scalability.test_agent_scaling(
    num_agents_list=[1, 2, 5, 10, 20, 50, 100],
    env_factory=create_environment
)

# Analyze
for result in results:
    print(f"{result['num_agents']} agents: {result['time']:.2f}s, {result['agents_per_second']:.1f} agents/sec")
```

## 🛡️ Robustness Evaluation

### Adversarial Robustness

```python
class RobustnessMetrics:
    """Evaluate agent robustness"""
    
    def test_noise_robustness(
        self,
        agent,
        env,
        noise_levels: List[float],
        episodes: int = 100
    ):
        """Test robustness to observation noise"""
        results = {}
        
        for noise_level in noise_levels:
            rewards = []
            
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    # Add Gaussian noise
                    noisy_state = state + np.random.normal(0, noise_level, state.shape)
                    action = agent.act(noisy_state)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                
                rewards.append(episode_reward)
            
            results[noise_level] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'performance_drop': 1 - (np.mean(rewards) / self.baseline_reward)
            }
        
        return results
    
    def test_distribution_shift(self, agent, train_env, test_envs: Dict):
        """Test generalization to different environments"""
        results = {}
        
        for env_name, test_env in test_envs.items():
            rewards = []
            
            for _ in range(100):
                episode_reward = self._run_episode(agent, test_env)
                rewards.append(episode_reward)
            
            results[env_name] = {
                'mean_reward': np.mean(rewards),
                'transfer_ratio': np.mean(rewards) / self.train_reward
            }
        
        return results
```

## 🌍 Generalization Metrics

### Zero-Shot & Few-Shot Performance

```python
class GeneralizationMetrics:
    """Measure generalization capabilities"""
    
    def zero_shot_evaluation(self, agent, novel_tasks: List):
        """Evaluate on unseen tasks without adaptation"""
        results = []
        
        for task in novel_tasks:
            env = create_environment(task)
            success_rate = self._evaluate_task(agent, env, episodes=50)
            
            results.append({
                'task': task.name,
                'success_rate': success_rate,
                'difficulty': task.difficulty
            })
        
        return results
    
    def few_shot_adaptation(
        self,
        agent,
        novel_task,
        num_examples: int = 10
    ):
        """Measure adaptation speed with few examples"""
        env = create_environment(novel_task)
        
        # Baseline (no adaptation)
        baseline = self._evaluate_task(agent, env, episodes=20)
        
        # Adapt with few examples
        for _ in range(num_examples):
            agent.adapt_episode(env)
        
        # Evaluate after adaptation
        adapted = self._evaluate_task(agent, env, episodes=20)
        
        return {
            'baseline': baseline,
            'adapted': adapted,
            'improvement': adapted - baseline
        }
    
    def cross_domain_transfer(self, agent, source_domain, target_domains: List):
        """Test transfer across domains"""
        # Train on source
        train_performance = self._train_agent(agent, source_domain)
        
        # Test on targets
        transfer_results = {}
        
        for target in target_domains:
            env = create_environment(target)
            performance = self._evaluate_task(agent, env, episodes=100)
            
            transfer_results[target.name] = {
                'performance': performance,
                'transfer_efficiency': performance / train_performance
            }
        
        return transfer_results
```

## 📊 Multi-Objective Evaluation

### Pareto Frontier Analysis

```python
class MultiObjectiveMetrics:
    """Evaluate multiple objectives simultaneously"""
    
    def __init__(self, objectives: List[str]):
        self.objectives = objectives
        self.results = []
    
    def record_agent(self, agent_name: str, scores: Dict[str, float]):
        """Record agent scores on all objectives"""
        self.results.append({
            'agent': agent_name,
            'scores': scores
        })
    
    def find_pareto_frontier(self) -> List[Dict]:
        """Find Pareto-optimal agents"""
        pareto_optimal = []
        
        for i, agent_i in enumerate(self.results):
            is_dominated = False
            
            for j, agent_j in enumerate(self.results):
                if i != j and self._dominates(agent_j, agent_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(agent_i)
        
        return pareto_optimal
    
    def _dominates(self, agent_a: Dict, agent_b: Dict) -> bool:
        """Check if agent_a dominates agent_b"""
        better_in_any = False
        
        for objective in self.objectives:
            score_a = agent_a['scores'][objective]
            score_b = agent_b['scores'][objective]
            
            if score_a < score_b:
                return False
            if score_a > score_b:
                better_in_any = True
        
        return better_in_any

# Example: Evaluate agents on multiple criteria
evaluator = MultiObjectiveMetrics(['reward', 'safety', 'efficiency'])

evaluator.record_agent('AgentA', {'reward': 100, 'safety': 0.95, 'efficiency': 80})
evaluator.record_agent('AgentB', {'reward': 120, 'safety': 0.85, 'efficiency': 60})
evaluator.record_agent('AgentC', {'reward': 90, 'safety': 0.99, 'efficiency': 90})

pareto_optimal = evaluator.find_pareto_frontier()
print("Pareto-optimal agents:", [a['agent'] for a in pareto_optimal])
```

## 📚 Resources

### Benchmarking Papers
- **"Benchmarking Deep Reinforcement Learning"** (OpenAI, 2016)
- **"AgentBench: Evaluating LLMs as Agents"** (2023)
- **"On the Measure of Intelligence"** (Chollet, 2019)

### Tools
- **Weights & Biases** - Experiment tracking
- **TensorBoard** - Visualization
- **MLflow** - ML lifecycle management
- **Ray Tune** - Hyperparameter tuning

## 🔗 Related Topics

- [Benchmarks & Datasets](./Benchmarks-Datasets.md)
- [Evaluation Frameworks](./Evaluation-Frameworks.md)
- [Continuous Learning](../Integration-Deployment/Continuous-Learning.md)
- [Real-World Applications](../Applications-Case-Studies/Real-World-Applications.md)

---

*This guide covers performance metrics for agent evaluation. For standard benchmarks, see Benchmarks & Datasets.*