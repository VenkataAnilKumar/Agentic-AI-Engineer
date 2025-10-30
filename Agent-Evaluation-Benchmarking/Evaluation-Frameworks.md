# 🔬 Evaluation Frameworks

## 📋 Overview

Rigorous testing and validation are critical for deploying reliable agent systems. This guide covers comprehensive testing frameworks from unit tests to production monitoring.

## 🧪 Unit Testing for Agents

### Testing Agent Components

```python
import unittest
import numpy as np
from unittest.mock import Mock, MagicMock

class TestAgentComponents(unittest.TestCase):
    """Unit tests for agent components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = DQNAgent(state_dim=4, action_dim=2, learning_rate=0.001)
        self.mock_env = Mock()
    
    def test_action_selection(self):
        """Test action selection logic"""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Test greedy action
        action = self.agent.select_action(state, epsilon=0.0)
        self.assertIn(action, [0, 1])
        self.assertIsInstance(action, int)
        
        # Test random action
        np.random.seed(42)
        action = self.agent.select_action(state, epsilon=1.0)
        self.assertIn(action, [0, 1])
    
    def test_q_value_update(self):
        """Test Q-value updates"""
        initial_q = self.agent.get_q_values(np.zeros(4)).copy()
        
        # Perform update
        self.agent.update(
            state=np.array([0.0, 0.0, 0.0, 0.0]),
            action=0,
            reward=1.0,
            next_state=np.array([1.0, 0.0, 0.0, 0.0]),
            done=False
        )
        
        updated_q = self.agent.get_q_values(np.zeros(4))
        
        # Q-values should change
        self.assertFalse(np.allclose(initial_q, updated_q))
    
    def test_replay_buffer(self):
        """Test experience replay buffer"""
        buffer = self.agent.replay_buffer
        
        # Add experience
        buffer.add(
            state=np.zeros(4),
            action=0,
            reward=1.0,
            next_state=np.ones(4),
            done=False
        )
        
        self.assertEqual(len(buffer), 1)
        
        # Sample batch
        if len(buffer) >= buffer.batch_size:
            batch = buffer.sample()
            self.assertEqual(len(batch['states']), buffer.batch_size)
    
    def test_policy_consistency(self):
        """Test policy returns consistent actions for same state"""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Deterministic policy should be consistent
        action1 = self.agent.select_action(state, epsilon=0.0)
        action2 = self.agent.select_action(state, epsilon=0.0)
        
        self.assertEqual(action1, action2)
    
    def test_reward_clipping(self):
        """Test reward clipping"""
        # Test large positive reward
        clipped = self.agent.clip_reward(100.0)
        self.assertEqual(clipped, 1.0)
        
        # Test large negative reward
        clipped = self.agent.clip_reward(-100.0)
        self.assertEqual(clipped, -1.0)
    
    def tearDown(self):
        """Clean up"""
        self.agent = None

if __name__ == '__main__':
    unittest.main()
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

class TestAgentProperties(unittest.TestCase):
    """Property-based tests"""
    
    @given(state=npst.arrays(dtype=np.float32, shape=4, elements=st.floats(-10, 10)))
    def test_q_values_bounded(self, state):
        """Q-values should be bounded"""
        agent = DQNAgent(state_dim=4, action_dim=2)
        q_values = agent.get_q_values(state)
        
        # Check shape
        self.assertEqual(q_values.shape, (2,))
        
        # Check bounds (if using tanh output)
        self.assertTrue(np.all(np.isfinite(q_values)))
    
    @given(
        epsilon=st.floats(min_value=0.0, max_value=1.0),
        state=npst.arrays(dtype=np.float32, shape=4, elements=st.floats(-10, 10))
    )
    def test_exploration_rate(self, epsilon, state):
        """Higher epsilon should increase randomness"""
        agent = DQNAgent(state_dim=4, action_dim=2)
        
        # Run multiple times
        actions = [agent.select_action(state, epsilon=epsilon) for _ in range(100)]
        
        if epsilon > 0.5:
            # Should see both actions with high epsilon
            unique_actions = len(set(actions))
            self.assertGreater(unique_actions, 1)
```

## 🔗 Integration Testing

### End-to-End Testing

```python
import pytest

class TestAgentEnvironmentIntegration:
    """Integration tests for agent-environment interaction"""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment and agent"""
        env = gym.make('CartPole-v1')
        agent = DQNAgent(state_dim=4, action_dim=2)
        return env, agent
    
    def test_full_episode(self, setup):
        """Test complete episode execution"""
        env, agent = setup
        
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 500:
            # Agent acts
            action = agent.select_action(state)
            
            # Environment responds
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Agent learns
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Assertions
        assert steps > 0, "Episode should run for at least one step"
        assert isinstance(total_reward, (int, float)), "Reward should be numeric"
        assert done, "Episode should terminate"
    
    def test_training_improves_performance(self, setup):
        """Test that training improves agent performance"""
        env, agent = setup
        
        # Measure initial performance
        initial_reward = self._evaluate_agent(agent, env, episodes=10)
        
        # Train agent
        for _ in range(100):
            self._train_episode(agent, env)
        
        # Measure final performance
        final_reward = self._evaluate_agent(agent, env, episodes=10)
        
        # Performance should improve
        assert final_reward > initial_reward, "Training should improve performance"
    
    def _train_episode(self, agent, env):
        """Helper: Train one episode"""
        state, _ = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
    
    def _evaluate_agent(self, agent, env, episodes=10):
        """Helper: Evaluate agent performance"""
        total_rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, epsilon=0.0)  # Greedy
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
```

## 🎮 Simulation-Based Testing

### Monte Carlo Testing

```python
class SimulationTester:
    """Monte Carlo simulation testing"""
    
    def __init__(self, agent, env, num_simulations=1000):
        self.agent = agent
        self.env = env
        self.num_simulations = num_simulations
    
    def test_robustness(self, noise_levels=[0.0, 0.1, 0.2, 0.5]):
        """Test agent robustness across noise levels"""
        results = {}
        
        for noise in noise_levels:
            rewards = []
            
            for _ in range(self.num_simulations):
                reward = self._simulate_with_noise(noise)
                rewards.append(reward)
            
            results[noise] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'percentile_25': np.percentile(rewards, 25),
                'percentile_75': np.percentile(rewards, 75)
            }
        
        return results
    
    def _simulate_with_noise(self, noise_level):
        """Run single simulation with noise"""
        state = self.env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            # Add noise to observations
            noisy_state = state + np.random.normal(0, noise_level, state.shape)
            action = self.agent.select_action(noisy_state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        return total_reward
    
    def test_edge_cases(self):
        """Test behavior in edge cases"""
        edge_cases = {
            'zero_state': np.zeros(self.env.observation_space.shape),
            'max_state': np.ones(self.env.observation_space.shape) * 10,
            'negative_state': np.ones(self.env.observation_space.shape) * -10,
            'nan_state': np.full(self.env.observation_space.shape, np.nan),
            'inf_state': np.full(self.env.observation_space.shape, np.inf)
        }
        
        results = {}
        
        for case_name, state in edge_cases.items():
            try:
                action = self.agent.select_action(state)
                results[case_name] = {'success': True, 'action': action}
            except Exception as e:
                results[case_name] = {'success': False, 'error': str(e)}
        
        return results
```

## 🔀 A/B Testing

### Online A/B Testing Framework

```python
from scipy import stats

class ABTester:
    """A/B testing for agent policies"""
    
    def __init__(self):
        self.variant_a_results = []
        self.variant_b_results = []
    
    def run_experiment(
        self,
        agent_a,
        agent_b,
        env,
        num_episodes=1000,
        allocation_ratio=0.5
    ):
        """Run A/B test"""
        for episode in range(num_episodes):
            # Randomly assign variant
            if np.random.random() < allocation_ratio:
                reward = self._run_episode(agent_a, env)
                self.variant_a_results.append(reward)
            else:
                reward = self._run_episode(agent_b, env)
                self.variant_b_results.append(reward)
        
        return self.analyze_results()
    
    def _run_episode(self, agent, env):
        """Run single episode"""
        state = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        return total_reward
    
    def analyze_results(self):
        """Statistical analysis of A/B test"""
        # T-test
        t_stat, p_value = stats.ttest_ind(
            self.variant_a_results,
            self.variant_b_results
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(self.variant_a_results)**2 + np.std(self.variant_b_results)**2) / 2
        )
        cohens_d = (np.mean(self.variant_a_results) - np.mean(self.variant_b_results)) / pooled_std
        
        return {
            'variant_a_mean': np.mean(self.variant_a_results),
            'variant_b_mean': np.mean(self.variant_b_results),
            'improvement': (np.mean(self.variant_b_results) - np.mean(self.variant_a_results)) / np.mean(self.variant_a_results),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohens_d
        }

# Usage
tester = ABTester()
results = tester.run_experiment(baseline_agent, new_agent, env, num_episodes=1000)

if results['significant'] and results['improvement'] > 0.05:
    print("Deploy new agent!")
else:
    print("Keep baseline agent")
```

## ⚡ Performance Profiling

### Profiling Agent Performance

```python
import cProfile
import pstats
from line_profiler import LineProfiler

class AgentProfiler:
    """Profile agent performance"""
    
    def profile_inference(self, agent, state, num_iterations=10000):
        """Profile inference time"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(num_iterations):
            action = agent.select_action(state)
        
        profiler.disable()
        
        # Print stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
    
    def profile_training(self, agent, env, episodes=100):
        """Profile training loop"""
        profiler = LineProfiler()
        profiler.add_function(agent.update)
        profiler.add_function(agent.select_action)
        
        profiler.enable()
        
        for _ in range(episodes):
            state = env.reset()[0]
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
        
        profiler.disable()
        profiler.print_stats()
    
    def memory_profile(self, agent, env, episodes=100):
        """Profile memory usage"""
        from memory_profiler import profile
        
        @profile
        def run_training():
            for _ in range(episodes):
                state = env.reset()[0]
                done = False
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
        
        run_training()
```

## 🛡️ Safety Testing

### Safety Constraint Verification

```python
class SafetyTester:
    """Test agent safety properties"""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.safety_violations = []
    
    def test_state_constraints(self, constraints: dict):
        """Verify state constraints"""
        state = self.env.reset()[0]
        done = False
        violations = 0
        
        while not done:
            # Check constraints
            for constraint_name, constraint_fn in constraints.items():
                if not constraint_fn(state):
                    violations += 1
                    self.safety_violations.append({
                        'constraint': constraint_name,
                        'state': state.copy()
                    })
            
            action = self.agent.select_action(state)
            state, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        
        return {
            'total_violations': violations,
            'violations_per_episode': violations,
            'safe': violations == 0
        }
    
    def test_action_bounds(self, episodes=100):
        """Verify actions are within bounds"""
        violations = 0
        
        for _ in range(episodes):
            state = self.env.reset()[0]
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                
                # Check bounds
                action_space = self.env.action_space
                if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                    if np.any(action < action_space.low) or np.any(action > action_space.high):
                        violations += 1
                
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
        
        return {'violations': violations, 'safe': violations == 0}
```

## 🤖 Automated Testing Pipelines

### CI/CD for Agents

```yaml
# .github/workflows/agent-tests.yml
name: Agent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: pytest tests/unit/ --cov=agent --cov-report=xml
    
    - name: Run integration tests
      run: pytest tests/integration/
    
    - name: Run performance benchmarks
      run: python benchmarks/run_benchmarks.py
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Continuous Evaluation

```python
class ContinuousEvaluator:
    """Continuously evaluate agent in production"""
    
    def __init__(self, agent, eval_env, metrics_logger):
        self.agent = agent
        self.eval_env = eval_env
        self.metrics_logger = metrics_logger
    
    def evaluate_periodically(self, interval_hours=24):
        """Run evaluation at regular intervals"""
        import schedule
        import time
        
        def job():
            metrics = self._run_evaluation()
            self.metrics_logger.log(metrics)
            
            # Alert if performance degrades
            if metrics['success_rate'] < 0.8:
                self._send_alert("Agent performance below threshold")
        
        schedule.every(interval_hours).hours.do(job)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def _run_evaluation(self):
        """Run evaluation suite"""
        rewards = []
        
        for _ in range(100):
            episode_reward = self._run_episode()
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean([r > 100 for r in rewards])
        }
```

## 📚 Resources

### Testing Frameworks
- **[pytest](https://pytest.org/)** - Python testing framework
- **[Hypothesis](https://hypothesis.readthedocs.io/)** - Property-based testing
- **[unittest](https://docs.python.org/3/library/unittest.html)** - Built-in testing

### Benchmarking Tools
- **[pytest-benchmark](https://pytest-benchmark.readthedocs.io/)** - Performance testing
- **[locust](https://locust.io/)** - Load testing
- **[Ray](https://docs.ray.io/)** - Distributed testing

## 🔗 Related Topics

- [Metrics & Methods](./Metrics-Methods.md)
- [Benchmarks & Datasets](./Benchmarks-Datasets.md)
- [Programming Fundamentals](../Supporting-Skills/Programming-Fundamentals.md)
- [System Design](../Supporting-Skills/System-Design.md)

---

*This guide covers testing frameworks for agent systems. For evaluation metrics, see Metrics & Methods.*