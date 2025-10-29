# 📚 Open Source Libraries

## 📋 Overview

This comprehensive guide covers essential open-source libraries for building intelligent agent systems, from deep learning frameworks to specialized agent toolkits.

## 🔥 Deep Learning Frameworks

### PyTorch

**The most popular framework for research and production**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class AgentNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(AgentNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

# Training loop
model = AgentNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    state = torch.randn(32, 4)  # batch of states
    action_probs = model(state)
    # Training logic here
```

**Key Features:**
- ✅ Dynamic computation graphs
- ✅ Pythonic and intuitive API
- ✅ Excellent for research
- ✅ Strong community support
- ✅ TorchScript for production

**Resources:**
- 📖 [Official Documentation](https://pytorch.org/docs/)
- 💻 [GitHub Repository](https://github.com/pytorch/pytorch)
- 🎓 [PyTorch Tutorials](https://pytorch.org/tutorials/)

### TensorFlow

**Google's production-ready ML framework**

```python
import tensorflow as tf
from tensorflow import keras

# Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(4,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

**Key Features:**
- ✅ Production deployment with TF Serving
- ✅ Mobile/edge with TF Lite
- ✅ Distributed training
- ✅ TensorBoard visualization
- ✅ Keras high-level API

**Resources:**
- 📖 [TensorFlow Guide](https://www.tensorflow.org/guide)
- 💻 [GitHub Repository](https://github.com/tensorflow/tensorflow)

## 🤗 Hugging Face Ecosystem

### Transformers

**State-of-the-art NLP and multimodal models**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "The agent decided to"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1,
    temperature=0.7
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# Use for agent reasoning
class LLMAgent:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def reason(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Popular Models:**
- 🤖 GPT-2/3 - Text generation
- 🤖 BERT - Language understanding
- 🤖 T5 - Text-to-text
- 🤖 CLIP - Vision-language
- 🤖 Llama 2/3 - Open LLMs

**Resources:**
- 📖 [Documentation](https://huggingface.co/docs/transformers/)
- 💻 [GitHub](https://github.com/huggingface/transformers)
- 🤗 [Model Hub](https://huggingface.co/models)

### Datasets

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("squad", split="train")

# Use for training
for example in dataset:
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0]
```

## 🎮 Reinforcement Learning

### Stable Baselines3

**High-quality RL algorithm implementations**

```python
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# Create environment
env = make_vec_env("CartPole-v1", n_envs=4)

# Train PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10
)

model.learn(total_timesteps=100_000)

# Save and load
model.save("ppo_cartpole")
model = PPO.load("ppo_cartpole")

# Evaluate
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

**Algorithms Included:**
- 🎯 PPO (Proximal Policy Optimization)
- 🎯 A2C/A3C (Advantage Actor-Critic)
- 🎯 DQN (Deep Q-Network)
- 🎯 SAC (Soft Actor-Critic)
- 🎯 TD3 (Twin Delayed DDPG)

**Resources:**
- 📖 [Documentation](https://stable-baselines3.readthedocs.io/)
- 💻 [GitHub](https://github.com/DLR-RM/stable-baselines3)

### Ray & RLlib

**Distributed RL at scale**

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init()

# Configure algorithm
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=4)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10
    )
)

# Train
algo = config.build()
for i in range(10):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']}")

# Or use Tune for hyperparameter search
tune.run(
    "PPO",
    config={
        "env": "CartPole-v1",
        "num_gpus": 1,
        "num_workers": 4,
        "lr": tune.grid_search([1e-3, 1e-4, 1e-5])
    }
)
```

**Resources:**
- 📖 [RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- 💻 [GitHub](https://github.com/ray-project/ray)

## 🕸️ Graph Neural Networks

### PyTorch Geometric

**Deep learning on graphs**

```python
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data

# Create graph data
edge_index = torch.tensor([[0, 1, 1, 2],
                          [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# Graph Neural Network
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN(in_channels=1, hidden_channels=16, out_channels=2)
```

**Use Cases:**
- 🔗 Multi-agent communication networks
- 🔗 Knowledge graph reasoning
- 🔗 Social network analysis
- 🔗 Molecular property prediction

**Resources:**
- 📖 [Documentation](https://pytorch-geometric.readthedocs.io/)
- 💻 [GitHub](https://github.com/pyg-team/pytorch_geometric)

### DGL (Deep Graph Library)

```python
import dgl
import torch

# Create graph
g = dgl.graph(([0, 0, 1, 2], [1, 2, 2, 3]))
g.ndata['feat'] = torch.randn(4, 10)

# Apply GNN
import dgl.nn as dglnn

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes)
    
    def forward(self, g, features):
        x = torch.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x
```

**Resources:**
- 📖 [Documentation](https://docs.dgl.ai/)
- 💻 [GitHub](https://github.com/dmlc/dgl)

## 🧠 Knowledge Graphs

### RDFLib

**Working with RDF and SPARQL**

```python
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, FOAF

# Create knowledge graph
g = Graph()

# Define namespace
EX = Namespace("http://example.org/")

# Add triples
agent = URIRef(EX.agent1)
g.add((agent, RDF.type, EX.Agent))
g.add((agent, FOAF.name, Literal("Agent Smith")))
g.add((agent, EX.hasCapability, Literal("planning")))

# Query with SPARQL
query = """
    SELECT ?agent ?name
    WHERE {
        ?agent rdf:type ex:Agent .
        ?agent foaf:name ?name .
    }
"""

for row in g.query(query):
    print(f"Agent: {row.agent}, Name: {row.name}")
```

**Resources:**
- 📖 [Documentation](https://rdflib.readthedocs.io/)
- 💻 [GitHub](https://github.com/RDFLib/rdflib)

## 🤖 Agent-Specific Libraries

### PettingZoo

**Multi-agent RL environments**

```python
from pettingzoo.mpe import simple_spread_v3

# Create multi-agent environment
env = simple_spread_v3.env(N=3, max_cycles=25)
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    
    env.step(action)

env.close()
```

**Resources:**
- 📖 [Documentation](https://pettingzoo.farama.org/)
- 💻 [GitHub](https://github.com/Farama-Foundation/PettingZoo)

### Gymnasium (Gym)

**Standard RL environment interface**

```python
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

**Resources:**
- 📖 [Documentation](https://gymnasium.farama.org/)
- 💻 [GitHub](https://github.com/Farama-Foundation/Gymnasium)

## 📊 Library Comparison Matrix

### Deep Learning

| Library | Best For | Learning Curve | Community | Production |
|---------|----------|----------------|-----------|------------|
| **PyTorch** | Research, flexibility | Medium | Excellent | Good |
| **TensorFlow** | Production, deployment | Medium-High | Excellent | Excellent |
| **JAX** | Performance, research | High | Growing | Good |

### Reinforcement Learning

| Library | Algorithms | Distributed | Documentation | Maturity |
|---------|-----------|-------------|---------------|----------|
| **Stable Baselines3** | 7+ core algorithms | No | Excellent | Mature |
| **RLlib** | 15+ algorithms | Yes | Good | Mature |
| **TorchRL** | Modular components | Yes | Growing | Beta |

### NLP/LLM

| Library | Models | Fine-tuning | Inference | Hub |
|---------|--------|-------------|-----------|-----|
| **Transformers** | 1000+ | Easy | Good | HuggingFace |
| **LangChain** | Integration layer | N/A | Good | Templates |
| **LlamaIndex** | Data framework | N/A | Excellent | Examples |

## 🛠️ Complete Library Reference

### Core ML/DL
- [PyTorch](https://github.com/pytorch/pytorch) - Deep learning framework
- [TensorFlow](https://github.com/tensorflow/tensorflow) - End-to-end ML platform
- [JAX](https://github.com/google/jax) - Autograd and XLA
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Traditional ML

### RL & Decision Making
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Ray RLlib](https://github.com/ray-project/ray) - Scalable RL
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - RL environments
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) - Multi-agent RL

### NLP & LLMs
- [Transformers](https://github.com/huggingface/transformers) - Pre-trained models
- [LangChain](https://github.com/langchain-ai/langchain) - LLM applications
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings

### Graph & Knowledge
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - GNNs
- [DGL](https://github.com/dmlc/dgl) - Graph neural networks
- [NetworkX](https://github.com/networkx/networkx) - Graph algorithms
- [RDFLib](https://github.com/RDFLib/rdflib) - Knowledge graphs

### Robotics & Simulation
- [PyBullet](https://github.com/bulletphysics/bullet3) - Physics simulation
- [MuJoCo](https://github.com/deepmind/mujoco) - Multi-joint dynamics
- [ROS](https://www.ros.org/) - Robot Operating System
- [Habitat](https://github.com/facebookresearch/habitat-sim) - Embodied AI

### Utilities
- [NumPy](https://github.com/numpy/numpy) - Numerical computing
- [Pandas](https://github.com/pandas-dev/pandas) - Data manipulation
- [Matplotlib](https://github.com/matplotlib/matplotlib) - Visualization
- [Weights & Biases](https://github.com/wandb/wandb) - Experiment tracking

## 📚 Learning Resources

### Getting Started Guides
- [PyTorch in 60 Minutes](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [HuggingFace Course](https://huggingface.co/course)
- [Stable Baselines3 Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)

### Books
- **"Deep Learning with PyTorch"** by Stevens et al.
- **"Hands-On Machine Learning"** by Géron
- **"Reinforcement Learning"** by Sutton & Barto

## 🔗 Related Topics

- [Orchestration Frameworks](./Orchestration-Frameworks.md)
- [Simulation Environments](../Architecture-Design/Simulation-Environments.md)
- [Reinforcement Learning](../Core-Concepts/Reinforcement-Learning.md)
- [Free Courses](../Resources/Free-Courses.md)

---

*This comprehensive guide covers essential open-source libraries for building intelligent agent systems. For detailed implementation examples, refer to each library's official documentation.*