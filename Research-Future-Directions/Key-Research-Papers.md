# 📄 Key Research Papers

## 📋 Overview

This curated collection covers foundational and cutting-edge research in Agentic AI, from classic multi-agent systems to modern LLM-based agents.

## 🏛️ Foundational Papers

### Multi-Agent Systems

**1. "Multiagent Systems: A Modern Approach"** (Wooldridge, 2009)
- **Key Contributions**: Comprehensive framework for MAS theory
- **Topics**: Agent architectures, communication, cooperation, negotiation
- **Impact**: Standard textbook for multi-agent systems

**2. "Contract Net Protocol"** (Smith, 1980)
- **Key Contributions**: Task allocation through bidding
- **Topics**: Distributed problem solving, negotiation protocols
- **Impact**: Foundation for agent coordination
- **Citation**: 6000+

**3. "BDI Model"** (Rao & Georgeff, 1995)
- **Key Contributions**: Beliefs-Desires-Intentions architecture
- **Topics**: Practical reasoning, agent plans
- **Impact**: Widely used agent architecture
- **Code Example**:
```python
class BDIAgent:
    def __init__(self):
        self.beliefs = {}      # What agent knows
        self.desires = []      # Goals agent wants to achieve
        self.intentions = []   # Committed plans
    
    def deliberate(self):
        """Select desires to pursue"""
        self.intentions = self.option_generation(self.beliefs, self.desires)
    
    def act(self):
        """Execute intentions"""
        for intention in self.intentions:
            self.execute(intention)
```

### Reinforcement Learning Milestones

**4. "Q-Learning"** (Watkins, 1989)
- **Key Contributions**: Model-free RL algorithm
- **Equation**: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
- **Impact**: Foundation for modern deep RL
- **Citation**: 20,000+

**5. "Playing Atari with Deep Reinforcement Learning"** (Mnih et al., 2013)
- **Key Contributions**: Deep Q-Networks (DQN)
- **Innovation**: CNNs for value function approximation
- **Results**: Human-level performance on Atari games
- **Citation**: 15,000+
- **[arXiv](https://arxiv.org/abs/1312.5602)**

**6. "Trust Region Policy Optimization (TRPO)"** (Schulman et al., 2015)
- **Key Contributions**: Stable policy gradient method
- **Innovation**: Monotonic improvement guarantee
- **Impact**: Reliable for continuous control
- **Citation**: 6,000+
- **[arXiv](https://arxiv.org/abs/1502.05477)**

**7. "Proximal Policy Optimization (PPO)"** (Schulman et al., 2017)
- **Key Contributions**: Simpler alternative to TRPO
- **Innovation**: Clipped surrogate objective
- **Impact**: De facto standard for on-policy RL
- **Citation**: 8,000+
- **[arXiv](https://arxiv.org/abs/1707.06347)**

**8. "Soft Actor-Critic (SAC)"** (Haarnoja et al., 2018)
- **Key Contributions**: Off-policy actor-critic with entropy regularization
- **Innovation**: Maximum entropy RL framework
- **Results**: State-of-the-art on continuous control
- **Citation**: 3,000+
- **[arXiv](https://arxiv.org/abs/1801.01290)**

### Planning & Reasoning

**9. "STRIPS Planning"** (Fikes & Nilsson, 1971)
- **Key Contributions**: Classical planning formalism
- **Topics**: State space search, goal-based planning
- **Impact**: Foundation for automated planning
- **Citation**: 6,000+

**10. "Monte Carlo Tree Search (MCTS)"** (Coulom, 2006)
- **Key Contributions**: Best-first search with random sampling
- **Innovation**: Balance exploration-exploitation
- **Impact**: Core algorithm in AlphaGo
- **Applications**: Game playing, planning

**11. "AlphaGo"** (Silver et al., 2016)
- **Key Contributions**: Defeated world champion at Go
- **Innovation**: MCTS + deep neural networks
- **Impact**: Landmark achievement in AI
- **Citation**: 8,000+
- **[Nature](https://www.nature.com/articles/nature16961)**

**12. "AlphaZero"** (Silver et al., 2017)
- **Key Contributions**: General game player (Chess, Go, Shogi)
- **Innovation**: Pure self-play learning
- **Results**: Superhuman performance without human data
- **Citation**: 3,000+
- **[Nature](https://www.nature.com/articles/nature24270)**

## 🚀 Recent Advances (2020-2024)

### LLM-Based Agents

**13. "ReAct: Reasoning and Acting"** (Yao et al., 2022)
- **Key Contributions**: Interleaving reasoning and actions
- **Innovation**: Thought-action-observation loop
- **Results**: Improved task completion on complex tasks
- **[arXiv](https://arxiv.org/abs/2210.03629)**
```python
# ReAct pattern
def react_agent(task):
    while not task_complete:
        thought = llm(f"Thought: {task}")
        action = llm(f"Action: {thought}")
        observation = execute(action)
        task = update_task(observation)
```

**14. "Toolformer"** (Schick et al., 2023)
- **Key Contributions**: LLMs learn to use tools
- **Innovation**: Self-supervised tool use learning
- **Impact**: Extends LLM capabilities beyond text
- **[arXiv](https://arxiv.org/abs/2302.04761)**

**15. "AutoGPT"** (Significant Gravitas, 2023)
- **Key Contributions**: Autonomous GPT-4 agent
- **Innovation**: Goal decomposition, memory, self-critique
- **Impact**: Popularized autonomous agents
- **[GitHub](https://github.com/Significant-Gravitas/AutoGPT)**

**16. "Generative Agents: Interactive Simulacra"** (Park et al., 2023)
- **Key Contributions**: Believable agent simulations
- **Innovation**: Memory stream, reflection, planning
- **Results**: Human-like behavior in virtual town
- **Citation**: 500+
- **[arXiv](https://arxiv.org/abs/2304.03442)**

**17. "AgentBench: Evaluating LLMs as Agents"** (Liu et al., 2023)
- **Key Contributions**: Comprehensive agent benchmark
- **Topics**: 8 diverse environments
- **Results**: Performance comparison across LLMs
- **[arXiv](https://arxiv.org/abs/2308.03688)**

**18. "The Rise and Potential of Large Language Model Based Agents"** (Xi et al., 2023)
- **Key Contributions**: Comprehensive survey of LLM agents
- **Topics**: Architectures, applications, evaluation
- **Impact**: Essential reading for LLM agent researchers
- **[arXiv](https://arxiv.org/abs/2309.07864)**

### Multi-Agent Collaboration

**19. "QMIX: Monotonic Value Function Factorisation"** (Rashid et al., 2018)
- **Key Contributions**: Multi-agent RL with value decomposition
- **Innovation**: Centralized training, decentralized execution
- **Results**: SOTA on StarCraft II
- **Citation**: 1,500+
- **[arXiv](https://arxiv.org/abs/1803.11485)**

**20. "CommNet: Learning Multiagent Communication"** (Sukhbaatar et al., 2016)
- **Key Contributions**: Learned communication protocols
- **Innovation**: Differentiable communication
- **Applications**: Cooperative tasks
- **Citation**: 1,200+
- **[arXiv](https://arxiv.org/abs/1605.07736)**

**21. "Multi-Agent Actor-Critic"** (Lowe et al., 2017)
- **Key Contributions**: Actor-critic for multi-agent
- **Innovation**: Centralized critic, decentralized actors
- **Results**: Improved coordination
- **Citation**: 2,000+
- **[arXiv](https://arxiv.org/abs/1706.02275)**

**22. "MetaGPT: Meta Programming for Multi-Agent Systems"** (Hong et al., 2023)
- **Key Contributions**: Software development with LLM agents
- **Innovation**: Role-based agent collaboration
- **Results**: Generate code, docs, tests
- **[arXiv](https://arxiv.org/abs/2308.00352)**

### Safety & Alignment

**23. "Constitutional AI"** (Bai et al., 2022)
- **Key Contributions**: Training AI with principles
- **Innovation**: Self-critique and revision
- **Impact**: Safer, more aligned AI systems
- **Citation**: 500+
- **[arXiv](https://arxiv.org/abs/2212.08073)**

**24. "AI Safety via Debate"** (Irving et al., 2018)
- **Key Contributions**: Agents debate to find truth
- **Innovation**: Adversarial oversight
- **Applications**: Scalable alignment
- **Citation**: 400+
- **[arXiv](https://arxiv.org/abs/1805.00899)**

**25. "Concrete Problems in AI Safety"** (Amodei et al., 2016)
- **Key Contributions**: Taxonomy of safety problems
- **Topics**: Robustness, reward hacking, scalable oversight
- **Impact**: Defined research agenda
- **Citation**: 2,000+
- **[arXiv](https://arxiv.org/abs/1606.06565)**

### Emergent Capabilities

**26. "Chain-of-Thought Prompting"** (Wei et al., 2022)
- **Key Contributions**: Elicit reasoning in LLMs
- **Innovation**: Intermediate reasoning steps
- **Results**: Dramatic improvement on reasoning tasks
- **Citation**: 2,000+
- **[arXiv](https://arxiv.org/abs/2201.11903)**

**27. "Tree of Thoughts"** (Yao et al., 2023)
- **Key Contributions**: Deliberate problem solving
- **Innovation**: Explore multiple reasoning paths
- **Results**: Better on complex tasks
- **[arXiv](https://arxiv.org/abs/2305.10601)**

**28. "Let's Verify Step by Step"** (Lightman et al., 2023)
- **Key Contributions**: Process vs outcome supervision
- **Innovation**: Verify reasoning process
- **Results**: More reliable reasoning
- **[arXiv](https://arxiv.org/abs/2305.20050)**

## 📊 Paper Reading Guide

### How to Read Research Papers

```python
class PaperReadingStrategy:
    """Systematic approach to reading papers"""
    
    def three_pass_method(self, paper):
        """Three-pass reading approach"""
        
        # Pass 1: Bird's eye view (5-10 minutes)
        abstract = self.read_abstract(paper)
        intro_conclusion = self.skim_intro_conclusion(paper)
        figures = self.review_figures(paper)
        
        decision_1 = self.decide_continue()
        if not decision_1:
            return "Not relevant"
        
        # Pass 2: Grasp content (1 hour)
        self.read_carefully(paper, skip_proofs=True)
        self.note_key_points(paper)
        self.identify_unread_references(paper)
        
        decision_2 = self.decide_deep_dive()
        if not decision_2:
            return "Understood main contributions"
        
        # Pass 3: Virtual re-implementation (4-5 hours)
        self.reconstruct_paper(paper)
        self.identify_assumptions(paper)
        self.critique_techniques(paper)
        self.propose_improvements(paper)
        
        return "Deep understanding achieved"
```

### Critical Reading Checklist

- [ ] What problem does the paper solve?
- [ ] What are the key contributions?
- [ ] What methods/algorithms are proposed?
- [ ] What experiments validate the claims?
- [ ] What are the limitations?
- [ ] How does it compare to prior work?
- [ ] What are potential applications?
- [ ] What future work is suggested?

## 📚 Conference & Venue Guide

### Top Conferences

| Conference | Focus | Deadline | Acceptance Rate |
|-----------|-------|----------|-----------------|
| **NeurIPS** | Machine Learning | May | ~20% |
| **ICML** | Machine Learning | Jan | ~22% |
| **ICLR** | Deep Learning | Sep | ~25% |
| **AAMAS** | Multi-Agent Systems | Nov | ~23% |
| **AAAI** | AI (Broad) | Aug | ~20% |
| **IJCAI** | AI (Broad) | Jan | ~15% |
| **CVPR** | Computer Vision | Nov | ~25% |
| **ACL** | NLP | Jan | ~22% |

### arXiv Categories

- **cs.AI**: Artificial Intelligence
- **cs.MA**: Multiagent Systems
- **cs.LG**: Machine Learning
- **cs.CL**: Computation and Language
- **cs.RO**: Robotics

## 🔍 Research Tools

### Paper Discovery

```python
# Semantic Scholar API
import requests

def search_papers(query, limit=10):
    """Search for papers"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query,
        'limit': limit,
        'fields': 'title,authors,year,citationCount,abstract,url'
    }
    
    response = requests.get(url, params=params)
    return response.json()

# Usage
papers = search_papers("multi-agent reinforcement learning")
for paper in papers['data']:
    print(f"{paper['title']} ({paper['year']}) - {paper['citationCount']} citations")
```

### Tools
- **[Connected Papers](https://www.connectedpapers.com/)** - Visual paper exploration
- **[Semantic Scholar](https://www.semanticscholar.org/)** - AI-powered search
- **[Papers With Code](https://paperswithcode.com/)** - Papers + implementations
- **[arXiv Sanity](http://www.arxiv-sanity.com/)** - arXiv paper recommender

## 📖 Reading Lists

### Beginner Track
1. Sutton & Barto - "Reinforcement Learning"
2. Russell & Norvig - "AI: A Modern Approach"
3. DQN Paper (Mnih et al., 2013)
4. PPO Paper (Schulman et al., 2017)

### Intermediate Track
1. TRPO, SAC papers
2. AlphaGo, AlphaZero papers
3. ReAct, Toolformer papers
4. Multi-agent RL surveys

### Advanced Track
1. Constitutional AI
2. AgentBench evaluation
3. Recent NeurIPS/ICML agent papers
4. Domain-specific applications

## 🔗 Related Topics

- [Emerging Trends](./Emerging-Trends.md)
- [Open Challenges](./Open-Challenges.md)
- [Orchestration Frameworks](../Frameworks-Tools/Orchestration-Frameworks.md)
- [Benchmarks & Datasets](../Agent-Evaluation-Benchmarking/Benchmarks-Datasets.md)

---

*This guide covers key research papers in Agentic AI. For current developments, see Emerging Trends.*