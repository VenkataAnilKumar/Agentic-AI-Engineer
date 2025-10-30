# 🤔 Critical Thinking

## 📋 Overview

Critical thinking and problem-solving skills are essential for building effective AI agents. This guide covers systematic approaches to analyzing problems, debugging systems, and making informed design decisions.

## 🔍 Systems Thinking

### Understanding Complex Systems

```python
class SystemAnalyzer:
    """Analyze agent system behavior"""
    
    def __init__(self):
        self.components = {}
        self.interactions = []
        self.feedback_loops = []
    
    def add_component(self, name: str, properties: dict):
        """Add system component"""
        self.components[name] = properties
    
    def add_interaction(self, source: str, target: str, relationship: str):
        """Add component interaction"""
        self.interactions.append({
            'source': source,
            'target': target,
            'relationship': relationship
        })
    
    def identify_feedback_loops(self) -> list:
        """Identify feedback loops in system"""
        # Graph-based analysis
        graph = self._build_graph()
        cycles = self._find_cycles(graph)
        return cycles
    
    def analyze_emergent_behavior(self):
        """Analyze system-level behavior"""
        return {
            'complexity': len(self.components) * len(self.interactions),
            'coupling': self._calculate_coupling(),
            'feedback_loops': len(self.feedback_loops)
        }

# Usage
analyzer = SystemAnalyzer()
analyzer.add_component('perception', {'latency': 10, 'accuracy': 0.95})
analyzer.add_component('planning', {'complexity': 'high', 'timeout': 1000})
analyzer.add_interaction('perception', 'planning', 'feeds_data')
```

### Root Cause Analysis

```
Five Whys Technique:

Problem: Agent performance degraded
├─ Why? → Response time increased
│  └─ Why? → Database queries are slow
│     └─ Why? → Missing index on frequently queried column
│        └─ Why? → Index was dropped during migration
│           └─ Why? → Migration script didn't preserve indices
│              └─ Solution: Update migration process + add index

Fishbone Diagram:

                    Agent Failure
                         │
        ┌────────────────┼────────────────┐
    Methods          Machines        Environment
        │                │                │
  ┌─────┴─────┐    ┌────┴────┐     ┌────┴────┐
  │ Algorithm │    │ Hardware│     │ Network │
  │  Faulty   │    │ Resource│     │ Latency │
  └───────────┘    │ Shortage│     └─────────┘
                   └─────────┘
```

## 🧩 Problem Decomposition

### Breaking Down Complex Problems

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Subproblem:
    name: str
    description: str
    dependencies: List[str]
    complexity: str
    estimated_effort: int

class ProblemDecomposer:
    """Decompose complex problems"""
    
    def decompose(self, problem: str) -> List[Subproblem]:
        """Break problem into subproblems"""
        subproblems = []
        
        # Example: Decomposing "Build autonomous vehicle agent"
        if "autonomous vehicle" in problem:
            subproblems = [
                Subproblem(
                    name="Perception",
                    description="Process sensor data (camera, lidar, radar)",
                    dependencies=[],
                    complexity="high",
                    estimated_effort=40
                ),
                Subproblem(
                    name="Localization",
                    description="Determine vehicle position",
                    dependencies=["Perception"],
                    complexity="medium",
                    estimated_effort=20
                ),
                Subproblem(
                    name="Path Planning",
                    description="Plan route from A to B",
                    dependencies=["Localization"],
                    complexity="high",
                    estimated_effort=30
                ),
                Subproblem(
                    name="Control",
                    description="Execute planned actions",
                    dependencies=["Path Planning"],
                    complexity="medium",
                    estimated_effort=25
                )
            ]
        
        return self._sort_by_dependencies(subproblems)
    
    def _sort_by_dependencies(self, subproblems: List[Subproblem]) -> List[Subproblem]:
        """Topological sort by dependencies"""
        # Implement topological sort
        sorted_problems = []
        completed = set()
        
        while len(sorted_problems) < len(subproblems):
            for sp in subproblems:
                if sp.name not in completed:
                    if all(dep in completed for dep in sp.dependencies):
                        sorted_problems.append(sp)
                        completed.add(sp.name)
        
        return sorted_problems
```

## 🐛 Debugging Strategies

### Systematic Debugging

```python
class Debugger:
    """Systematic debugging framework"""
    
    def __init__(self):
        self.hypotheses = []
        self.experiments = []
        self.findings = []
    
    def scientific_debugging(self, bug_description: str):
        """
        Scientific method for debugging:
        1. Observe
        2. Hypothesize
        3. Experiment
        4. Analyze
        5. Conclude
        """
        
        # 1. Observation
        observations = self.gather_evidence(bug_description)
        
        # 2. Form hypotheses
        self.hypotheses = self.generate_hypotheses(observations)
        
        # 3. Design experiments
        for hypothesis in self.hypotheses:
            experiment = self.design_experiment(hypothesis)
            
            # 4. Run experiment
            result = self.run_experiment(experiment)
            
            # 5. Analyze results
            if self.validates_hypothesis(result, hypothesis):
                return self.create_fix(hypothesis)
        
        return None
    
    def binary_search_debugging(self, code_history: list):
        """Use binary search to find bug introduction"""
        left, right = 0, len(code_history) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            if self.has_bug(code_history[mid]):
                right = mid
            else:
                left = mid + 1
        
        return code_history[left]  # First buggy version

# Debugging checklist
debug_checklist = """
□ Can you reproduce the bug consistently?
□ What changed recently?
□ What are the input conditions?
□ What is the expected vs actual behavior?
□ Are there error messages/logs?
□ Does it work in isolation?
□ Are there resource constraints (memory, CPU)?
□ Are there timing/concurrency issues?
□ Have you checked assumptions?
□ Did you use a debugger/profiler?
"""
```

### Logging Best Practices

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured logging for debugging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if hasattr(record, 'extra_data'):
                log_data.update(record.extra_data)
            
            return json.dumps(log_data)
    
    def log_agent_decision(self, agent_id: str, state: dict, action: str, reasoning: str):
        """Log agent decision with context"""
        self.logger.info(
            "Agent decision",
            extra={
                'extra_data': {
                    'agent_id': agent_id,
                    'state': state,
                    'action': action,
                    'reasoning': reasoning
                }
            }
        )

# Usage
logger = StructuredLogger('agent_system')
logger.log_agent_decision(
    agent_id='agent_1',
    state={'position': [0, 0], 'goal': [10, 10]},
    action='move_north',
    reasoning='Shortest path to goal'
)
```

## 📚 Research Methodology

### Literature Review Process

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    year: int
    venue: str
    key_contributions: List[str]
    limitations: List[str]
    citations: int

class LiteratureReview:
    """Systematic literature review"""
    
    def __init__(self):
        self.papers = []
        self.themes = {}
    
    def search_papers(self, query: str, databases: List[str]):
        """Search academic databases"""
        # Search Google Scholar, ArXiv, ACM, IEEE
        results = []
        for db in databases:
            results.extend(self._query_database(db, query))
        return results
    
    def screen_papers(self, papers: List[ResearchPaper], criteria: dict):
        """Screen papers by inclusion criteria"""
        filtered = []
        
        for paper in papers:
            if self._meets_criteria(paper, criteria):
                filtered.append(paper)
        
        return filtered
    
    def extract_insights(self, papers: List[ResearchPaper]):
        """Extract key insights"""
        insights = {
            'common_approaches': [],
            'gaps': [],
            'trends': [],
            'best_practices': []
        }
        
        # Analyze papers
        for paper in papers:
            # Extract patterns
            pass
        
        return insights

# Research workflow
workflow = """
1. Define Research Question
   └─ What problem are we solving?

2. Search Literature
   ├─ Google Scholar
   ├─ ArXiv
   ├─ IEEE Xplore
   └─ ACM Digital Library

3. Screen Papers
   ├─ Read abstracts
   ├─ Apply inclusion criteria
   └─ Assess quality

4. Extract Data
   ├─ Methods
   ├─ Results
   └─ Limitations

5. Synthesize Findings
   └─ Identify patterns, gaps, opportunities

6. Apply to Project
   └─ Inform design decisions
"""
```

## 🎨 Design Thinking

### User-Centered Agent Design

```python
class DesignThinkingProcess:
    """Apply design thinking to agent development"""
    
    def empathize(self, user_research: dict):
        """Understand user needs"""
        return {
            'pain_points': self._identify_pain_points(user_research),
            'goals': self._extract_goals(user_research),
            'context': self._analyze_context(user_research)
        }
    
    def define(self, insights: dict):
        """Define problem statement"""
        return f"""
        User Persona: {insights['persona']}
        
        Needs: {insights['needs']}
        
        Problem: {insights['pain_points']}
        
        Insight: {insights['key_insight']}
        """
    
    def ideate(self, problem_statement: str):
        """Generate solution ideas"""
        ideas = [
            "Conversational agent with natural language",
            "Proactive agent that anticipates needs",
            "Multi-agent system with specialized roles",
            "Explainable AI for transparency"
        ]
        return ideas
    
    def prototype(self, selected_idea: str):
        """Build low-fidelity prototype"""
        # Quick implementation
        pass
    
    def test(self, prototype, users: list):
        """Test with users"""
        feedback = []
        
        for user in users:
            result = user.interact_with(prototype)
            feedback.append({
                'user': user.id,
                'satisfaction': result.rating,
                'comments': result.feedback
            })
        
        return self.analyze_feedback(feedback)
```

## ⚖️ Decision Frameworks

### Multi-Criteria Decision Making

```python
import numpy as np

class DecisionMatrix:
    """Evaluate options with multiple criteria"""
    
    def __init__(self, options: List[str], criteria: List[str]):
        self.options = options
        self.criteria = criteria
        self.scores = np.zeros((len(options), len(criteria)))
        self.weights = np.ones(len(criteria)) / len(criteria)
    
    def set_score(self, option: str, criterion: str, score: float):
        """Set score for option-criterion pair"""
        i = self.options.index(option)
        j = self.criteria.index(criterion)
        self.scores[i, j] = score
    
    def set_weights(self, weights: dict):
        """Set criteria weights"""
        for criterion, weight in weights.items():
            j = self.criteria.index(criterion)
            self.weights[j] = weight
        
        # Normalize weights
        self.weights /= self.weights.sum()
    
    def calculate_scores(self) -> dict:
        """Calculate weighted scores"""
        weighted_scores = self.scores @ self.weights
        
        results = {
            option: score
            for option, score in zip(self.options, weighted_scores)
        }
        
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# Example: Choosing RL algorithm
decision = DecisionMatrix(
    options=['DQN', 'PPO', 'SAC', 'A3C'],
    criteria=['Sample Efficiency', 'Stability', 'Ease of Use', 'Performance']
)

# Set scores (0-10)
decision.set_score('DQN', 'Sample Efficiency', 6)
decision.set_score('DQN', 'Stability', 7)
decision.set_score('PPO', 'Stability', 9)
decision.set_score('PPO', 'Ease of Use', 8)

# Set weights
decision.set_weights({
    'Sample Efficiency': 0.3,
    'Stability': 0.4,
    'Ease of Use': 0.2,
    'Performance': 0.1
})

# Get rankings
rankings = decision.calculate_scores()
print(f"Best option: {list(rankings.keys())[0]}")
```

## 📚 Resources

### Books
- **"Thinking, Fast and Slow"** by Daniel Kahneman
- **"The Design of Everyday Things"** by Don Norman
- **"How to Solve It"** by George Pólya
- **"The Art of Problem Solving"** by Russell L. Ackoff

### Tools
- **Miro** - Visual collaboration
- **Notion** - Knowledge management
- **Obsidian** - Connected thinking
- **Draw.io** - Diagramming

## 🔗 Related Topics

- [System Design](./System-Design.md)
- [Programming Fundamentals](./Programming-Fundamentals.md)
- [Research & Future Directions](../Research-Future-Directions/)
- [Evaluation Frameworks](../Agent-Evaluation-Benchmarking/Evaluation-Frameworks.md)

---

*This guide covers critical thinking for AI engineering. For technical implementation, see related topics.*