# 🌐 Real-World Applications

## 📋 Overview

Agent systems are transforming industries by automating complex decision-making, optimizing operations, and providing intelligent assistance. This guide explores production deployments across key sectors with architectures, implementations, and lessons learned.

## 🏥 Healthcare

### Diagnostic Agent System

**Use Case**: AI-powered diagnostic assistance for radiologists

```python
from typing import List, Dict
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

class MedicalDiagnosticAgent:
    """Agent for medical diagnosis assistance"""
    
    def __init__(self, model_path: str, knowledge_base_path: str):
        # Vision model for imaging
        self.vision_model = torch.load(f"{model_path}/vision_model.pth")
        
        # Language model for reports
        self.llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
        self.llm_model = AutoModel.from_pretrained("microsoft/BioGPT")
        
        # Medical knowledge base
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        self.diagnosis_history = []
    
    def analyze_imaging(self, image: np.ndarray, modality: str) -> Dict:
        """Analyze medical imaging"""
        # Preprocess image
        image_tensor = self.preprocess_image(image, modality)
        
        # Run vision model
        with torch.no_grad():
            features = self.vision_model(image_tensor)
            predictions = torch.softmax(features, dim=-1)
        
        # Get top findings
        top_k = 5
        values, indices = torch.topk(predictions, top_k)
        
        findings = []
        for prob, idx in zip(values[0], indices[0]):
            condition = self.get_condition_name(idx.item())
            findings.append({
                'condition': condition,
                'confidence': float(prob),
                'severity': self.assess_severity(condition)
            })
        
        return {
            'modality': modality,
            'findings': findings,
            'requires_review': any(f['severity'] == 'high' for f in findings)
        }
    
    def generate_report(self, findings: List[Dict], patient_history: Dict) -> str:
        """Generate diagnostic report"""
        # Construct prompt with findings
        prompt = self.construct_diagnostic_prompt(findings, patient_history)
        
        # Generate report using LLM
        inputs = self.llm_tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=500,
                temperature=0.3,
                do_sample=True
            )
        
        report = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return report
    
    def recommend_tests(self, findings: List[Dict]) -> List[str]:
        """Recommend additional tests"""
        recommendations = []
        
        for finding in findings:
            if finding['confidence'] < 0.7:
                # Query knowledge base for confirmatory tests
                tests = self.knowledge_base.get_confirmatory_tests(
                    finding['condition']
                )
                recommendations.extend(tests)
        
        return list(set(recommendations))
    
    def assess_urgency(self, findings: List[Dict]) -> str:
        """Assess urgency level"""
        high_severity = sum(1 for f in findings if f['severity'] == 'high')
        
        if high_severity > 0:
            return "urgent"
        elif any(f['confidence'] > 0.8 for f in findings):
            return "routine_priority"
        else:
            return "routine"

# Example usage
agent = MedicalDiagnosticAgent(
    model_path="/models/medical",
    knowledge_base_path="/data/medical_kb"
)

# Analyze chest X-ray
xray_image = load_image("chest_xray.dcm")
results = agent.analyze_imaging(xray_image, modality="X-ray")

# Generate report
report = agent.generate_report(
    results['findings'],
    patient_history={'age': 65, 'smoking': True}
)

print(f"Urgency: {agent.assess_urgency(results['findings'])}")
print(f"Report: {report}")
```

**Architecture**:
```
┌─────────────────┐
│  Medical Images │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Vision Model           │
│  (ResNet + Attention)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Diagnostic Agent       │
│  - Findings extraction  │
│  - Severity assessment  │
│  - Report generation    │
└────────┬────────────────┘
         │
         ├──► Knowledge Base
         ├──► Patient History
         │
         ▼
┌─────────────────────────┐
│  Output                 │
│  - Diagnostic report    │
│  - Recommendations      │
│  - Urgency level        │
└─────────────────────────┘
```

### Drug Discovery Agent

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch.nn as nn

class DrugDiscoveryAgent:
    """Agent for molecular property prediction"""
    
    def __init__(self, target_protein: str):
        self.target_protein = target_protein
        self.property_model = self.load_property_model()
        self.docking_engine = DockingEngine(target_protein)
        
    def screen_molecules(self, smiles_list: List[str]) -> List[Dict]:
        """Screen molecules for drug-like properties"""
        candidates = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate properties
            properties = self.calculate_properties(mol)
            
            # Check Lipinski's Rule of Five
            if self.passes_lipinski(properties):
                # Predict binding affinity
                binding_score = self.predict_binding(mol)
                
                # Predict toxicity
                toxicity_score = self.predict_toxicity(mol)
                
                if binding_score > 0.7 and toxicity_score < 0.3:
                    candidates.append({
                        'smiles': smiles,
                        'binding_affinity': binding_score,
                        'toxicity': toxicity_score,
                        'properties': properties
                    })
        
        # Rank by score
        candidates.sort(
            key=lambda x: x['binding_affinity'] - x['toxicity'],
            reverse=True
        )
        
        return candidates
    
    def calculate_properties(self, mol) -> Dict:
        """Calculate molecular properties"""
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol)
        }
    
    def passes_lipinski(self, properties: Dict) -> bool:
        """Check Lipinski's Rule of Five"""
        return (
            properties['molecular_weight'] <= 500 and
            properties['logp'] <= 5 and
            properties['hbd'] <= 5 and
            properties['hba'] <= 10
        )
```

## 💰 Finance

### Algorithmic Trading Agent

```python
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Trade:
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: pd.Timestamp

class TradingAgent:
    """Reinforcement learning-based trading agent"""
    
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.portfolio = {}
        self.trade_history = []
        
        # RL components
        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.replay_buffer = []
        
    def get_state(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract state features from market data"""
        features = []
        
        # Technical indicators
        features.append(self.calculate_rsi(market_data))
        features.append(self.calculate_macd(market_data))
        features.append(self.calculate_bollinger_bands(market_data))
        
        # Market sentiment
        features.append(self.get_sentiment_score(market_data))
        
        # Portfolio state
        features.append(self.capital / 100000)  # Normalized
        features.append(len(self.portfolio))
        
        return np.array(features)
    
    def decide_action(self, state: np.ndarray, symbol: str) -> str:
        """Decide trading action"""
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(['buy', 'sell', 'hold'])
        
        # Get Q-values
        q_values = self.q_network.predict(state.reshape(1, -1))[0]
        action_idx = np.argmax(q_values)
        
        actions = ['buy', 'sell', 'hold']
        return actions[action_idx]
    
    def execute_trade(self, symbol: str, action: str, 
                      current_price: float, quantity: int) -> bool:
        """Execute trade with risk management"""
        # Risk checks
        if action == 'buy':
            cost = current_price * quantity
            if cost > self.capital * 0.1:  # Max 10% per position
                return False
            
            if cost > self.capital:
                return False
            
            # Execute buy
            self.capital -= cost
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
            
        elif action == 'sell':
            if symbol not in self.portfolio or self.portfolio[symbol] < quantity:
                return False
            
            # Execute sell
            revenue = current_price * quantity
            self.capital += revenue
            self.portfolio[symbol] -= quantity
        
        # Log trade
        self.trade_history.append(Trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=current_price,
            timestamp=pd.Timestamp.now()
        ))
        
        return True
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def risk_management(self) -> Dict:
        """Implement risk management rules"""
        total_value = self.capital
        
        # Calculate portfolio value
        for symbol, quantity in self.portfolio.items():
            current_price = self.get_current_price(symbol)
            total_value += current_price * quantity
        
        # Stop loss check
        if total_value < 0.95 * 100000:  # 5% drawdown
            return {'action': 'reduce_positions', 'urgency': 'high'}
        
        return {'action': 'continue', 'urgency': 'normal'}
```

### Fraud Detection Agent

```python
from sklearn.ensemble import IsolationForest
import pandas as pd

class FraudDetectionAgent:
    """Real-time fraud detection system"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.01)
        self.transaction_history = []
        self.user_profiles = {}
        
    def analyze_transaction(self, transaction: Dict) -> Dict:
        """Analyze transaction for fraud"""
        user_id = transaction['user_id']
        
        # Extract features
        features = self.extract_features(transaction, user_id)
        
        # Anomaly score
        anomaly_score = self.anomaly_detector.score_samples([features])[0]
        
        # Behavioral analysis
        behavior_score = self.analyze_behavior(transaction, user_id)
        
        # Network analysis
        network_risk = self.analyze_network(transaction)
        
        # Combined risk score
        risk_score = self.combine_scores(
            anomaly_score, behavior_score, network_risk
        )
        
        # Decision
        if risk_score > 0.8:
            action = 'block'
            reason = self.explain_decision(features)
        elif risk_score > 0.5:
            action = 'review'
            reason = 'Moderate risk detected'
        else:
            action = 'approve'
            reason = 'Normal transaction'
        
        return {
            'action': action,
            'risk_score': risk_score,
            'reason': reason,
            'requires_verification': risk_score > 0.5
        }
    
    def extract_features(self, transaction: Dict, user_id: str) -> List:
        """Extract fraud detection features"""
        profile = self.user_profiles.get(user_id, {})
        
        features = [
            transaction['amount'],
            transaction['amount'] / profile.get('avg_amount', 100),
            abs(transaction['timestamp'].hour - profile.get('typical_hour', 12)),
            self.location_distance(
                transaction['location'],
                profile.get('typical_location')
            ),
            len(self.transaction_history[-100:]),  # Recent activity
            transaction.get('new_merchant', 0),
            transaction.get('international', 0)
        ]
        
        return features
    
    def analyze_behavior(self, transaction: Dict, user_id: str) -> float:
        """Analyze user behavior patterns"""
        profile = self.user_profiles.get(user_id, {})
        
        # Velocity checks
        recent_transactions = [
            t for t in self.transaction_history[-50:]
            if t['user_id'] == user_id
        ]
        
        if len(recent_transactions) > 10 in 3600:  # 10 in 1 hour
            return 0.8
        
        # Amount pattern
        if transaction['amount'] > profile.get('max_amount', 0) * 2:
            return 0.7
        
        return 0.2
```

## 🤖 Robotics

### Warehouse Robot Navigation

```python
import numpy as np
from typing import Tuple, List

class WarehouseRobotAgent:
    """Autonomous warehouse navigation agent"""
    
    def __init__(self, warehouse_map: np.ndarray):
        self.map = warehouse_map
        self.position = (0, 0)
        self.carrying = None
        self.path_planner = AStarPlanner()
        self.obstacle_detector = ObstacleDetector()
        
    def plan_pick_and_place(self, item_location: Tuple, 
                            drop_location: Tuple) -> List[Tuple]:
        """Plan pick and place mission"""
        # Phase 1: Navigate to item
        path_to_item = self.path_planner.plan(
            self.position, item_location, self.map
        )
        
        # Phase 2: Navigate to drop location
        path_to_drop = self.path_planner.plan(
            item_location, drop_location, self.map
        )
        
        return {
            'pickup_path': path_to_item,
            'delivery_path': path_to_drop,
            'estimated_time': self.estimate_time(path_to_item + path_to_drop)
        }
    
    def navigate(self, goal: Tuple) -> bool:
        """Execute navigation with obstacle avoidance"""
        path = self.path_planner.plan(self.position, goal, self.map)
        
        for waypoint in path:
            # Check for dynamic obstacles
            if self.obstacle_detector.is_blocked(waypoint):
                # Replan
                path = self.path_planner.plan(self.position, goal, self.map)
                continue
            
            # Move to waypoint
            success = self.move_to(waypoint)
            if not success:
                return False
            
            self.position = waypoint
        
        return True
    
    def coordinate_with_fleet(self, fleet_manager) -> Dict:
        """Coordinate with other robots"""
        # Request path from fleet manager
        reserved_path = fleet_manager.reserve_path(
            self.position, self.goal
        )
        
        # Check for conflicts
        conflicts = fleet_manager.check_conflicts(self.id, reserved_path)
        
        if conflicts:
            # Wait or reroute
            alternative = fleet_manager.suggest_alternative(self.id)
            return alternative
        
        return {'path': reserved_path, 'priority': self.priority}
```

## 💬 Customer Service

### Multi-Channel Support Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

class CustomerServiceAgent:
    """Intelligent customer service agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.tools = self.setup_tools()
        self.conversation_history = {}
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def setup_tools(self) -> List[Tool]:
        """Setup agent tools"""
        return [
            Tool(
                name="check_order_status",
                func=self.check_order_status,
                description="Check the status of a customer order"
            ),
            Tool(
                name="process_refund",
                func=self.process_refund,
                description="Process a refund request"
            ),
            Tool(
                name="search_knowledge_base",
                func=self.search_kb,
                description="Search knowledge base for answers"
            ),
            Tool(
                name="escalate_to_human",
                func=self.escalate,
                description="Escalate to human agent"
            )
        ]
    
    def handle_customer_query(self, customer_id: str, 
                              message: str, channel: str) -> Dict:
        """Handle customer query with context"""
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(message)
        
        # Get conversation history
        history = self.conversation_history.get(customer_id, [])
        
        # Determine urgency
        urgency = self.assess_urgency(message, sentiment, history)
        
        # Generate response
        if urgency == 'high' or sentiment['score'] < 0.3:
            # Escalate immediately
            return self.escalate(customer_id, message, reason='high_urgency')
        
        # Use agent to respond
        response = self.agent_executor.invoke({
            'input': message,
            'chat_history': history
        })
        
        # Update history
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': response['output']})
        self.conversation_history[customer_id] = history
        
        return {
            'response': response['output'],
            'sentiment': sentiment,
            'resolved': self.check_if_resolved(history),
            'follow_up_needed': urgency == 'medium'
        }
    
    def assess_urgency(self, message: str, sentiment: Dict, 
                       history: List) -> str:
        """Assess urgency of customer issue"""
        urgent_keywords = ['urgent', 'immediately', 'asap', 'emergency']
        
        if any(kw in message.lower() for kw in urgent_keywords):
            return 'high'
        
        if sentiment['score'] < 0.4:
            return 'high'
        
        if len(history) > 5:  # Long conversation
            return 'medium'
        
        return 'normal'
```

## 📊 Impact Metrics

### Real-World Results

| Domain | Application | Impact | ROI |
|--------|-------------|--------|-----|
| **Healthcare** | Diagnostic assistance | 15% faster diagnosis | 200% |
| **Finance** | Fraud detection | 80% reduction in fraud | 500% |
| **Robotics** | Warehouse automation | 40% efficiency gain | 300% |
| **Customer Service** | AI chatbots | 70% query automation | 400% |
| **Manufacturing** | Quality control | 95% defect detection | 250% |

## 🛠️ Implementation Tools

### Healthcare
- [Hugging Face Med-PaLM](https://huggingface.co/google/med-palm-2)
- [RDKit](https://www.rdkit.org/) for drug discovery
- [MONAI](https://monai.io/) for medical imaging

### Finance
- [QuantConnect](https://www.quantconnect.com/) for algorithmic trading
- [Apache Kafka](https://kafka.apache.org/) for real-time streaming
- [scikit-learn](https://scikit-learn.org/) for fraud detection

### Robotics
- [ROS](https://www.ros.org/) - Robot Operating System
- [MoveIt](https://moveit.ros.org/) for motion planning
- [Gazebo](http://gazebosim.org/) for simulation

## 📚 Learning Resources

### Case Studies
- **Google DeepMind** - AlphaFold for protein folding
- **Tesla** - Autopilot system
- **Amazon** - Warehouse robotics
- **JPMorgan** - COIN contract intelligence

### Books
- **"AI in Healthcare"** by Adam Bohr
- **"Machine Learning for Asset Managers"** by López de Prado
- **"Probabilistic Robotics"** by Thrun et al.

## 🔗 Related Topics

- [Industry Implementations](./Industry-Implementations.md)
- [Agent Architectures](../Architecture-Design/Agent-Architectures.md)
- [Reinforcement Learning](../Core-Concepts/Reinforcement-Learning.md)
- [Agent Evaluation](../Agent-Evaluation-Benchmarking/Metrics-Methods.md)

---

*This guide showcases real-world agent applications across industries. For enterprise case studies, see Industry Implementations.*