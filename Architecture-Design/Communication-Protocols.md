# 📡 Communication Protocols

## 📋 Overview

Communication protocols enable agents to exchange information, coordinate actions, and collaborate effectively. Standardized protocols ensure interoperability between heterogeneous agent systems.

## 🗣️ Agent Communication Languages

### FIPA ACL (Agent Communication Language)

**Foundation for Intelligent Physical Agents**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, List

class Performative(Enum):
    """FIPA ACL performatives"""
    INFORM = "inform"          # Assert information
    REQUEST = "request"        # Request action
    QUERY_IF = "query-if"      # Ask yes/no question
    QUERY_REF = "query-ref"    # Ask for value
    CFP = "cfp"                # Call for proposals
    PROPOSE = "propose"        # Propose solution
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    AGREE = "agree"            # Agree to perform action
    REFUSE = "refuse"          # Refuse to perform
    FAILURE = "failure"        # Action failed
    INFORM_IF = "inform-if"    # Conditional inform
    CONFIRM = "confirm"        # Confirm truth
    DISCONFIRM = "disconfirm"  # Deny truth

@dataclass
class ACLMessage:
    """FIPA ACL message structure"""
    performative: Performative
    sender: str
    receiver: List[str]
    content: Any
    language: str = "python"
    ontology: str = "default"
    protocol: str = "fipa-request"
    conversation_id: str = None
    reply_with: str = None
    in_reply_to: str = None
    reply_by: float = None
    
    def to_string(self):
        """Convert to FIPA ACL string format"""
        msg = f"({self.performative.value}\n"
        msg += f"  :sender {self.sender}\n"
        msg += f"  :receiver {' '.join(self.receiver)}\n"
        msg += f"  :content \"{self.content}\"\n"
        msg += f"  :language {self.language}\n"
        msg += f"  :ontology {self.ontology}\n"
        
        if self.conversation_id:
            msg += f"  :conversation-id {self.conversation_id}\n"
        if self.reply_with:
            msg += f"  :reply-with {self.reply_with}\n"
        if self.in_reply_to:
            msg += f"  :in-reply-to {self.in_reply_to}\n"
            
        msg += ")"
        return msg

# Example usage
message = ACLMessage(
    performative=Performative.REQUEST,
    sender="agent1",
    receiver=["agent2"],
    content="move(robot, location_B)",
    conversation_id="conv123"
)
```

### KQML (Knowledge Query and Manipulation Language)

```python
class KQMLPerformative(Enum):
    """KQML performatives"""
    TELL = "tell"              # Assert knowledge
    ASK_IF = "ask-if"          # Query truth
    ASK_ONE = "ask-one"        # Query one answer
    ASK_ALL = "ask-all"        # Query all answers
    STREAM = "stream"          # Stream answers
    ACHIEVE = "achieve"        # Request goal achievement
    SUBSCRIBE = "subscribe"    # Subscribe to updates
    MONITOR = "monitor"        # Monitor condition

@dataclass
class KQMLMessage:
    """KQML message structure"""
    performative: KQMLPerformative
    sender: str
    receiver: str
    content: str
    language: str = "KIF"      # Knowledge Interchange Format
    ontology: str = None
    reply_with: str = None
    in_reply_to: str = None
    
    def serialize(self):
        """Serialize to KQML format"""
        parts = [f"({self.performative.value}"]
        parts.append(f":sender {self.sender}")
        parts.append(f":receiver {self.receiver}")
        parts.append(f":content {self.content}")
        parts.append(f":language {self.language}")
        
        if self.ontology:
            parts.append(f":ontology {self.ontology}")
        if self.reply_with:
            parts.append(f":reply-with {self.reply_with}")
        if self.in_reply_to:
            parts.append(f":in-reply-to {self.in_reply_to}")
            
        return " ".join(parts) + ")"
```

## 🔄 Message Passing Patterns

### Point-to-Point Communication

```python
import asyncio
from collections import defaultdict

class MessageBroker:
    """Simple message broker for agent communication"""
    
    def __init__(self):
        self.agents = {}
        self.message_queues = defaultdict(asyncio.Queue)
    
    def register(self, agent_id, agent):
        """Register an agent"""
        self.agents[agent_id] = agent
    
    async def send(self, sender_id, receiver_id, message):
        """Send message to specific agent"""
        if receiver_id not in self.agents:
            raise ValueError(f"Agent {receiver_id} not found")
        
        await self.message_queues[receiver_id].put({
            'sender': sender_id,
            'message': message,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    async def receive(self, agent_id):
        """Receive next message"""
        return await self.message_queues[agent_id].get()
    
    async def broadcast(self, sender_id, message, exclude=None):
        """Broadcast message to all agents"""
        exclude = exclude or []
        tasks = []
        
        for agent_id in self.agents:
            if agent_id != sender_id and agent_id not in exclude:
                tasks.append(self.send(sender_id, agent_id, message))
        
        await asyncio.gather(*tasks)
```

### Publish-Subscribe Pattern

```python
from typing import Callable, Set
import asyncio

class PubSubBroker:
    """Publish-subscribe message broker"""
    
    def __init__(self):
        self.topics = defaultdict(set)  # topic -> subscribers
        self.callbacks = {}  # subscriber_id -> callback
    
    def subscribe(self, subscriber_id: str, topic: str, 
                  callback: Callable):
        """Subscribe to a topic"""
        self.topics[topic].add(subscriber_id)
        self.callbacks[subscriber_id] = callback
    
    def unsubscribe(self, subscriber_id: str, topic: str):
        """Unsubscribe from a topic"""
        if topic in self.topics:
            self.topics[topic].discard(subscriber_id)
    
    async def publish(self, topic: str, message: Any):
        """Publish message to topic"""
        if topic not in self.topics:
            return
        
        tasks = []
        for subscriber_id in self.topics[topic]:
            callback = self.callbacks.get(subscriber_id)
            if callback:
                tasks.append(self._notify(callback, message))
        
        await asyncio.gather(*tasks)
    
    async def _notify(self, callback: Callable, message: Any):
        """Notify subscriber"""
        if asyncio.iscoroutinefunction(callback):
            await callback(message)
        else:
            callback(message)

# Example usage
broker = PubSubBroker()

# Agent subscribes to sensor data
def handle_sensor_data(data):
    print(f"Received sensor data: {data}")

broker.subscribe("agent1", "sensors/temperature", handle_sensor_data)

# Publish sensor reading
await broker.publish("sensors/temperature", {"value": 25.5, "unit": "C"})
```

## 📋 Blackboard Architecture

```python
from typing import Any, Dict, List
import threading

class Blackboard:
    """Shared memory space for agent collaboration"""
    
    def __init__(self):
        self.data = {}
        self.lock = threading.RLock()
        self.observers = defaultdict(list)
    
    def write(self, key: str, value: Any, agent_id: str):
        """Write data to blackboard"""
        with self.lock:
            self.data[key] = {
                'value': value,
                'author': agent_id,
                'timestamp': time.time()
            }
            self._notify_observers(key)
    
    def read(self, key: str) -> Any:
        """Read data from blackboard"""
        with self.lock:
            if key in self.data:
                return self.data[key]['value']
            return None
    
    def query(self, pattern: str) -> Dict:
        """Query blackboard with pattern matching"""
        with self.lock:
            results = {}
            for key, data in self.data.items():
                if pattern in key:
                    results[key] = data['value']
            return results
    
    def observe(self, key: str, callback: Callable):
        """Register observer for key changes"""
        self.observers[key].append(callback)
    
    def _notify_observers(self, key: str):
        """Notify observers of changes"""
        for callback in self.observers.get(key, []):
            callback(key, self.data[key])

class KnowledgeSource:
    """Agent that contributes to blackboard"""
    
    def __init__(self, agent_id: str, blackboard: Blackboard):
        self.agent_id = agent_id
        self.blackboard = blackboard
    
    def can_contribute(self, state: Dict) -> bool:
        """Check if this KS can contribute"""
        raise NotImplementedError
    
    def contribute(self):
        """Write to blackboard"""
        raise NotImplementedError
```

## 🌐 Service-Oriented Architecture (SOA)

```python
from typing import Callable, Dict
import inspect

class ServiceRegistry:
    """Registry for agent services"""
    
    def __init__(self):
        self.services = {}
    
    def register(self, service_name: str, agent_id: str, 
                 service_func: Callable, description: str = ""):
        """Register a service"""
        self.services[service_name] = {
            'provider': agent_id,
            'function': service_func,
            'description': description,
            'signature': str(inspect.signature(service_func))
        }
    
    def discover(self, query: str) -> List[Dict]:
        """Discover services matching query"""
        results = []
        for name, info in self.services.items():
            if query.lower() in name.lower() or \
               query.lower() in info['description'].lower():
                results.append({
                    'name': name,
                    'provider': info['provider'],
                    'description': info['description'],
                    'signature': info['signature']
                })
        return results
    
    async def invoke(self, service_name: str, *args, **kwargs):
        """Invoke a registered service"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        service_func = self.services[service_name]['function']
        
        if asyncio.iscoroutinefunction(service_func):
            return await service_func(*args, **kwargs)
        else:
            return service_func(*args, **kwargs)

# Example usage
registry = ServiceRegistry()

# Agent registers translation service
def translate_text(text: str, target_lang: str) -> str:
    # Translation logic here
    return f"Translated: {text}"

registry.register(
    "translation",
    "agent_translator",
    translate_text,
    "Translates text to target language"
)

# Another agent discovers and uses the service
services = registry.discover("translate")
result = await registry.invoke("translation", "Hello", "es")
```

## 🔒 Secure Communication

```python
from cryptography.fernet import Fernet
import hashlib
import hmac

class SecureChannel:
    """Encrypted communication channel"""
    
    def __init__(self, shared_key: bytes = None):
        if shared_key is None:
            shared_key = Fernet.generate_key()
        self.cipher = Fernet(shared_key)
        self.shared_key = shared_key
    
    def encrypt_message(self, message: str) -> bytes:
        """Encrypt message"""
        return self.cipher.encrypt(message.encode())
    
    def decrypt_message(self, encrypted: bytes) -> str:
        """Decrypt message"""
        return self.cipher.decrypt(encrypted).decode()
    
    def sign_message(self, message: str) -> str:
        """Create HMAC signature"""
        signature = hmac.new(
            self.shared_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, message: str, signature: str) -> bool:
        """Verify HMAC signature"""
        expected = self.sign_message(message)
        return hmac.compare_digest(expected, signature)
```

## 📚 Protocol Comparison

| Protocol | Type | Semantics | Use Case |
|----------|------|-----------|----------|
| **FIPA ACL** | Agent Communication | Rich performatives | Multi-agent systems |
| **KQML** | Knowledge Exchange | Query/manipulation | Knowledge bases |
| **MQTT** | Pub/Sub | Topic-based | IoT, sensors |
| **AMQP** | Message Queue | Routing, reliability | Enterprise systems |
| **gRPC** | RPC | Type-safe | Microservices |
| **WebSocket** | Bidirectional | Real-time | Web agents |

## 🛠️ Libraries & Frameworks

### Python

| Library | Purpose | Repository |
|---------|---------|-----------|
| [SPADE](https://github.com/javipalanca/spade) | FIPA-compliant MAS | [GitHub](https://github.com/javipalanca/spade) |
| [osbrain](https://github.com/opensistemas-hub/osbrain) | Multi-agent with ZeroMQ | [GitHub](https://github.com/opensistemas-hub/osbrain) |
| [paho-mqtt](https://github.com/eclipse/paho.mqtt.python) | MQTT client | [GitHub](https://github.com/eclipse/paho.mqtt.python) |
| [aio-pika](https://github.com/mosquito/aio-pika) | AMQP (RabbitMQ) | [GitHub](https://github.com/mosquito/aio-pika) |

### JavaScript

| Library | Purpose | Repository |
|---------|---------|-----------|
| [MQTT.js](https://github.com/mqttjs/MQTT.js) | MQTT client | [GitHub](https://github.com/mqttjs/MQTT.js) |
| [Socket.IO](https://socket.io/) | Real-time bidirectional | [Website](https://socket.io/) |

### Java

| Library | Purpose | Repository |
|---------|---------|-----------|
| [JADE](https://jade.tilab.com/) | FIPA-compliant platform | [Website](https://jade.tilab.com/) |
| [Apache Kafka](https://kafka.apache.org/) | Distributed streaming | [Website](https://kafka.apache.org/) |

## 📖 Learning Resources

### Books
- **"Agent Communication"** by Chaib-draa & Dignum
- **"Multiagent Systems"** by Wooldridge - Communication chapter
- **"Distributed Systems"** by Tanenbaum - Communication protocols

### Papers
- "FIPA ACL Message Structure Specification" - FIPA
- "KQML as an Agent Communication Language" - Finin et al.
- "The Blackboard Model of Problem Solving" - Erman et al.

### Online Resources
- [FIPA Specifications](http://www.fipa.org/specifications/)
- [MQTT Protocol](https://mqtt.org/)
- [gRPC Documentation](https://grpc.io/docs/)

## 🔗 Related Topics

- [Multi-Agent Systems](../Core-Concepts/Multi-Agent-Systems.md)
- [Agent Architectures](./Agent-Architectures.md)
- [Orchestration Frameworks](../Frameworks-Tools/Orchestration-Frameworks.md)
- [System Design](../Supporting-Skills/System-Design.md)

---

*This document covers essential communication protocols for agent systems. For implementation examples, refer to the framework documentation and code samples provided.*