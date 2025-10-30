# 🏗️ System Design

## 📋 Overview

Designing scalable, reliable agent systems requires understanding distributed systems principles, architectural patterns, and operational best practices. This guide covers system design for production AI agents.

## 🎯 System Architecture

### Microservices for Agent Systems

```
┌─────────────────────────────────────────────────────┐
│                  API Gateway                         │
│             (Load Balancer + Routing)                │
└───────────┬─────────────┬──────────────┬────────────┘
            │             │              │
    ┌───────▼──────┐ ┌───▼─────────┐ ┌─▼──────────────┐
    │   Agent      │ │  Perception │ │   Planning     │
    │  Orchestrator│ │   Service   │ │   Service      │
    └───────┬──────┘ └───┬─────────┘ └─┬──────────────┘
            │            │              │
    ┌───────▼────────────▼──────────────▼──────────────┐
    │         Message Queue (Kafka/RabbitMQ)           │
    └───────┬────────────┬──────────────┬──────────────┘
            │            │              │
    ┌───────▼──────┐ ┌──▼──────────┐ ┌─▼──────────────┐
    │  Execution   │ │  Learning   │ │   Monitoring   │
    │   Service    │ │   Service   │ │    Service     │
    └──────────────┘ └─────────────┘ └────────────────┘
```

### Service Communication

```python
# FastAPI Microservice Example
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

class AgentRequest(BaseModel):
    state: list
    task: str

class AgentResponse(BaseModel):
    action: str
    confidence: float

@app.post("/agent/decision", response_model=AgentResponse)
async def make_decision(request: AgentRequest):
    """Agent decision-making endpoint"""
    try:
        # Call perception service
        async with httpx.AsyncClient() as client:
            perception = await client.post(
                "http://perception-service:8001/analyze",
                json={"state": request.state}
            )
        
        # Call planning service
        async with httpx.AsyncClient() as client:
            plan = await client.post(
                "http://planning-service:8002/plan",
                json={
                    "perception": perception.json(),
                    "task": request.task
                }
            )
        
        return AgentResponse(
            action=plan.json()["action"],
            confidence=plan.json()["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🔄 Load Balancing

### Round-Robin with Health Checks

```python
from typing import List
import random

class ServiceInstance:
    def __init__(self, url: str):
        self.url = url
        self.healthy = True
        self.request_count = 0
    
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.url}/health", timeout=2.0)
                self.healthy = response.status_code == 200
        except:
            self.healthy = False
        return self.healthy

class LoadBalancer:
    """Round-robin load balancer with health checks"""
    
    def __init__(self, instances: List[str]):
        self.instances = [ServiceInstance(url) for url in instances]
        self.current_index = 0
    
    async def get_healthy_instance(self) -> ServiceInstance:
        """Get next healthy instance"""
        attempts = 0
        max_attempts = len(self.instances)
        
        while attempts < max_attempts:
            instance = self.instances[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.instances)
            
            if instance.healthy or await instance.health_check():
                return instance
            
            attempts += 1
        
        raise Exception("No healthy instances available")
    
    async def make_request(self, endpoint: str, data: dict):
        """Make load-balanced request"""
        instance = await self.get_healthy_instance()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{instance.url}{endpoint}",
                    json=data
                )
            instance.request_count += 1
            return response.json()
        except Exception as e:
            instance.healthy = False
            raise

# Usage
lb = LoadBalancer([
    "http://agent-1:8000",
    "http://agent-2:8000",
    "http://agent-3:8000"
])

result = await lb.make_request("/agent/decision", {"state": [1, 2, 3]})
```

## 💾 Caching Strategies

### Multi-Level Caching

```python
import redis
from functools import wraps
import pickle
import hashlib

class CacheManager:
    """Multi-level cache (Memory + Redis)"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.memory_cache = {}
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
    
    def _generate_key(self, func_name: str, args, kwargs) -> str:
        """Generate cache key"""
        key_data = f"{func_name}:{args}:{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache(self, ttl: int = 3600):
        """Caching decorator"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Check memory cache
                if cache_key in self.memory_cache:
                    return self.memory_cache[cache_key]
                
                # Check Redis cache
                cached = self.redis_client.get(cache_key)
                if cached:
                    result = pickle.loads(cached)
                    self.memory_cache[cache_key] = result
                    return result
                
                # Compute and cache
                result = func(*args, **kwargs)
                
                # Store in both caches
                self.memory_cache[cache_key] = result
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(result)
                )
                
                return result
            return wrapper
        return decorator

# Usage
cache_manager = CacheManager()

@cache_manager.cache(ttl=3600)
def get_agent_policy(agent_id: str, state: tuple):
    """Expensive policy computation"""
    # Complex computation
    return compute_policy(agent_id, state)
```

### Cache Invalidation

```python
class SmartCache:
    """Cache with intelligent invalidation"""
    
    def __init__(self):
        self.cache = {}
        self.dependencies = {}
    
    def set(self, key: str, value, dependencies: List[str] = None):
        """Set value with dependencies"""
        self.cache[key] = value
        if dependencies:
            self.dependencies[key] = dependencies
    
    def get(self, key: str):
        """Get cached value"""
        return self.cache.get(key)
    
    def invalidate(self, key: str):
        """Invalidate key and dependents"""
        if key in self.cache:
            del self.cache[key]
        
        # Invalidate dependent keys
        for cached_key, deps in list(self.dependencies.items()):
            if key in deps:
                self.invalidate(cached_key)
```

## 🗄️ Database Design

### Time-Series for Agent Metrics

```python
from influxdb_client import InfluxDBClient, Point
from datetime import datetime

class MetricsStore:
    """Time-series storage for agent metrics"""
    
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()
        self.bucket = bucket
    
    def record_metric(self, agent_id: str, metric: str, value: float):
        """Record agent metric"""
        point = Point("agent_metrics") \
            .tag("agent_id", agent_id) \
            .field(metric, value) \
            .time(datetime.utcnow())
        
        self.write_api.write(bucket=self.bucket, record=point)
    
    def get_metrics(self, agent_id: str, metric: str, hours: int = 24):
        """Query metrics"""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r._measurement == "agent_metrics")
          |> filter(fn: (r) => r.agent_id == "{agent_id}")
          |> filter(fn: (r) => r._field == "{metric}")
        '''
        
        return self.query_api.query(query)
```

### Document Store for Agent States

```python
from pymongo import MongoClient
from datetime import datetime

class AgentStateStore:
    """MongoDB for agent state persistence"""
    
    def __init__(self, connection_string: str):
        self.client = MongoClient(connection_string)
        self.db = self.client.agent_system
        self.states = self.db.agent_states
    
    def save_state(self, agent_id: str, state: dict):
        """Save agent state"""
        document = {
            'agent_id': agent_id,
            'state': state,
            'timestamp': datetime.utcnow(),
            'version': self._get_next_version(agent_id)
        }
        
        self.states.insert_one(document)
    
    def get_latest_state(self, agent_id: str) -> dict:
        """Get most recent state"""
        return self.states.find_one(
            {'agent_id': agent_id},
            sort=[('timestamp', -1)]
        )
    
    def get_state_history(self, agent_id: str, limit: int = 10):
        """Get state history"""
        return list(self.states.find(
            {'agent_id': agent_id},
            sort=[('timestamp', -1)],
            limit=limit
        ))
    
    def _get_next_version(self, agent_id: str) -> int:
        """Get next version number"""
        latest = self.get_latest_state(agent_id)
        return (latest['version'] + 1) if latest else 1
```

## 🛡️ Fault Tolerance

### Circuit Breaker Pattern

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for service calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)

def call_external_service():
    try:
        return breaker.call(external_api_call)
    except Exception as e:
        # Fallback logic
        return get_cached_response()
```

### Retry with Exponential Backoff

```python
import time
import random

def retry_with_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Retry with exponential backoff"""
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            
            if retries >= max_retries:
                raise
            
            # Calculate delay
            delay = min(delay * exponential_base, max_delay)
            
            # Add jitter
            if jitter:
                delay = delay * (0.5 + random.random())
            
            print(f"Retry {retries}/{max_retries} after {delay:.2f}s")
            time.sleep(delay)
```

## 📊 Monitoring & Observability

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

def agent_pipeline(state):
    """Agent pipeline with tracing"""
    with tracer.start_as_current_span("agent_pipeline") as span:
        span.set_attribute("state_size", len(state))
        
        # Perception
        with tracer.start_as_current_span("perception"):
            features = extract_features(state)
        
        # Planning
        with tracer.start_as_current_span("planning"):
            plan = generate_plan(features)
        
        # Execution
        with tracer.start_as_current_span("execution"):
            result = execute_plan(plan)
        
        span.set_attribute("result_status", result.status)
        return result
```

## 📚 Resources

### System Design
- **"Designing Data-Intensive Applications"** by Martin Kleppmann
- **"System Design Interview"** by Alex Xu
- **[AWS Architecture Center](https://aws.amazon.com/architecture/)**
- **[Google Cloud Architecture](https://cloud.google.com/architecture)**

### Tools
- **Kubernetes** - Container orchestration
- **Istio** - Service mesh
- **Prometheus** - Monitoring
- **Grafana** - Visualization
- **Jaeger** - Distributed tracing

## 🔗 Related Topics

- [Cloud Platforms](../Integration-Deployment/Cloud-Platforms.md)
- [APIs & Pipelines](../Integration-Deployment/APIs-Pipelines.md)
- [Programming Fundamentals](./Programming-Fundamentals.md)
- [Automation & Scheduling](../Integration-Deployment/Automation-Scheduling.md)

---

*This guide covers system design for scalable agent architectures. For implementation details, see related topics.*