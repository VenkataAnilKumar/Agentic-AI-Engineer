# 🔌 APIs & Pipelines

## 📋 Overview

APIs and data pipelines form the backbone of agent system integration, enabling communication with external services, data ingestion, and action execution. This guide covers modern API patterns and pipeline architectures for production agent systems.

## 🌐 RESTful APIs

### Building Agent APIs with FastAPI

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid

app = FastAPI(title="Agent API", version="1.0.0")

# Data models
class AgentRequest(BaseModel):
    task: str
    context: Optional[dict] = {}
    priority: int = 1

class AgentResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskResult(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

# In-memory task store
tasks = {}

@app.post("/agent/execute", response_model=AgentResponse)
async def execute_task(request: AgentRequest, background_tasks: BackgroundTasks):
    """Execute agent task asynchronously"""
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        "status": "pending",
        "request": request.dict()
    }
    
    # Add background task
    background_tasks.add_task(process_agent_task, task_id, request)
    
    return AgentResponse(
        task_id=task_id,
        status="accepted",
        message="Task queued for processing"
    )

@app.get("/agent/status/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """Check task status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResult(**tasks[task_id])

@app.get("/agent/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "active_tasks": len(tasks)}

async def process_agent_task(task_id: str, request: AgentRequest):
    """Background task processor"""
    try:
        # Simulate agent processing
        tasks[task_id]["status"] = "processing"
        
        # Agent logic here
        result = {"output": f"Processed: {request.task}"}
        
        tasks[task_id].update({
            "status": "completed",
            "result": result
        })
    except Exception as e:
        tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })

# Run with: uvicorn main:app --reload
```

### API Client for Agent Integration

```python
import httpx
import asyncio
from typing import Dict, Any

class AgentAPIClient:
    """Client for interacting with agent API"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def execute_task(self, task: str, context: Dict = None) -> str:
        """Submit task to agent"""
        response = await self.client.post(
            f"{self.base_url}/agent/execute",
            json={"task": task, "context": context or {}},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["task_id"]
    
    async def get_result(self, task_id: str) -> Dict[str, Any]:
        """Poll for task result"""
        response = await self.client.get(
            f"{self.base_url}/agent/status/{task_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    async def wait_for_completion(self, task_id: str, max_wait: int = 300):
        """Wait for task completion with polling"""
        for _ in range(max_wait):
            result = await self.get_result(task_id)
            
            if result["status"] in ["completed", "failed"]:
                return result
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Task {task_id} did not complete")
    
    async def close(self):
        await self.client.aclose()

# Usage
async def main():
    client = AgentAPIClient("http://localhost:8000")
    
    task_id = await client.execute_task("Analyze sentiment of reviews")
    result = await client.wait_for_completion(task_id)
    
    print(f"Result: {result}")
    await client.close()
```

## 📊 GraphQL APIs

### Agent GraphQL Schema

```python
import strawberry
from typing import List, Optional
import asyncio

@strawberry.type
class Agent:
    id: str
    name: str
    status: str
    capabilities: List[str]

@strawberry.type
class Task:
    id: str
    description: str
    status: str
    agent_id: Optional[str] = None
    result: Optional[str] = None

@strawberry.type
class Query:
    @strawberry.field
    def agents(self) -> List[Agent]:
        """Get all agents"""
        return [
            Agent(id="1", name="Agent Alpha", status="active", 
                  capabilities=["planning", "reasoning"]),
            Agent(id="2", name="Agent Beta", status="idle",
                  capabilities=["vision", "nlp"])
        ]
    
    @strawberry.field
    def agent(self, id: str) -> Optional[Agent]:
        """Get specific agent"""
        agents = self.agents()
        return next((a for a in agents if a.id == id), None)
    
    @strawberry.field
    def tasks(self, agent_id: Optional[str] = None) -> List[Task]:
        """Get tasks, optionally filtered by agent"""
        all_tasks = [
            Task(id="1", description="Analyze data", status="completed", agent_id="1"),
            Task(id="2", description="Generate report", status="pending", agent_id="2")
        ]
        
        if agent_id:
            return [t for t in all_tasks if t.agent_id == agent_id]
        return all_tasks

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def assign_task(self, task_id: str, agent_id: str) -> Task:
        """Assign task to agent"""
        # Assignment logic here
        return Task(
            id=task_id,
            description="Sample task",
            status="assigned",
            agent_id=agent_id
        )
    
    @strawberry.mutation
    async def create_task(self, description: str) -> Task:
        """Create new task"""
        task_id = str(uuid.uuid4())
        return Task(
            id=task_id,
            description=description,
            status="pending"
        )

schema = strawberry.Schema(query=Query, mutation=Mutation)

# Run with Strawberry + FastAPI
from strawberry.fastapi import GraphQLRouter

graphql_app = GraphQLRouter(schema)

# Mount in FastAPI
app.include_router(graphql_app, prefix="/graphql")
```

## ⚡ gRPC for High-Performance Communication

### Protocol Buffer Definition

```protobuf
// agent.proto
syntax = "proto3";

package agent;

service AgentService {
  rpc ExecuteTask(TaskRequest) returns (TaskResponse);
  rpc StreamResults(TaskRequest) returns (stream TaskUpdate);
  rpc GetAgentStatus(AgentStatusRequest) returns (AgentStatusResponse);
}

message TaskRequest {
  string task_id = 1;
  string task_description = 2;
  map<string, string> context = 3;
  int32 priority = 4;
}

message TaskResponse {
  string task_id = 1;
  string status = 2;
  string message = 3;
}

message TaskUpdate {
  string task_id = 1;
  string status = 2;
  int32 progress = 3;
  string message = 4;
}

message AgentStatusRequest {
  string agent_id = 1;
}

message AgentStatusResponse {
  string agent_id = 1;
  string status = 2;
  int32 active_tasks = 3;
}
```

### gRPC Server Implementation

```python
import grpc
from concurrent import futures
import agent_pb2
import agent_pb2_grpc
import time

class AgentServicer(agent_pb2_grpc.AgentServiceServicer):
    """gRPC service implementation"""
    
    def ExecuteTask(self, request, context):
        """Execute task synchronously"""
        print(f"Executing task: {request.task_description}")
        
        # Process task
        time.sleep(1)  # Simulate work
        
        return agent_pb2.TaskResponse(
            task_id=request.task_id,
            status="completed",
            message="Task executed successfully"
        )
    
    def StreamResults(self, request, context):
        """Stream task progress"""
        for progress in range(0, 101, 20):
            yield agent_pb2.TaskUpdate(
                task_id=request.task_id,
                status="processing",
                progress=progress,
                message=f"Progress: {progress}%"
            )
            time.sleep(0.5)
        
        yield agent_pb2.TaskUpdate(
            task_id=request.task_id,
            status="completed",
            progress=100,
            message="Task completed"
        )
    
    def GetAgentStatus(self, request, context):
        """Get agent status"""
        return agent_pb2.AgentStatusResponse(
            agent_id=request.agent_id,
            status="active",
            active_tasks=3
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(
        AgentServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

## 🐰 Message Queues

### RabbitMQ Integration

```python
import pika
import json
from typing import Callable

class MessageQueue:
    """RabbitMQ wrapper for agent communication"""
    
    def __init__(self, host='localhost', queue_name='agent_tasks'):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host)
        )
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        
        # Declare queue
        self.channel.queue_declare(queue=queue_name, durable=True)
    
    def publish(self, message: dict):
        """Publish message to queue"""
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
            )
        )
        print(f"Published: {message}")
    
    def consume(self, callback: Callable):
        """Consume messages from queue"""
        def on_message(ch, method, properties, body):
            message = json.loads(body)
            callback(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=on_message
        )
        
        print("Waiting for messages...")
        self.channel.start_consuming()
    
    def close(self):
        self.connection.close()

# Agent producer
def agent_producer():
    queue = MessageQueue(queue_name='agent_tasks')
    
    queue.publish({
        'task_id': '123',
        'task': 'Analyze dataset',
        'priority': 1
    })
    
    queue.close()

# Agent consumer
def process_task(message):
    print(f"Processing task: {message['task']}")
    # Agent logic here

def agent_consumer():
    queue = MessageQueue(queue_name='agent_tasks')
    queue.consume(process_task)
```

### Apache Kafka for Stream Processing

```python
from kafka import KafkaProducer, KafkaConsumer
import json

class KafkaAgentPipeline:
    """Kafka-based agent pipeline"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
    
    def create_producer(self) -> KafkaProducer:
        """Create Kafka producer"""
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def create_consumer(self, topic: str, group_id: str) -> KafkaConsumer:
        """Create Kafka consumer"""
        return KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest'
        )
    
    def publish_event(self, topic: str, event: dict):
        """Publish event to topic"""
        producer = self.create_producer()
        producer.send(topic, event)
        producer.flush()
        producer.close()
    
    def consume_events(self, topic: str, group_id: str, handler: Callable):
        """Consume events from topic"""
        consumer = self.create_consumer(topic, group_id)
        
        for message in consumer:
            handler(message.value)

# Usage
pipeline = KafkaAgentPipeline()

# Publish agent decision
pipeline.publish_event('agent_decisions', {
    'agent_id': 'agent_1',
    'decision': 'execute_plan_a',
    'confidence': 0.95
})

# Consume and process
def handle_decision(event):
    print(f"Agent {event['agent_id']}: {event['decision']}")

pipeline.consume_events('agent_decisions', 'processor_group', handle_decision)
```

## 🔄 ETL Pipelines

### Data Pipeline with Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def extract_data(**context):
    """Extract data from sources"""
    # Extraction logic
    data = {"records": 1000, "source": "database"}
    return data

def transform_data(**context):
    """Transform extracted data"""
    ti = context['task_instance']
    data = ti.xcom_pull(task_ids='extract')
    
    # Transformation logic
    transformed = {
        "processed_records": data["records"],
        "transformations_applied": ["normalize", "filter"]
    }
    return transformed

def load_data(**context):
    """Load transformed data"""
    ti = context['task_instance']
    data = ti.xcom_pull(task_ids='transform')
    
    # Load to agent system
    print(f"Loaded {data['processed_records']} records")

# Define DAG
default_args = {
    'owner': 'agent_system',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'agent_data_pipeline',
    default_args=default_args,
    description='ETL pipeline for agent training data',
    schedule_interval='@daily',
    catchup=False
)

extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag
)

# Set dependencies
extract_task >> transform_task >> load_task
```

## 🛠️ Libraries & Tools

### API Frameworks

| Library | Type | Performance | Use Case |
|---------|------|-------------|----------|
| [FastAPI](https://github.com/tiangolo/fastapi) | REST | Excellent | Modern async APIs |
| [Flask](https://github.com/pallets/flask) | REST | Good | Simple APIs |
| [Strawberry](https://github.com/strawberry-graphql/strawberry) | GraphQL | Good | Python GraphQL |
| [gRPC](https://grpc.io/) | RPC | Excellent | High-performance |

### Message Queues

| Library | Pattern | Scalability | Repository |
|---------|---------|-------------|-----------|
| [RabbitMQ](https://www.rabbitmq.com/) | Queue/Pub-Sub | High | [GitHub](https://github.com/rabbitmq/rabbitmq-server) |
| [Apache Kafka](https://kafka.apache.org/) | Stream | Very High | [GitHub](https://github.com/apache/kafka) |
| [Redis](https://redis.io/) | Pub-Sub/Queue | High | [GitHub](https://github.com/redis/redis) |
| [Celery](https://github.com/celery/celery) | Task Queue | High | [GitHub](https://github.com/celery/celery) |

### Pipeline Orchestration

| Tool | Complexity | Features | Repository |
|------|-----------|----------|-----------|
| [Apache Airflow](https://github.com/apache/airflow) | Medium-High | DAGs, scheduling | [GitHub](https://github.com/apache/airflow) |
| [Prefect](https://github.com/PrefectHQ/prefect) | Medium | Modern, Pythonic | [GitHub](https://github.com/PrefectHQ/prefect) |
| [Dagster](https://github.com/dagster-io/dagster) | Medium | Data-aware | [GitHub](https://github.com/dagster-io/dagster) |

## 📚 Learning Resources

### Books
- **"Designing Data-Intensive Applications"** by Kleppmann
- **"RESTful Web APIs"** by Richardson & Ruby
- **"Building Microservices"** by Newman

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GraphQL Best Practices](https://graphql.org/learn/best-practices/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)

## 🔗 Related Topics

- [Cloud Platforms](./Cloud-Platforms.md)
- [Orchestration Frameworks](../Frameworks-Tools/Orchestration-Frameworks.md)
- [System Design](../Supporting-Skills/System-Design.md)
- [Communication Protocols](../Architecture-Design/Communication-Protocols.md)

---

*This guide covers modern API patterns and pipeline architectures for production agent systems. For deployment strategies, see the Cloud Platforms guide.*