# ⏰ Automation & Scheduling

## 📋 Overview

Automation and scheduling are critical for orchestrating agent tasks, managing resources, and ensuring reliable execution. This guide covers workflow engines, task schedulers, and event-driven architectures for production agent systems.

## ⏱️ Task Schedulers

### APScheduler - Python Scheduling Library

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import logging

class AgentScheduler:
    """Schedule agent tasks"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        logging.basicConfig(level=logging.INFO)
    
    def schedule_cron(self, task, cron_expr, task_id=None):
        """Schedule with cron expression"""
        self.scheduler.add_job(
            task,
            trigger=CronTrigger.from_crontab(cron_expr),
            id=task_id,
            replace_existing=True
        )
        logging.info(f"Scheduled task {task_id} with cron: {cron_expr}")
    
    def schedule_interval(self, task, seconds, task_id=None):
        """Schedule at fixed intervals"""
        self.scheduler.add_job(
            task,
            trigger=IntervalTrigger(seconds=seconds),
            id=task_id,
            replace_existing=True
        )
        logging.info(f"Scheduled task {task_id} every {seconds}s")
    
    def schedule_once(self, task, run_date, task_id=None):
        """Schedule one-time execution"""
        self.scheduler.add_job(
            task,
            trigger='date',
            run_date=run_date,
            id=task_id
        )
        logging.info(f"Scheduled task {task_id} at {run_date}")
    
    def remove_job(self, task_id):
        """Remove scheduled task"""
        self.scheduler.remove_job(task_id)
    
    def pause_job(self, task_id):
        """Pause a job"""
        self.scheduler.pause_job(task_id)
    
    def resume_job(self, task_id):
        """Resume a paused job"""
        self.scheduler.resume_job(task_id)
    
    def shutdown(self):
        """Shutdown scheduler"""
        self.scheduler.shutdown()

# Define agent tasks
def monitor_system():
    print(f"[{datetime.now()}] Monitoring system...")
    # Agent monitoring logic

def process_daily_data():
    print(f"[{datetime.now()}] Processing daily data...")
    # Data processing logic

def cleanup_old_logs():
    print(f"[{datetime.now()}] Cleaning up logs...")
    # Cleanup logic

# Usage
scheduler = AgentScheduler()

# Every minute
scheduler.schedule_cron(monitor_system, "* * * * *", "system_monitor")

# Daily at 2 AM
scheduler.schedule_cron(process_daily_data, "0 2 * * *", "daily_processing")

# Every 6 hours
scheduler.schedule_interval(cleanup_old_logs, 6*3600, "log_cleanup")
```

### Celery - Distributed Task Queue

```python
from celery import Celery, group, chain, chord
from celery.schedules import crontab
import time

# Initialize Celery
app = Celery(
    'agent_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Define tasks
@app.task(bind=True, max_retries=3)
def analyze_data(self, data_id):
    """Analyze data with retries"""
    try:
        print(f"Analyzing data: {data_id}")
        time.sleep(2)  # Simulate work
        return {"status": "success", "data_id": data_id}
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

@app.task
def generate_report(analysis_results):
    """Generate report from analysis"""
    print(f"Generating report from: {analysis_results}")
    return {"report": "generated", "results": analysis_results}

@app.task
def send_notification(report):
    """Send notification"""
    print(f"Sending notification: {report}")
    return {"sent": True}

@app.task
def aggregate_results(results):
    """Aggregate multiple results"""
    print(f"Aggregating {len(results)} results")
    return {"aggregated": results}

# Task composition patterns

# Chain: Execute sequentially
def process_pipeline(data_id):
    """Chain tasks together"""
    workflow = chain(
        analyze_data.s(data_id),
        generate_report.s(),
        send_notification.s()
    )
    return workflow.apply_async()

# Group: Execute in parallel
def parallel_analysis(data_ids):
    """Analyze multiple datasets in parallel"""
    job = group(analyze_data.s(data_id) for data_id in data_ids)
    return job.apply_async()

# Chord: Map-Reduce pattern
def map_reduce_analysis(data_ids):
    """Map-reduce style processing"""
    workflow = chord(
        group(analyze_data.s(data_id) for data_id in data_ids),
        aggregate_results.s()
    )
    return workflow.apply_async()

# Periodic tasks with Celery Beat
app.conf.beat_schedule = {
    'monitor-every-minute': {
        'task': 'agent_tasks.analyze_data',
        'schedule': 60.0,  # Every 60 seconds
        'args': ('system_metrics',)
    },
    'daily-report': {
        'task': 'agent_tasks.generate_report',
        'schedule': crontab(hour=8, minute=0),  # Daily at 8 AM
        'args': ({'daily': True},)
    },
    'weekly-cleanup': {
        'task': 'agent_tasks.cleanup',
        'schedule': crontab(day_of_week='sunday', hour=2, minute=0),
    }
}

# Run worker: celery -A agent_tasks worker --loglevel=info
# Run beat: celery -A agent_tasks beat --loglevel=info
```

## 🌊 Workflow Engines

### Prefect - Modern Workflow Orchestration

```python
from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import ConcurrentTaskRunner
from datetime import timedelta
import httpx

@task(retries=3, retry_delay_seconds=60)
def fetch_data(source_url: str):
    """Fetch data from source"""
    logger = get_run_logger()
    logger.info(f"Fetching from {source_url}")
    
    response = httpx.get(source_url)
    response.raise_for_status()
    return response.json()

@task
def transform_data(raw_data: dict):
    """Transform raw data"""
    logger = get_run_logger()
    logger.info("Transforming data")
    
    # Transformation logic
    transformed = {
        "processed": True,
        "record_count": len(raw_data.get("items", []))
    }
    return transformed

@task
def store_results(data: dict, destination: str):
    """Store processed data"""
    logger = get_run_logger()
    logger.info(f"Storing results to {destination}")
    
    # Storage logic
    return {"stored": True, "location": destination}

@flow(name="agent-etl-pipeline", task_runner=ConcurrentTaskRunner())
def etl_pipeline(sources: list, destination: str):
    """ETL pipeline flow"""
    logger = get_run_logger()
    logger.info("Starting ETL pipeline")
    
    # Fetch from multiple sources in parallel
    raw_data_futures = []
    for source in sources:
        future = fetch_data.submit(source)
        raw_data_futures.append(future)
    
    # Wait for all fetches
    raw_data_results = [f.result() for f in raw_data_futures]
    
    # Transform sequentially
    transformed_results = []
    for raw_data in raw_data_results:
        transformed = transform_data(raw_data)
        transformed_results.append(transformed)
    
    # Store results
    store_result = store_results(transformed_results, destination)
    
    return store_result

# Create deployment with schedule
deployment = Deployment.build_from_flow(
    flow=etl_pipeline,
    name="daily-etl",
    schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2 AM
    parameters={
        "sources": ["http://api.example.com/data"],
        "destination": "s3://bucket/data"
    },
    work_queue_name="agent-queue"
)

# Deploy: deployment.apply()
```

### Apache Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'agent-system',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

def train_agent(**context):
    """Train agent model"""
    print("Training agent...")
    # Training logic
    return {"model_path": "/models/agent_v1.pth", "accuracy": 0.95}

def evaluate_agent(**context):
    """Evaluate agent performance"""
    ti = context['task_instance']
    training_result = ti.xcom_pull(task_ids='train_agent')
    
    print(f"Evaluating model: {training_result['model_path']}")
    # Evaluation logic
    return {"passed": True}

def deploy_agent(**context):
    """Deploy agent to production"""
    ti = context['task_instance']
    eval_result = ti.xcom_pull(task_ids='evaluate_agent')
    
    if eval_result['passed']:
        print("Deploying to production...")
        # Deployment logic
        return {"deployed": True, "version": "v1.0.0"}
    else:
        raise ValueError("Evaluation failed, skipping deployment")

with DAG(
    'agent_training_pipeline',
    default_args=default_args,
    description='Automated agent training and deployment',
    schedule_interval='@daily',
    catchup=False,
    tags=['agent', 'ml', 'production']
) as dag:
    
    # Wait for data preparation
    wait_for_data = ExternalTaskSensor(
        task_id='wait_for_data',
        external_dag_id='data_preparation',
        external_task_id='prepare_data',
        timeout=600
    )
    
    # Prepare environment
    setup = BashOperator(
        task_id='setup_environment',
        bash_command='source /opt/venv/bin/activate && pip install -r requirements.txt'
    )
    
    # Train agent
    train = PythonOperator(
        task_id='train_agent',
        python_callable=train_agent,
        provide_context=True
    )
    
    # Evaluate
    evaluate = PythonOperator(
        task_id='evaluate_agent',
        python_callable=evaluate_agent,
        provide_context=True
    )
    
    # Deploy
    deploy = PythonOperator(
        task_id='deploy_agent',
        python_callable=deploy_agent,
        provide_context=True
    )
    
    # Notify
    notify = BashOperator(
        task_id='send_notification',
        bash_command='echo "Agent deployed successfully" | mail -s "Deployment Success" team@example.com'
    )
    
    # Define dependencies
    wait_for_data >> setup >> train >> evaluate >> deploy >> notify
```

## 🎯 Event-Driven Architecture

### Event-Driven Agent System

```python
from typing import Callable, Dict, List
import asyncio
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    AGENT_IDLE = "agent_idle"
    SYSTEM_ERROR = "system_error"
    DATA_AVAILABLE = "data_available"

@dataclass
class Event:
    type: EventType
    data: Dict
    timestamp: float

class EventBus:
    """Central event bus for agent system"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history = []
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from event"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
    
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        self.event_history.append(event)
        
        if event.type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event.type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    handler(event)
            
            if tasks:
                await asyncio.gather(*tasks)

# Event handlers
async def on_task_created(event: Event):
    """Handle new task creation"""
    print(f"New task created: {event.data['task_id']}")
    # Assign task to available agent
    await assign_task_to_agent(event.data['task_id'])

async def on_task_completed(event: Event):
    """Handle task completion"""
    print(f"Task completed: {event.data['task_id']}")
    # Update metrics, notify stakeholders
    await update_metrics(event.data)

async def on_agent_idle(event: Event):
    """Handle idle agent"""
    print(f"Agent idle: {event.data['agent_id']}")
    # Assign new task if available
    await check_pending_tasks(event.data['agent_id'])

async def on_system_error(event: Event):
    """Handle system errors"""
    print(f"System error: {event.data['error']}")
    # Alert, log, attempt recovery
    await alert_ops_team(event.data)

# Setup event bus
event_bus = EventBus()
event_bus.subscribe(EventType.TASK_CREATED, on_task_created)
event_bus.subscribe(EventType.TASK_COMPLETED, on_task_completed)
event_bus.subscribe(EventType.AGENT_IDLE, on_agent_idle)
event_bus.subscribe(EventType.SYSTEM_ERROR, on_system_error)

# Example usage
async def main():
    # Publish events
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"task_id": "123", "description": "Analyze data"},
        timestamp=time.time()
    ))
    
    await asyncio.sleep(1)
    
    await event_bus.publish(Event(
        type=EventType.TASK_COMPLETED,
        data={"task_id": "123", "result": "success"},
        timestamp=time.time()
    ))
```

## 📊 Priority Management

### Priority Queue Scheduler

```python
import heapq
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

@dataclass(order=True)
class PrioritizedTask:
    """Task with priority"""
    priority: int
    task_id: str = field(compare=False)
    task_data: Any = field(compare=False)
    created_at: datetime = field(default_factory=datetime.now, compare=False)

class PriorityTaskScheduler:
    """Schedule tasks based on priority"""
    
    def __init__(self):
        self.task_queue = []
        self.active_tasks = {}
    
    def add_task(self, task_id: str, priority: int, task_data: Any):
        """Add task to queue (lower priority number = higher priority)"""
        task = PrioritizedTask(priority, task_id, task_data)
        heapq.heappush(self.task_queue, task)
        print(f"Added task {task_id} with priority {priority}")
    
    def get_next_task(self) -> PrioritizedTask:
        """Get highest priority task"""
        if self.task_queue:
            task = heapq.heappop(self.task_queue)
            self.active_tasks[task.task_id] = task
            return task
        return None
    
    def complete_task(self, task_id: str):
        """Mark task as completed"""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            print(f"Completed task {task_id}")
    
    def update_priority(self, task_id: str, new_priority: int):
        """Update task priority"""
        # Remove and re-add with new priority
        self.task_queue = [t for t in self.task_queue if t.task_id != task_id]
        heapq.heapify(self.task_queue)
        
        # Re-add if was in queue
        for task in self.task_queue:
            if task.task_id == task_id:
                self.add_task(task_id, new_priority, task.task_data)
                break

# Usage
scheduler = PriorityTaskScheduler()

# Add tasks with different priorities
scheduler.add_task("task1", priority=1, task_data={"type": "critical"})
scheduler.add_task("task2", priority=5, task_data={"type": "normal"})
scheduler.add_task("task3", priority=3, task_data={"type": "important"})

# Process tasks in priority order
while task := scheduler.get_next_task():
    print(f"Processing: {task.task_id} (priority: {task.priority})")
    # Process task...
    scheduler.complete_task(task.task_id)
```

## 🛠️ Libraries & Tools

### Task Schedulers

| Library | Type | Distributed | Repository |
|---------|------|-------------|-----------|
| [APScheduler](https://github.com/agronholm/apscheduler) | Scheduler | No | [GitHub](https://github.com/agronholm/apscheduler) |
| [Celery](https://github.com/celery/celery) | Task Queue | Yes | [GitHub](https://github.com/celery/celery) |
| [RQ](https://github.com/rq/rq) | Simple Queue | Yes | [GitHub](https://github.com/rq/rq) |
| [Huey](https://github.com/coleifer/huey) | Task Queue | No | [GitHub](https://github.com/coleifer/huey) |

### Workflow Engines

| Tool | Complexity | Features | Repository |
|------|-----------|----------|-----------|
| [Apache Airflow](https://github.com/apache/airflow) | High | DAGs, rich UI | [GitHub](https://github.com/apache/airflow) |
| [Prefect](https://github.com/PrefectHQ/prefect) | Medium | Modern, Pythonic | [GitHub](https://github.com/PrefectHQ/prefect) |
| [Temporal](https://github.com/temporalio/temporal) | Medium | Durable execution | [GitHub](https://github.com/temporalio/temporal) |
| [Dagster](https://github.com/dagster-io/dagster) | Medium | Data-aware | [GitHub](https://github.com/dagster-io/dagster) |

## 📚 Learning Resources

### Documentation
- [Celery Documentation](https://docs.celeryq.dev/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Prefect Guide](https://docs.prefect.io/)

### Books
- **"Data Pipelines Pocket Reference"** by Densmore
- **"Designing Event-Driven Systems"** by Stopford

## 🔗 Related Topics

- [APIs & Pipelines](./APIs-Pipelines.md)
- [Cloud Platforms](./Cloud-Platforms.md)
- [Orchestration Frameworks](../Frameworks-Tools/Orchestration-Frameworks.md)
- [System Design](../Supporting-Skills/System-Design.md)

---

*This guide covers automation and scheduling patterns for production agent systems. For workflow orchestration at scale, see the Cloud Platforms guide.*