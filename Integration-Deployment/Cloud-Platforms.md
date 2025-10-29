# ☁️ Cloud Platforms

## 📋 Overview

Cloud platforms provide scalable infrastructure for deploying and managing agent systems. This guide covers deployment strategies, services, and best practices across major cloud providers.

## ☁️ Amazon Web Services (AWS)

### AWS Services for Agent Systems

| Service | Purpose | Use Case |
|---------|---------|----------|
| **SageMaker** | ML model training/deployment | Train RL agents, host models |
| **Lambda** | Serverless compute | Lightweight agent tasks |
| **ECS/EKS** | Container orchestration | Deploy multi-agent systems |
| **Step Functions** | Workflow orchestration | Agent task coordination |
| **Bedrock** | Foundation models | LLM-based agents |
| **S3** | Object storage | Training data, model artifacts |

### Deploying Agent with AWS Lambda

```python
# lambda_function.py
import json
import boto3
from typing import Dict, Any

# Initialize AWS clients
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    AWS Lambda handler for agent execution
    """
    try:
        # Extract request
        body = json.loads(event['body'])
        task = body.get('task')
        
        # Agent logic
        response = execute_agent_task(task)
        
        # Store result in S3
        s3.put_object(
            Bucket='agent-results',
            Key=f"results/{context.request_id}.json",
            Body=json.dumps(response)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {'Content-Type': 'application/json'}
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def execute_agent_task(task: str) -> Dict[str, Any]:
    """Execute agent task using Bedrock"""
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps({
            'prompt': f"\n\nHuman: {task}\n\nAssistant:",
            'max_tokens': 1000
        })
    )
    
    result = json.loads(response['body'].read())
    return {'result': result['completion']}
```

### AWS SAM Template

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  AgentFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: agent/
      Handler: lambda_function.lambda_handler
      Runtime: python3.11
      Timeout: 300
      MemorySize: 1024
      Environment:
        Variables:
          RESULTS_BUCKET: !Ref ResultsBucket
      Policies:
        - S3CrudPolicy:
            BucketName: !Ref ResultsBucket
        - Statement:
          - Effect: Allow
            Action:
              - bedrock:InvokeModel
            Resource: '*'
      Events:
        AgentAPI:
          Type: Api
          Properties:
            Path: /agent/execute
            Method: post

  ResultsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: agent-results-bucket

Outputs:
  AgentApiUrl:
    Description: "API Gateway endpoint URL"
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/agent/execute"
```

### ECS Deployment with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Expose port
EXPOSE 8000

# Run agent server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# ecs-task-definition.json
{
  "family": "agent-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "agent-container",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/agent:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MODEL_PATH", "value": "s3://models/agent-model.pth"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/agent-task",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## 🔵 Google Cloud Platform (GCP)

### GCP Services for Agents

| Service | Purpose | Use Case |
|---------|---------|----------|
| **Vertex AI** | ML platform | Train/deploy agents |
| **Cloud Run** | Serverless containers | Deploy agent APIs |
| **GKE** | Kubernetes | Multi-agent orchestration |
| **Cloud Functions** | Serverless compute | Event-driven agents |
| **Cloud Storage** | Object storage | Data & models |
| **Pub/Sub** | Messaging | Agent communication |

### Cloud Run Deployment

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: agent-service
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: '10'
        autoscaling.knative.dev/minScale: '1'
    spec:
      containers:
      - image: gcr.io/project-id/agent:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_BUCKET
          value: gs://agent-models
        resources:
          limits:
            memory: 2Gi
            cpu: '2'
```

### Vertex AI Training Job

```python
from google.cloud import aiplatform

def train_agent_on_vertex(project_id: str, location: str):
    """Train agent using Vertex AI"""
    
    aiplatform.init(project=project_id, location=location)
    
    # Create custom training job
    job = aiplatform.CustomTrainingJob(
        display_name="agent-training",
        script_path="train.py",
        container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
        requirements=["stable-baselines3", "gymnasium"],
        model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13:latest"
    )
    
    # Run training
    model = job.run(
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        args=["--episodes", "10000", "--lr", "0.001"]
    )
    
    # Deploy model
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1
    )
    
    return endpoint
```

## 🔷 Microsoft Azure

### Azure Services for Agents

| Service | Purpose | Use Case |
|---------|---------|----------|
| **Azure ML** | ML platform | Train/deploy agents |
| **Container Instances** | Serverless containers | Quick deployment |
| **AKS** | Kubernetes | Production orchestration |
| **Functions** | Serverless compute | Event-driven tasks |
| **Cognitive Services** | AI APIs | Pre-built AI capabilities |
| **Cosmos DB** | Database | Agent state storage |

### Azure Functions Deployment

```python
# function_app.py
import azure.functions as func
import json
import logging

app = func.FunctionApp()

@app.route(route="agent/execute", methods=["POST"])
async def agent_execute(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function for agent execution"""
    
    logging.info('Agent execution request received')
    
    try:
        # Parse request
        req_body = req.get_json()
        task = req_body.get('task')
        
        # Execute agent
        result = await execute_agent(task)
        
        return func.HttpResponse(
            json.dumps({'result': result}),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500
        )

async def execute_agent(task: str):
    """Agent execution logic"""
    # Implementation here
    return f"Executed: {task}"
```

### Azure ML Pipeline

```python
from azureml.core import Workspace, Experiment
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

def create_training_pipeline(workspace: Workspace):
    """Create Azure ML pipeline for agent training"""
    
    # Define steps
    data_prep_step = PythonScriptStep(
        name="Data Preparation",
        script_name="prepare_data.py",
        compute_target="cpu-cluster",
        source_directory="./scripts"
    )
    
    training_step = PythonScriptStep(
        name="Agent Training",
        script_name="train_agent.py",
        compute_target="gpu-cluster",
        source_directory="./scripts"
    )
    
    evaluation_step = PythonScriptStep(
        name="Model Evaluation",
        script_name="evaluate.py",
        compute_target="cpu-cluster",
        source_directory="./scripts"
    )
    
    # Create pipeline
    pipeline = Pipeline(
        workspace=workspace,
        steps=[data_prep_step, training_step, evaluation_step]
    )
    
    return pipeline
```

## 🐳 Docker & Kubernetes

### Multi-Container Agent System

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-api:
    build: ./agent-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://db:5432/agents
    depends_on:
      - redis
      - postgres
  
  agent-worker:
    build: ./agent-worker
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 3
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agents
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
    spec:
      containers:
      - name: agent
        image: agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /models/agent.pth
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📊 Platform Comparison

| Platform | Strengths | Best For | Pricing |
|----------|-----------|----------|---------|
| **AWS** | Mature, comprehensive | Enterprise, scale | Pay-as-you-go |
| **GCP** | ML/AI focus, ease of use | AI workloads, startups | Competitive |
| **Azure** | Enterprise integration | Microsoft shops | Enterprise-friendly |

## 🛠️ Deployment Tools

| Tool | Purpose | Repository |
|------|---------|-----------|
| [Terraform](https://github.com/hashicorp/terraform) | Infrastructure as Code | [GitHub](https://github.com/hashicorp/terraform) |
| [Pulumi](https://github.com/pulumi/pulumi) | Modern IaC | [GitHub](https://github.com/pulumi/pulumi) |
| [Helm](https://github.com/helm/helm) | Kubernetes packages | [GitHub](https://github.com/helm/helm) |
| [Skaffold](https://github.com/GoogleContainerTools/skaffold) | K8s development | [GitHub](https://github.com/GoogleContainerTools/skaffold) |

## 📚 Learning Resources

### Documentation
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [GCP Best Practices](https://cloud.google.com/architecture/best-practices)
- [Azure Architecture Center](https://docs.microsoft.com/en-us/azure/architecture/)

### Courses
- **AWS Certified Solutions Architect**
- **Google Cloud Professional ML Engineer**
- **Azure AI Engineer Associate**

## 🔗 Related Topics

- [APIs & Pipelines](./APIs-Pipelines.md)
- [Automation & Scheduling](./Automation-Scheduling.md)
- [System Design](../Supporting-Skills/System-Design.md)
- [Orchestration Frameworks](../Frameworks-Tools/Orchestration-Frameworks.md)

---

*This guide covers cloud deployment strategies for production agent systems. For API integration patterns, see the APIs & Pipelines guide.*