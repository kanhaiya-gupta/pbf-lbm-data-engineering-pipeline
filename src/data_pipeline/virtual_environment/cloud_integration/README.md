# Cloud Integration for PBF-LB/M Virtual Environment

This module provides comprehensive cloud integration capabilities for the PBF-LB/M virtual environment, including cloud provider management, distributed computing, containerization, and serverless computing.

## Features

- **Cloud Provider Management**: Support for AWS, Azure, and GCP
- **Distributed Computing**: Task distribution and parallel processing
- **Containerization**: Docker and Kubernetes orchestration
- **Serverless Computing**: Function deployment and event-driven execution

## Quick Start

```python
from src.data_pipeline.virtual_environment.cloud_integration import (
    CloudProviderManager,
    DistributedComputingManager,
    ContainerManager,
    ServerlessManager
)

# Initialize managers
cloud_manager = CloudProviderManager()
dist_manager = DistributedComputingManager()
container_manager = ContainerManager()
serverless_manager = ServerlessManager()

# Create cloud provider
provider_id = await cloud_manager.create_provider(
    name="AWS Provider",
    provider_type=CloudProviderType.AWS,
    region="us-west-2"
)

# Create distributed computing cluster
cluster_id = await dist_manager.create_cluster(
    name="PBF-LBM Cluster",
    node_count=3
)

# Create container
container_id = await container_manager.create_container(
    name="pbf-lbm-simulator",
    image="pbf-lbm/simulator:latest"
)

# Create serverless function
function_id = await serverless_manager.create_function(
    name="pbf-lbm-analyzer",
    function_type=FunctionType.ANALYSIS,
    code="def main(payload): return {'result': 'success'}"
)
```

## Components

### Cloud Providers
- **AWS**: EC2, ECS, Lambda integration
- **Azure**: Virtual Machines, Container Instances, Functions
- **GCP**: Compute Engine, Cloud Run, Cloud Functions

### Distributed Computing
- **Task Distribution**: Automatic task distribution across nodes
- **Parallel Processing**: Concurrent execution of multiple tasks
- **Load Balancing**: Intelligent load distribution
- **Fault Tolerance**: Automatic failover and recovery

### Containerization
- **Docker**: Container creation and management
- **Kubernetes**: Pod orchestration and scaling
- **Container Registry**: Image management and distribution
- **Service Mesh**: Inter-service communication

### Serverless Computing
- **Function Deployment**: Serverless function management
- **Event-Driven Execution**: Trigger-based function invocation
- **Auto-scaling**: Automatic scaling based on demand
- **Cost Optimization**: Pay-per-use pricing model

## Examples

See `examples/cloud_integration_example.py` for comprehensive usage examples.

## Dependencies

- `boto3` - AWS SDK
- `azure-mgmt-compute` - Azure management
- `google-cloud-compute` - GCP management
- `docker` - Docker client
- `kubernetes` - Kubernetes client
- `asyncio` - Asynchronous programming

## Configuration

Configure cloud credentials and settings through environment variables or configuration files:

```bash
# AWS
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Azure
export AZURE_SUBSCRIPTION_ID=your_subscription_id
export AZURE_CLIENT_ID=your_client_id
export AZURE_CLIENT_SECRET=your_client_secret
export AZURE_TENANT_ID=your_tenant_id

# GCP
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your_project_id
```



