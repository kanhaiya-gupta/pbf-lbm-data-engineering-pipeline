"""
Cloud Integration Example for PBF-LB/M Virtual Environment

This example demonstrates how to use the cloud integration components
including cloud providers, distributed computing, containerization,
and serverless computing for PBF-LB/M virtual testing and simulation.
"""

import asyncio
import logging
from typing import Dict, Any, List
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.data_pipeline.virtual_environment.cloud_integration.cloud_providers import (
    CloudProviderManager, CloudProviderType, CloudInstanceType
)
from src.data_pipeline.virtual_environment.cloud_integration.distributed_computing import (
    DistributedComputingManager, TaskType, TaskStatus
)
from src.data_pipeline.virtual_environment.cloud_integration.containerization import (
    ContainerManager, ContainerType, ContainerStatus
)
from src.data_pipeline.virtual_environment.cloud_integration.serverless import (
    ServerlessManager, FunctionType, EventType, FunctionStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def cloud_provider_example():
    """Demonstrate cloud provider management."""
    logger.info("=== Cloud Provider Example ===")
    
    # Initialize cloud provider manager
    cloud_manager = CloudProviderManager()
    
    # Create cloud provider
    provider_id = await cloud_manager.create_provider(
        name="AWS Provider",
        provider_type=CloudProviderType.AWS,
        region="us-west-2",
        credentials={
            "access_key": "your_access_key",
            "secret_key": "your_secret_key"
        }
    )
    
    if provider_id:
        logger.info(f"Cloud provider created: {provider_id}")
        
        # Create cloud instance
        instance_id = await cloud_manager.create_instance(
            provider_id=provider_id,
            instance_type=CloudInstanceType.COMPUTE_OPTIMIZED,
            name="PBF-LBM Test Instance",
            cpu_cores=4,
            memory_gb=8,
            storage_gb=100
        )
        
        if instance_id:
            logger.info(f"Cloud instance created: {instance_id}")
            
            # Get instance status
            status = await cloud_manager.get_instance_status(instance_id)
            logger.info(f"Instance status: {status}")
            
            # Start instance
            success = await cloud_manager.start_instance(instance_id)
            if success:
                logger.info("Instance started successfully")
                
                # Wait a bit
                await asyncio.sleep(2)
                
                # Stop instance
                success = await cloud_manager.stop_instance(instance_id)
                if success:
                    logger.info("Instance stopped successfully")
                
                # Delete instance
                success = await cloud_manager.delete_instance(instance_id)
                if success:
                    logger.info("Instance deleted successfully")
        
        # Delete provider
        success = await cloud_manager.delete_provider(provider_id)
        if success:
            logger.info("Cloud provider deleted successfully")


async def distributed_computing_example():
    """Demonstrate distributed computing."""
    logger.info("=== Distributed Computing Example ===")
    
    # Initialize distributed computing manager
    dist_manager = DistributedComputingManager()
    
    # Create compute cluster
    cluster_id = await dist_manager.create_cluster(
        name="PBF-LBM Cluster",
        node_count=3,
        cpu_cores_per_node=4,
        memory_gb_per_node=8,
        storage_gb_per_node=100
    )
    
    if cluster_id:
        logger.info(f"Compute cluster created: {cluster_id}")
        
        # Create task
        task_id = await dist_manager.create_task(
            name="PBF-LBM Simulation",
            task_type=TaskType.SIMULATION,
            code="""
def simulate_pbf_lbm(parameters):
    # Simulate PBF-LB/M process
    result = {
        'temperature': parameters.get('temperature', 1000),
        'pressure': parameters.get('pressure', 1.0),
        'quality': 'good'
    }
    return result
            """,
            parameters={
                'temperature': 1200,
                'pressure': 1.2,
                'laser_power': 250
            }
        )
        
        if task_id:
            logger.info(f"Task created: {task_id}")
            
            # Submit task to cluster
            execution_id = await dist_manager.submit_task(cluster_id, task_id)
            if execution_id:
                logger.info(f"Task submitted: {execution_id}")
                
                # Wait for task completion
                await asyncio.sleep(2)
                
                # Get task status
                status = await dist_manager.get_task_status(task_id)
                logger.info(f"Task status: {status}")
                
                # Get execution result
                result = await dist_manager.get_execution_result(execution_id)
                logger.info(f"Execution result: {result}")
        
        # Delete cluster
        success = await dist_manager.delete_cluster(cluster_id)
        if success:
            logger.info("Compute cluster deleted successfully")


async def containerization_example():
    """Demonstrate containerization."""
    logger.info("=== Containerization Example ===")
    
    # Initialize container manager
    container_manager = ContainerManager()
    
    # Create container
    container_id = await container_manager.create_container(
        name="pbf-lbm-simulator",
        image="pbf-lbm/simulator:latest",
        container_type=ContainerType.SIMULATION,
        cpu_limit="2",
        memory_limit="4Gi",
        port_mappings={8080: 8080},
        environment_variables={
            "SIMULATION_MODE": "thermal",
            "LOG_LEVEL": "INFO"
        }
    )
    
    if container_id:
        logger.info(f"Container created: {container_id}")
        
        # Start container
        success = await container_manager.start_container(container_id)
        if success:
            logger.info("Container started successfully")
            
            # Get container status
            status = await container_manager.get_container_status(container_id)
            logger.info(f"Container status: {status}")
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Stop container
            success = await container_manager.stop_container(container_id)
            if success:
                logger.info("Container stopped successfully")
            
            # Delete container
            success = await container_manager.delete_container(container_id)
            if success:
                logger.info("Container deleted successfully")


async def serverless_example():
    """Demonstrate serverless computing."""
    logger.info("=== Serverless Computing Example ===")
    
    # Initialize serverless manager
    serverless_manager = ServerlessManager()
    
    # Create function
    function_id = await serverless_manager.create_function(
        name="pbf-lbm-analyzer",
        function_type=FunctionType.ANALYSIS,
        code="""
def main(payload):
    # Analyze PBF-LB/M data
    data = payload.get('data', {})
    
    # Perform analysis
    analysis_result = {
        'quality_score': 0.95,
        'defects_detected': 2,
        'recommendations': ['Optimize laser power', 'Adjust scan speed']
    }
    
    return analysis_result
        """,
        runtime="python3.9",
        memory_size=256,
        timeout=300
    )
    
    if function_id:
        logger.info(f"Function created: {function_id}")
        
        # Deploy function
        success = await serverless_manager.deploy_function(function_id)
        if success:
            logger.info("Function deployed successfully")
            
            # Create event trigger
            trigger_id = await serverless_manager.create_event_trigger(
                function_id=function_id,
                event_type=EventType.HTTP_REQUEST,
                source="api_gateway"
            )
            
            if trigger_id:
                logger.info(f"Event trigger created: {trigger_id}")
                
                # Emit event
                event_id = await serverless_manager.emit_event(
                    event_type=EventType.HTTP_REQUEST,
                    source="api_gateway",
                    payload={
                        'data': {
                            'temperature': 1200,
                            'pressure': 1.2,
                            'laser_power': 250
                        }
                    }
                )
                
                if event_id:
                    logger.info(f"Event emitted: {event_id}")
                    
                    # Wait for function execution
                    await asyncio.sleep(2)
                    
                    # Get function status
                    status = await serverless_manager.get_function_status(function_id)
                    logger.info(f"Function status: {status}")
                    
                    # List executions
                    executions = await serverless_manager.list_executions()
                    logger.info(f"Executions: {len(executions)}")
            
            # Delete function
            success = await serverless_manager.delete_function(function_id)
            if success:
                logger.info("Function deleted successfully")


async def integrated_workflow_example():
    """Demonstrate integrated cloud workflow."""
    logger.info("=== Integrated Cloud Workflow Example ===")
    
    # Initialize managers
    cloud_manager = CloudProviderManager()
    container_manager = ContainerManager()
    serverless_manager = ServerlessManager()
    
    try:
        # 1. Create cloud provider
        provider_id = await cloud_manager.create_provider(
            name="Integrated Provider",
            provider_type=CloudProviderType.AZURE,
            region="east-us",
            credentials={"subscription_id": "your_subscription_id"}
        )
        
        if provider_id:
            logger.info(f"Cloud provider created: {provider_id}")
            
            # 2. Create cloud instance
            instance_id = await cloud_manager.create_instance(
                provider_id=provider_id,
                instance_type=CloudInstanceType.GENERAL_PURPOSE,
                name="Integrated Instance",
                cpu_cores=2,
                memory_gb=4,
                storage_gb=50
            )
            
            if instance_id:
                logger.info(f"Cloud instance created: {instance_id}")
                
                # 3. Create container on the instance
                container_id = await container_manager.create_container(
                    name="integrated-simulator",
                    image="pbf-lbm/integrated:latest",
                    container_type=ContainerType.SIMULATION,
                    cpu_limit="1",
                    memory_limit="2Gi"
                )
                
                if container_id:
                    logger.info(f"Container created: {container_id}")
                    
                    # 4. Create serverless function for monitoring
                    function_id = await serverless_manager.create_function(
                        name="monitor-simulation",
                        function_type=FunctionType.MONITORING,
                        code="""
def main(payload):
    # Monitor simulation progress
    progress = payload.get('progress', 0)
    
    if progress >= 100:
        return {'status': 'completed', 'message': 'Simulation finished'}
    else:
        return {'status': 'running', 'progress': progress}
                        """,
                        runtime="python3.9",
                        memory_size=128,
                        timeout=60
                    )
                    
                    if function_id:
                        logger.info(f"Monitoring function created: {function_id}")
                        
                        # 5. Deploy and test the integrated workflow
                        await serverless_manager.deploy_function(function_id)
                        
                        # Emit monitoring event
                        await serverless_manager.emit_event(
                            event_type=EventType.TIMER,
                            source="simulation_monitor",
                            payload={'progress': 75}
                        )
                        
                        # Wait for processing
                        await asyncio.sleep(1)
                        
                        # Get results
                        executions = await serverless_manager.list_executions()
                        logger.info(f"Monitoring executions: {len(executions)}")
                        
                        # Cleanup
                        await serverless_manager.delete_function(function_id)
                        await container_manager.delete_container(container_id)
                        await cloud_manager.delete_instance(instance_id)
                        await cloud_manager.delete_provider(provider_id)
                        
                        logger.info("Integrated workflow completed successfully")
    
    except Exception as e:
        logger.error(f"Error in integrated workflow: {e}")


async def main():
    """Main function to run all examples."""
    logger.info("Starting Cloud Integration Examples")
    
    try:
        # Run individual examples
        await cloud_provider_example()
        await asyncio.sleep(1)
        
        await distributed_computing_example()
        await asyncio.sleep(1)
        
        await containerization_example()
        await asyncio.sleep(1)
        
        await serverless_example()
        await asyncio.sleep(1)
        
        # Run integrated workflow example
        await integrated_workflow_example()
        
        logger.info("All Cloud Integration Examples completed successfully")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
