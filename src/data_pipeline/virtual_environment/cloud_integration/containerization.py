"""
Containerization for PBF-LB/M Virtual Environment

This module provides containerization capabilities including Docker management,
Kubernetes orchestration, and container-based deployment for PBF-LB/M virtual
testing and simulation environments.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import subprocess
import threading
import yaml
import warnings

# Optional containerization imports
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    warnings.warn("docker not available. Docker functionality will be limited.")

try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    warnings.warn("kubernetes not available. Kubernetes functionality will be limited.")

logger = logging.getLogger(__name__)


class ContainerStatus(Enum):
    """Container status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"


class ContainerType(Enum):
    """Container type enumeration."""
    SIMULATION = "simulation"
    DATA_PROCESSING = "data_processing"
    WEB_SERVICE = "web_service"
    DATABASE = "database"
    MONITORING = "monitoring"


@dataclass
class ContainerConfig:
    """Container configuration."""
    
    container_id: str
    name: str
    image: str
    container_type: ContainerType
    created_at: datetime
    updated_at: datetime
    
    # Container specifications
    cpu_limit: str = "1"
    memory_limit: str = "1Gi"
    storage_limit: str = "10Gi"
    
    # Port mappings
    port_mappings: Dict[int, int] = None
    
    # Environment variables
    environment_variables: Dict[str, str] = None
    
    # Volume mounts
    volume_mounts: Dict[str, str] = None
    
    # Container state
    status: ContainerStatus = ContainerStatus.CREATED


@dataclass
class PodConfig:
    """Kubernetes pod configuration."""
    
    pod_id: str
    name: str
    namespace: str
    
    # Pod specifications
    containers: List[ContainerConfig]
    created_at: datetime
    updated_at: datetime
    
    # Pod specifications with defaults
    replicas: int = 1
    
    # Resource requirements
    cpu_request: str = "1"
    memory_request: str = "1Gi"
    cpu_limit: str = "2"
    memory_limit: str = "2Gi"
    
    # Pod state
    status: str = "pending"


class ContainerManager:
    """
    Container manager for PBF-LB/M virtual environment.
    
    This class provides container management capabilities including Docker
    container management, container orchestration, and container lifecycle
    management for PBF-LB/M virtual testing and simulation environments.
    """
    
    def __init__(self):
        """Initialize the container manager."""
        self.docker_manager = DockerManager()
        self.kubernetes_manager = KubernetesManager()
        self.containers = {}
        self.pods = {}
        
        logger.info("Container Manager initialized")
    
    async def create_container(
        self,
        name: str,
        image: str,
        container_type: ContainerType,
        cpu_limit: str = "1",
        memory_limit: str = "1Gi",
        port_mappings: Dict[int, int] = None,
        environment_variables: Dict[str, str] = None,
        volume_mounts: Dict[str, str] = None
    ) -> str:
        """
        Create a container.
        
        Args:
            name: Container name
            image: Container image
            container_type: Type of container
            cpu_limit: CPU limit
            memory_limit: Memory limit
            port_mappings: Port mappings
            environment_variables: Environment variables
            volume_mounts: Volume mounts
            
        Returns:
            str: Container ID
        """
        try:
            container_id = await self.docker_manager.create_container(
                name, image, cpu_limit, memory_limit,
                port_mappings, environment_variables, volume_mounts
            )
            
            # Store container configuration
            config = ContainerConfig(
                container_id=container_id,
                name=name,
                image=image,
                container_type=container_type,
                cpu_limit=cpu_limit,
                memory_limit=memory_limit,
                port_mappings=port_mappings or {},
                environment_variables=environment_variables or {},
                volume_mounts=volume_mounts or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.containers[container_id] = config
            
            logger.info(f"Container created: {container_id}")
            return container_id
            
        except Exception as e:
            logger.error(f"Error creating container: {e}")
            return ""
    
    async def start_container(self, container_id: str) -> bool:
        """Start a container."""
        try:
            if container_id not in self.containers:
                raise ValueError(f"Container not found: {container_id}")
            
            success = await self.docker_manager.start_container(container_id)
            
            if success:
                self.containers[container_id].status = ContainerStatus.RUNNING
                self.containers[container_id].updated_at = datetime.now()
                
                logger.info(f"Container started: {container_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting container: {e}")
            return False
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop a container."""
        try:
            if container_id not in self.containers:
                raise ValueError(f"Container not found: {container_id}")
            
            success = await self.docker_manager.stop_container(container_id)
            
            if success:
                self.containers[container_id].status = ContainerStatus.EXITED
                self.containers[container_id].updated_at = datetime.now()
                
                logger.info(f"Container stopped: {container_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping container: {e}")
            return False
    
    async def delete_container(self, container_id: str) -> bool:
        """Delete a container."""
        try:
            if container_id not in self.containers:
                raise ValueError(f"Container not found: {container_id}")
            
            success = await self.docker_manager.delete_container(container_id)
            
            if success:
                del self.containers[container_id]
                logger.info(f"Container deleted: {container_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting container: {e}")
            return False
    
    async def get_container_status(self, container_id: str) -> Dict[str, Any]:
        """Get container status."""
        try:
            if container_id not in self.containers:
                raise ValueError(f"Container not found: {container_id}")
            
            config = self.containers[container_id]
            status = await self.docker_manager.get_container_status(container_id)
            
            return {
                'container_id': container_id,
                'name': config.name,
                'image': config.image,
                'container_type': config.container_type.value,
                'status': status.get('status', 'unknown'),
                'cpu_limit': config.cpu_limit,
                'memory_limit': config.memory_limit,
                'port_mappings': config.port_mappings,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting container status: {e}")
            return {}
    
    async def list_containers(self) -> List[Dict[str, Any]]:
        """List all containers."""
        try:
            containers = []
            
            for container_id in self.containers:
                status = await self.get_container_status(container_id)
                containers.append(status)
            
            return containers
            
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []
    
    async def create_pod(
        self,
        name: str,
        namespace: str,
        containers: List[ContainerConfig],
        replicas: int = 1,
        cpu_request: str = "1",
        memory_request: str = "1Gi",
        cpu_limit: str = "2",
        memory_limit: str = "2Gi"
    ) -> str:
        """
        Create a Kubernetes pod.
        
        Args:
            name: Pod name
            namespace: Kubernetes namespace
            containers: List of container configurations
            replicas: Number of replicas
            cpu_request: CPU request
            memory_request: Memory request
            cpu_limit: CPU limit
            memory_limit: Memory limit
            
        Returns:
            str: Pod ID
        """
        try:
            pod_id = await self.kubernetes_manager.create_pod(
                name, namespace, containers, replicas,
                cpu_request, memory_request, cpu_limit, memory_limit
            )
            
            # Store pod configuration
            pod_config = PodConfig(
                pod_id=pod_id,
                name=name,
                namespace=namespace,
                containers=containers,
                replicas=replicas,
                cpu_request=cpu_request,
                memory_request=memory_request,
                cpu_limit=cpu_limit,
                memory_limit=memory_limit,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.pods[pod_id] = pod_config
            
            logger.info(f"Pod created: {pod_id}")
            return pod_id
            
        except Exception as e:
            logger.error(f"Error creating pod: {e}")
            return ""
    
    async def delete_pod(self, pod_id: str) -> bool:
        """Delete a Kubernetes pod."""
        try:
            if pod_id not in self.pods:
                raise ValueError(f"Pod not found: {pod_id}")
            
            pod_config = self.pods[pod_id]
            success = await self.kubernetes_manager.delete_pod(
                pod_config.name, pod_config.namespace
            )
            
            if success:
                del self.pods[pod_id]
                logger.info(f"Pod deleted: {pod_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting pod: {e}")
            return False
    
    async def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """Get pod status."""
        try:
            if pod_id not in self.pods:
                raise ValueError(f"Pod not found: {pod_id}")
            
            pod_config = self.pods[pod_id]
            status = await self.kubernetes_manager.get_pod_status(
                pod_config.name, pod_config.namespace
            )
            
            return {
                'pod_id': pod_id,
                'name': pod_config.name,
                'namespace': pod_config.namespace,
                'status': status.get('status', 'unknown'),
                'containers': len(pod_config.containers),
                'replicas': pod_config.replicas,
                'cpu_request': pod_config.cpu_request,
                'memory_request': pod_config.memory_request,
                'cpu_limit': pod_config.cpu_limit,
                'memory_limit': pod_config.memory_limit,
                'created_at': pod_config.created_at.isoformat(),
                'updated_at': pod_config.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pod status: {e}")
            return {}


class DockerManager:
    """
    Docker manager for container management.
    
    This class provides Docker-specific container management capabilities
    including container creation, lifecycle management, and Docker operations.
    """
    
    def __init__(self):
        """Initialize the Docker manager."""
        if not DOCKER_AVAILABLE:
            logger.warning("Docker not available. Docker functionality will be limited.")
            self.docker_client = None
            return
            
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker Manager initialized")
        except Exception as e:
            logger.error(f"Error initializing Docker client: {e}")
            self.docker_client = None
    
    async def create_container(
        self,
        name: str,
        image: str,
        cpu_limit: str = "1",
        memory_limit: str = "1Gi",
        port_mappings: Dict[int, int] = None,
        environment_variables: Dict[str, str] = None,
        volume_mounts: Dict[str, str] = None
    ) -> str:
        """Create a Docker container."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            # Prepare container configuration
            container_config = {
                'image': image,
                'name': name,
                'detach': True,
                'environment': environment_variables or {},
                'ports': port_mappings or {},
                'mem_limit': memory_limit,
                'cpu_quota': int(float(cpu_limit) * 100000),  # Convert to microseconds
                'cpu_period': 100000
            }
            
            # Add volume mounts
            if volume_mounts:
                container_config['volumes'] = volume_mounts
            
            # Create container
            container = self.docker_client.containers.create(**container_config)
            
            logger.info(f"Docker container created: {container.id}")
            return container.id
            
        except Exception as e:
            logger.error(f"Error creating Docker container: {e}")
            return ""
    
    async def start_container(self, container_id: str) -> bool:
        """Start a Docker container."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            container = self.docker_client.containers.get(container_id)
            container.start()
            
            logger.info(f"Docker container started: {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Docker container: {e}")
            return False
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop a Docker container."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            container = self.docker_client.containers.get(container_id)
            container.stop()
            
            logger.info(f"Docker container stopped: {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Docker container: {e}")
            return False
    
    async def delete_container(self, container_id: str) -> bool:
        """Delete a Docker container."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            container = self.docker_client.containers.get(container_id)
            container.remove(force=True)
            
            logger.info(f"Docker container deleted: {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Docker container: {e}")
            return False
    
    async def get_container_status(self, container_id: str) -> Dict[str, Any]:
        """Get Docker container status."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            container = self.docker_client.containers.get(container_id)
            
            return {
                'status': container.status,
                'state': container.attrs['State']['Status'],
                'created': container.attrs['Created'],
                'ports': container.attrs['NetworkSettings']['Ports'],
                'image': container.attrs['Config']['Image']
            }
            
        except Exception as e:
            logger.error(f"Error getting Docker container status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def list_containers(self) -> List[Dict[str, Any]]:
        """List Docker containers."""
        try:
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            containers = self.docker_client.containers.list(all=True)
            
            container_list = []
            for container in containers:
                container_list.append({
                    'id': container.id,
                    'name': container.name,
                    'image': container.image.tags[0] if container.image.tags else container.image.id,
                    'status': container.status,
                    'created': container.attrs['Created']
                })
            
            return container_list
            
        except Exception as e:
            logger.error(f"Error listing Docker containers: {e}")
            return []


class KubernetesManager:
    """
    Kubernetes manager for container orchestration.
    
    This class provides Kubernetes-specific container orchestration capabilities
    including pod management, deployment management, and Kubernetes operations.
    """
    
    def __init__(self):
        """Initialize the Kubernetes manager."""
        if not KUBERNETES_AVAILABLE:
            logger.warning("Kubernetes not available. Kubernetes functionality will be limited.")
            self.v1 = None
            self.apps_v1 = None
            return
            
        try:
            # Load Kubernetes configuration
            config.load_incluster_config()
            
            # Initialize Kubernetes clients
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            
            logger.info("Kubernetes Manager initialized")
        except Exception as e:
            logger.error(f"Error initializing Kubernetes client: {e}")
            self.v1 = None
            self.apps_v1 = None
    
    async def create_pod(
        self,
        name: str,
        namespace: str,
        containers: List[ContainerConfig],
        replicas: int = 1,
        cpu_request: str = "1",
        memory_request: str = "1Gi",
        cpu_limit: str = "2",
        memory_limit: str = "2Gi"
    ) -> str:
        """Create a Kubernetes pod."""
        try:
            if not self.v1:
                raise RuntimeError("Kubernetes client not available")
            
            # Create pod specification
            pod_spec = client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name=container.name,
                        image=container.image,
                        resources=client.V1ResourceRequirements(
                            requests={
                                'cpu': cpu_request,
                                'memory': memory_request
                            },
                            limits={
                                'cpu': cpu_limit,
                                'memory': memory_limit
                            }
                        ),
                        env=[
                            client.V1EnvVar(name=k, value=v)
                            for k, v in container.environment_variables.items()
                        ],
                        ports=[
                            client.V1ContainerPort(container_port=port)
                            for port in container.port_mappings.keys()
                        ]
                    )
                    for container in containers
                ],
                restart_policy='Always'
            )
            
            # Create pod metadata
            pod_metadata = client.V1ObjectMeta(
                name=name,
                namespace=namespace,
                labels={
                    'app': name,
                    'environment': 'pbf-lbm'
                }
            )
            
            # Create pod
            pod = client.V1Pod(
                metadata=pod_metadata,
                spec=pod_spec
            )
            
            # Create pod in Kubernetes
            created_pod = self.v1.create_namespaced_pod(
                namespace=namespace,
                body=pod
            )
            
            logger.info(f"Kubernetes pod created: {created_pod.metadata.name}")
            return created_pod.metadata.name
            
        except Exception as e:
            logger.error(f"Error creating Kubernetes pod: {e}")
            return ""
    
    async def delete_pod(self, name: str, namespace: str) -> bool:
        """Delete a Kubernetes pod."""
        try:
            if not self.v1:
                raise RuntimeError("Kubernetes client not available")
            
            # Delete pod
            self.v1.delete_namespaced_pod(
                name=name,
                namespace=namespace
            )
            
            logger.info(f"Kubernetes pod deleted: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Kubernetes pod: {e}")
            return False
    
    async def get_pod_status(self, name: str, namespace: str) -> Dict[str, Any]:
        """Get Kubernetes pod status."""
        try:
            if not self.v1:
                raise RuntimeError("Kubernetes client not available")
            
            # Get pod
            pod = self.v1.read_namespaced_pod(
                name=name,
                namespace=namespace
            )
            
            return {
                'status': pod.status.phase,
                'created': pod.metadata.creation_timestamp.isoformat(),
                'containers': len(pod.spec.containers),
                'node_name': pod.spec.node_name,
                'pod_ip': pod.status.pod_ip
            }
            
        except Exception as e:
            logger.error(f"Error getting Kubernetes pod status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def list_pods(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """List Kubernetes pods."""
        try:
            if not self.v1:
                raise RuntimeError("Kubernetes client not available")
            
            # List pods
            pods = self.v1.list_namespaced_pod(namespace=namespace)
            
            pod_list = []
            for pod in pods.items:
                pod_list.append({
                    'name': pod.metadata.name,
                    'namespace': pod.metadata.namespace,
                    'status': pod.status.phase,
                    'created': pod.metadata.creation_timestamp.isoformat(),
                    'containers': len(pod.spec.containers),
                    'node_name': pod.spec.node_name,
                    'pod_ip': pod.status.pod_ip
                })
            
            return pod_list
            
        except Exception as e:
            logger.error(f"Error listing Kubernetes pods: {e}")
            return []
    
    async def create_deployment(
        self,
        name: str,
        namespace: str,
        image: str,
        replicas: int = 1,
        cpu_request: str = "1",
        memory_request: str = "1Gi",
        cpu_limit: str = "2",
        memory_limit: str = "2Gi"
    ) -> str:
        """Create a Kubernetes deployment."""
        try:
            if not self.apps_v1:
                raise RuntimeError("Kubernetes apps client not available")
            
            # Create container specification
            container = client.V1Container(
                name=name,
                image=image,
                resources=client.V1ResourceRequirements(
                    requests={
                        'cpu': cpu_request,
                        'memory': memory_request
                    },
                    limits={
                        'cpu': cpu_limit,
                        'memory': memory_limit
                    }
                )
            )
            
            # Create pod template
            pod_template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={'app': name}
                ),
                spec=client.V1PodSpec(
                    containers=[container]
                )
            )
            
            # Create deployment specification
            deployment_spec = client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={'app': name}
                ),
                template=pod_template
            )
            
            # Create deployment metadata
            deployment_metadata = client.V1ObjectMeta(
                name=name,
                namespace=namespace,
                labels={
                    'app': name,
                    'environment': 'pbf-lbm'
                }
            )
            
            # Create deployment
            deployment = client.V1Deployment(
                metadata=deployment_metadata,
                spec=deployment_spec
            )
            
            # Create deployment in Kubernetes
            created_deployment = self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Kubernetes deployment created: {created_deployment.metadata.name}")
            return created_deployment.metadata.name
            
        except Exception as e:
            logger.error(f"Error creating Kubernetes deployment: {e}")
            return ""
    
    async def delete_deployment(self, name: str, namespace: str) -> bool:
        """Delete a Kubernetes deployment."""
        try:
            if not self.apps_v1:
                raise RuntimeError("Kubernetes apps client not available")
            
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            
            logger.info(f"Kubernetes deployment deleted: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Kubernetes deployment: {e}")
            return False
