"""
Cloud Providers for PBF-LB/M Virtual Environment

This module provides cloud provider integration capabilities including AWS, Azure,
and GCP integration for PBF-LB/M virtual testing and simulation environments.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import warnings

# Optional cloud provider imports - only import when needed
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    warnings.warn("boto3 not available. AWS functionality will be limited.")

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.resource import ResourceManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    warnings.warn("Azure SDK not available. Azure functionality will be limited.")

try:
    from google.cloud import compute_v1
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    warnings.warn("Google Cloud SDK not available. GCP functionality will be limited.")

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Cloud provider enumeration."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LOCAL = "local"


class InstanceType(Enum):
    """Instance type enumeration."""
    COMPUTE_OPTIMIZED = "compute_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    STORAGE_OPTIMIZED = "storage_optimized"
    GPU_OPTIMIZED = "gpu_optimized"
    GENERAL_PURPOSE = "general_purpose"


@dataclass
class CloudInstance:
    """Cloud instance configuration."""
    
    instance_id: str
    provider: CloudProvider
    instance_type: InstanceType
    
    # Instance specifications
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    created_at: datetime
    updated_at: datetime
    
    # Instance specifications with defaults
    gpu_count: int = 0
    
    # Network configuration
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    security_groups: List[str] = None
    
    # Instance state
    status: str = "pending"
    region: str = "us-east-1"
    zone: str = "us-east-1a"


@dataclass
class CloudConfig:
    """Cloud configuration."""
    
    provider: CloudProvider
    region: str
    credentials: Dict[str, Any]
    
    # Provider-specific settings
    aws_config: Dict[str, Any] = None
    azure_config: Dict[str, Any] = None
    gcp_config: Dict[str, Any] = None
    
    # Default settings
    default_instance_type: InstanceType = InstanceType.GENERAL_PURPOSE
    default_security_groups: List[str] = None


class CloudProviderManager:
    """
    Cloud provider manager for PBF-LB/M virtual environment.
    
    This class provides unified cloud provider management capabilities including
    instance provisioning, resource management, and multi-cloud orchestration
    for PBF-LB/M virtual testing and simulation environments.
    """
    
    def __init__(self):
        """Initialize the cloud provider manager."""
        self.providers = {}
        self.instances = {}
        self.configurations = {}
        
        logger.info("Cloud Provider Manager initialized")
    
    async def register_provider(
        self,
        provider: CloudProvider,
        config: CloudConfig
    ) -> bool:
        """
        Register a cloud provider.
        
        Args:
            provider: Cloud provider type
            config: Cloud configuration
            
        Returns:
            bool: Success status
        """
        try:
            if provider == CloudProvider.AWS:
                provider_instance = AWSProvider(config)
            elif provider == CloudProvider.AZURE:
                provider_instance = AzureProvider(config)
            elif provider == CloudProvider.GCP:
                provider_instance = GCPProvider(config)
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
            
            self.providers[provider] = provider_instance
            self.configurations[provider] = config
            
            logger.info(f"Cloud provider registered: {provider.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering cloud provider: {e}")
            return False
    
    async def create_instance(
        self,
        provider: CloudProvider,
        instance_type: InstanceType,
        instance_spec: Dict[str, Any]
    ) -> str:
        """
        Create a cloud instance.
        
        Args:
            provider: Cloud provider
            instance_type: Instance type
            instance_spec: Instance specifications
            
        Returns:
            str: Instance ID
        """
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider not registered: {provider}")
            
            provider_instance = self.providers[provider]
            instance_id = await provider_instance.create_instance(instance_type, instance_spec)
            
            # Store instance information
            cloud_instance = CloudInstance(
                instance_id=instance_id,
                provider=provider,
                instance_type=instance_type,
                cpu_cores=instance_spec.get('cpu_cores', 2),
                memory_gb=instance_spec.get('memory_gb', 4),
                storage_gb=instance_spec.get('storage_gb', 20),
                gpu_count=instance_spec.get('gpu_count', 0),
                region=instance_spec.get('region', 'us-east-1'),
                zone=instance_spec.get('zone', 'us-east-1a'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.instances[instance_id] = cloud_instance
            
            logger.info(f"Cloud instance created: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating cloud instance: {e}")
            return ""
    
    async def delete_instance(self, instance_id: str) -> bool:
        """
        Delete a cloud instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            bool: Success status
        """
        try:
            if instance_id not in self.instances:
                raise ValueError(f"Instance not found: {instance_id}")
            
            instance = self.instances[instance_id]
            provider = self.providers[instance.provider]
            
            success = await provider.delete_instance(instance_id)
            
            if success:
                del self.instances[instance_id]
                logger.info(f"Cloud instance deleted: {instance_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting cloud instance: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status."""
        try:
            if instance_id not in self.instances:
                raise ValueError(f"Instance not found: {instance_id}")
            
            instance = self.instances[instance_id]
            provider = self.providers[instance.provider]
            
            status = await provider.get_instance_status(instance_id)
            
            return {
                'instance_id': instance_id,
                'provider': instance.provider.value,
                'instance_type': instance.instance_type.value,
                'status': status.get('status', 'unknown'),
                'public_ip': status.get('public_ip'),
                'private_ip': status.get('private_ip'),
                'created_at': instance.created_at.isoformat(),
                'updated_at': instance.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting instance status: {e}")
            return {}
    
    async def list_instances(self, provider: CloudProvider = None) -> List[Dict[str, Any]]:
        """List cloud instances."""
        try:
            instances = []
            
            for instance_id, instance in self.instances.items():
                if provider is None or instance.provider == provider:
                    status = await self.get_instance_status(instance_id)
                    instances.append(status)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error listing instances: {e}")
            return []


class AWSProvider:
    """
    AWS cloud provider for PBF-LB/M virtual environment.
    
    This class provides AWS-specific cloud capabilities including EC2 instance
    management, VPC configuration, and AWS service integration.
    """
    
    def __init__(self, config: CloudConfig):
        """Initialize the AWS provider."""
        self.config = config
        self.ec2_client = None
        self.ec2_resource = None
        
        # Initialize AWS clients
        self._initialize_aws_clients()
        
        logger.info("AWS Provider initialized")
    
    def _initialize_aws_clients(self):
        """Initialize AWS clients."""
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not available. AWS functionality will be limited.")
            self.ec2_client = None
            self.ec2_resource = None
            return
            
        try:
            # Initialize EC2 client
            self.ec2_client = boto3.client(
                'ec2',
                region_name=self.config.region,
                aws_access_key_id=self.config.credentials.get('access_key_id'),
                aws_secret_access_key=self.config.credentials.get('secret_access_key')
            )
            
            # Initialize EC2 resource
            self.ec2_resource = boto3.resource(
                'ec2',
                region_name=self.config.region,
                aws_access_key_id=self.config.credentials.get('access_key_id'),
                aws_secret_access_key=self.config.credentials.get('secret_access_key')
            )
            
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {e}")
            self.ec2_client = None
            self.ec2_resource = None
    
    async def create_instance(
        self,
        instance_type: InstanceType,
        instance_spec: Dict[str, Any]
    ) -> str:
        """Create AWS EC2 instance."""
        try:
            # Map instance type to AWS instance type
            aws_instance_type = self._map_instance_type(instance_type)
            
            # Create instance
            response = self.ec2_client.run_instances(
                ImageId=instance_spec.get('ami_id', 'ami-0c02fb55956c7d316'),  # Amazon Linux 2
                MinCount=1,
                MaxCount=1,
                InstanceType=aws_instance_type,
                KeyName=instance_spec.get('key_name', 'default-key'),
                SecurityGroupIds=instance_spec.get('security_groups', ['default']),
                SubnetId=instance_spec.get('subnet_id'),
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': instance_spec.get('name', 'pbf-vm-instance')},
                            {'Key': 'Environment', 'Value': 'pbf-lbm'},
                            {'Key': 'CreatedBy', 'Value': 'pbf-virtual-environment'}
                        ]
                    }
                ]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            
            logger.info(f"AWS instance created: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating AWS instance: {e}")
            return ""
    
    async def delete_instance(self, instance_id: str) -> bool:
        """Delete AWS EC2 instance."""
        try:
            # Terminate instance
            response = self.ec2_client.terminate_instances(
                InstanceIds=[instance_id]
            )
            
            # Check if termination was successful
            if response['TerminatingInstances'][0]['InstanceId'] == instance_id:
                logger.info(f"AWS instance terminated: {instance_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error deleting AWS instance: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get AWS instance status."""
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[instance_id]
            )
            
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                
                return {
                    'status': instance['State']['Name'],
                    'public_ip': instance.get('PublicIpAddress'),
                    'private_ip': instance.get('PrivateIpAddress'),
                    'instance_type': instance['InstanceType'],
                    'launch_time': instance['LaunchTime'].isoformat()
                }
            else:
                return {'status': 'not_found'}
                
        except Exception as e:
            logger.error(f"Error getting AWS instance status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _map_instance_type(self, instance_type: InstanceType) -> str:
        """Map instance type to AWS instance type."""
        mapping = {
            InstanceType.COMPUTE_OPTIMIZED: 'c5.large',
            InstanceType.MEMORY_OPTIMIZED: 'r5.large',
            InstanceType.STORAGE_OPTIMIZED: 'i3.large',
            InstanceType.GPU_OPTIMIZED: 'p3.2xlarge',
            InstanceType.GENERAL_PURPOSE: 't3.medium'
        }
        
        return mapping.get(instance_type, 't3.medium')


class AzureProvider:
    """
    Azure cloud provider for PBF-LB/M virtual environment.
    
    This class provides Azure-specific cloud capabilities including VM management,
    resource group configuration, and Azure service integration.
    """
    
    def __init__(self, config: CloudConfig):
        """Initialize the Azure provider."""
        self.config = config
        self.compute_client = None
        self.resource_client = None
        
        # Initialize Azure clients
        self._initialize_azure_clients()
        
        logger.info("Azure Provider initialized")
    
    def _initialize_azure_clients(self):
        """Initialize Azure clients."""
        if not AZURE_AVAILABLE:
            logger.warning("Azure SDK not available. Azure functionality will be limited.")
            self.compute_client = None
            self.resource_client = None
            return
            
        try:
            # Initialize credentials
            credential = DefaultAzureCredential()
            
            # Initialize compute client
            self.compute_client = ComputeManagementClient(
                credential,
                self.config.credentials.get('subscription_id')
            )
            
            # Initialize resource client
            self.resource_client = ResourceManagementClient(
                credential,
                self.config.credentials.get('subscription_id')
            )
            
        except Exception as e:
            logger.error(f"Error initializing Azure clients: {e}")
    
    async def create_instance(
        self,
        instance_type: InstanceType,
        instance_spec: Dict[str, Any]
    ) -> str:
        """Create Azure VM instance."""
        try:
            # Map instance type to Azure VM size
            vm_size = self._map_instance_type(instance_type)
            
            # Create resource group if it doesn't exist
            resource_group_name = instance_spec.get('resource_group', 'pbf-lbm-rg')
            location = instance_spec.get('location', self.config.region)
            
            try:
                self.resource_client.resource_groups.create_or_update(
                    resource_group_name,
                    {'location': location}
                )
            except Exception:
                pass  # Resource group might already exist
            
            # Create VM
            vm_name = instance_spec.get('name', f'pbf-vm-{uuid.uuid4().hex[:8]}')
            
            # VM configuration
            vm_config = {
                'location': location,
                'os_profile': {
                    'computer_name': vm_name,
                    'admin_username': instance_spec.get('admin_username', 'azureuser'),
                    'admin_password': instance_spec.get('admin_password', 'P@ssw0rd123!')
                },
                'hardware_profile': {
                    'vm_size': vm_size
                },
                'storage_profile': {
                    'image_reference': {
                        'publisher': 'Canonical',
                        'offer': 'UbuntuServer',
                        'sku': '18.04-LTS',
                        'version': 'latest'
                    }
                },
                'network_profile': {
                    'network_interfaces': [{
                        'id': await self._create_network_interface(resource_group_name, location, vm_name)
                    }]
                }
            }
            
            # Create VM
            vm_operation = self.compute_client.virtual_machines.begin_create_or_update(
                resource_group_name,
                vm_name,
                vm_config
            )
            
            vm_result = vm_operation.result()
            
            logger.info(f"Azure VM created: {vm_result.name}")
            return vm_result.name
            
        except Exception as e:
            logger.error(f"Error creating Azure VM: {e}")
            return ""
    
    async def delete_instance(self, instance_id: str) -> bool:
        """Delete Azure VM instance."""
        try:
            # Extract resource group and VM name from instance ID
            resource_group_name = 'pbf-lbm-rg'  # Default resource group
            vm_name = instance_id
            
            # Delete VM
            vm_operation = self.compute_client.virtual_machines.begin_delete(
                resource_group_name,
                vm_name
            )
            
            vm_operation.result()
            
            logger.info(f"Azure VM deleted: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Azure VM: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get Azure VM status."""
        try:
            resource_group_name = 'pbf-lbm-rg'
            vm_name = instance_id
            
            # Get VM status
            vm = self.compute_client.virtual_machines.get(
                resource_group_name,
                vm_name,
                expand='instanceView'
            )
            
            # Get instance view
            instance_view = vm.instance_view
            status = instance_view.statuses[0].display_status if instance_view.statuses else 'unknown'
            
            return {
                'status': status,
                'public_ip': None,  # Would need additional network interface lookup
                'private_ip': None,  # Would need additional network interface lookup
                'instance_type': vm.hardware_profile.vm_size,
                'launch_time': vm.instance_view.statuses[0].time.isoformat() if instance_view.statuses else None
            }
            
        except Exception as e:
            logger.error(f"Error getting Azure VM status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _create_network_interface(self, resource_group_name: str, location: str, vm_name: str) -> str:
        """Create network interface for VM."""
        try:
            # This is a simplified implementation
            # In a real scenario, you would create VNet, subnet, and network interface
            return f"/subscriptions/{self.config.credentials.get('subscription_id')}/resourceGroups/{resource_group_name}/providers/Microsoft.Network/networkInterfaces/{vm_name}-nic"
            
        except Exception as e:
            logger.error(f"Error creating network interface: {e}")
            return ""
    
    def _map_instance_type(self, instance_type: InstanceType) -> str:
        """Map instance type to Azure VM size."""
        mapping = {
            InstanceType.COMPUTE_OPTIMIZED: 'Standard_D2s_v3',
            InstanceType.MEMORY_OPTIMIZED: 'Standard_E2s_v3',
            InstanceType.STORAGE_OPTIMIZED: 'Standard_L4s',
            InstanceType.GPU_OPTIMIZED: 'Standard_NC6s_v3',
            InstanceType.GENERAL_PURPOSE: 'Standard_B2s'
        }
        
        return mapping.get(instance_type, 'Standard_B2s')


class GCPProvider:
    """
    GCP cloud provider for PBF-LB/M virtual environment.
    
    This class provides GCP-specific cloud capabilities including Compute Engine
    management, VPC configuration, and GCP service integration.
    """
    
    def __init__(self, config: CloudConfig):
        """Initialize the GCP provider."""
        self.config = config
        self.compute_client = None
        self.storage_client = None
        
        # Initialize GCP clients
        self._initialize_gcp_clients()
        
        logger.info("GCP Provider initialized")
    
    def _initialize_gcp_clients(self):
        """Initialize GCP clients."""
        if not GCP_AVAILABLE:
            logger.warning("Google Cloud SDK not available. GCP functionality will be limited.")
            self.compute_client = None
            self.storage_client = None
            return
            
        try:
            # Initialize compute client
            self.compute_client = compute_v1.InstancesClient()
            
            # Initialize storage client
            self.storage_client = storage.Client()
            
        except Exception as e:
            logger.error(f"Error initializing GCP clients: {e}")
            self.compute_client = None
            self.storage_client = None
    
    async def create_instance(
        self,
        instance_type: InstanceType,
        instance_spec: Dict[str, Any]
    ) -> str:
        """Create GCP Compute Engine instance."""
        try:
            # Map instance type to GCP machine type
            machine_type = self._map_instance_type(instance_type)
            
            # Create instance
            instance_name = instance_spec.get('name', f'pbf-vm-{uuid.uuid4().hex[:8]}')
            zone = instance_spec.get('zone', f'{self.config.region}-a')
            project_id = self.config.credentials.get('project_id')
            
            # Instance configuration
            instance_config = {
                'name': instance_name,
                'machine_type': f'zones/{zone}/machineTypes/{machine_type}',
                'disks': [{
                    'boot': True,
                    'auto_delete': True,
                    'initialize_params': {
                        'source_image': 'projects/ubuntu-os-cloud/global/images/family/ubuntu-1804-lts'
                    }
                }],
                'network_interfaces': [{
                    'access_configs': [{
                        'type': 'ONE_TO_ONE_NAT',
                        'name': 'External NAT'
                    }],
                    'network': f'projects/{project_id}/global/networks/default'
                }],
                'metadata': {
                    'items': [{
                        'key': 'startup-script',
                        'value': '#!/bin/bash\napt-get update\napt-get install -y python3 python3-pip'
                    }]
                },
                'tags': {
                    'items': ['pbf-lbm', 'virtual-environment']
                }
            }
            
            # Create instance
            operation = self.compute_client.insert(
                project=project_id,
                zone=zone,
                instance_resource=instance_config
            )
            
            # Wait for operation to complete
            operation.result()
            
            logger.info(f"GCP instance created: {instance_name}")
            return instance_name
            
        except Exception as e:
            logger.error(f"Error creating GCP instance: {e}")
            return ""
    
    async def delete_instance(self, instance_id: str) -> bool:
        """Delete GCP Compute Engine instance."""
        try:
            project_id = self.config.credentials.get('project_id')
            zone = f'{self.config.region}-a'  # Default zone
            
            # Delete instance
            operation = self.compute_client.delete(
                project=project_id,
                zone=zone,
                instance=instance_id
            )
            
            # Wait for operation to complete
            operation.result()
            
            logger.info(f"GCP instance deleted: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting GCP instance: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get GCP instance status."""
        try:
            project_id = self.config.credentials.get('project_id')
            zone = f'{self.config.region}-a'  # Default zone
            
            # Get instance
            instance = self.compute_client.get(
                project=project_id,
                zone=zone,
                instance=instance_id
            )
            
            # Get public IP
            public_ip = None
            if instance.network_interfaces:
                access_configs = instance.network_interfaces[0].access_configs
                if access_configs:
                    public_ip = access_configs[0].nat_i_p
            
            # Get private IP
            private_ip = None
            if instance.network_interfaces:
                private_ip = instance.network_interfaces[0].network_i_p
            
            return {
                'status': instance.status,
                'public_ip': public_ip,
                'private_ip': private_ip,
                'instance_type': instance.machine_type.split('/')[-1],
                'launch_time': instance.creation_timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting GCP instance status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _map_instance_type(self, instance_type: InstanceType) -> str:
        """Map instance type to GCP machine type."""
        mapping = {
            InstanceType.COMPUTE_OPTIMIZED: 'c2-standard-4',
            InstanceType.MEMORY_OPTIMIZED: 'm2-standard-4',
            InstanceType.STORAGE_OPTIMIZED: 'c2-standard-4',
            InstanceType.GPU_OPTIMIZED: 'n1-standard-4',
            InstanceType.GENERAL_PURPOSE: 'e2-medium'
        }
        
        return mapping.get(instance_type, 'e2-medium')
