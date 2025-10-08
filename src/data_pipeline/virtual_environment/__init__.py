"""
Virtual Environment Module for PBF-LB/M Data Pipeline

This module provides comprehensive virtual machine integration and virtual environment
capabilities for PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) additive manufacturing
systems. It extends the data pipeline with virtual testing and simulation environments
that enable researchers to conduct controlled experiments, validate models, and test
process parameters using pipeline data.

Key Features:
- VM Management: Virtual machine orchestration, provisioning, storage, and security
- Simulation Engines: Thermal, fluid, mechanical, material, and multi-physics simulation
- Digital Twin: Real-time synchronization, prediction, and validation capabilities
- Testing Frameworks: Experimental design, automated testing, and validation
- Cloud Integration: Cloud providers, distributed computing, containerization, serverless

Architecture:
- VM Management: Complete virtual machine lifecycle management
- Simulation Engines: Multi-physics simulation capabilities
- Digital Twin: Digital representation of physical PBF-LB/M systems
- Testing Frameworks: Comprehensive virtual testing and validation
- Cloud Integration: Scalable cloud and distributed computing support
"""

# VM Management Components - Core functionality
try:
    from .vm_management.orchestration import VMOrchestrator, VMLifecycleManager, VMResourceManager
    from .vm_management.provisioning import VMProvisioner, VMConfigurationManager, VMImageManager
    from .vm_management.storage import VMStorageManager, VMDataManager, VMBackupManager
    from .vm_management.security import VMSecurityManager, VMIsolationManager, VMAccessControl
    VM_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    VM_MANAGEMENT_AVAILABLE = False
    import warnings
    warnings.warn(f"VM Management components not available: {e}")

# Simulation Engine Components - Optional heavy dependencies
try:
    from .simulation_engines.thermal_simulation import ThermalSimulator, ThermalSolver, ThermalAnalyzer
    from .simulation_engines.fluid_dynamics import FluidDynamicsSimulator, CFDSolver, FlowAnalyzer
    from .simulation_engines.mechanical_simulation import MechanicalSimulator, StressSolver, DeformationAnalyzer
    from .simulation_engines.material_physics import MaterialPhysicsSimulator, PhaseChangeSolver, MicrostructureAnalyzer
    from .simulation_engines.multi_physics import MultiPhysicsSimulator, PhysicsCoupler, CoupledSolver
    SIMULATION_ENGINES_AVAILABLE = True
except ImportError as e:
    SIMULATION_ENGINES_AVAILABLE = False
    import warnings
    warnings.warn(f"Simulation engines not available: {e}")

# Digital Twin Components - Optional heavy dependencies
try:
    from .digital_twin.twin_models import DigitalTwinModel, ProcessTwinModel, QualityTwinModel
    from .digital_twin.synchronization import TwinSynchronizer, RealTimeSync, DataSyncManager
    from .digital_twin.prediction import TwinPredictor, QualityPredictor, ProcessPredictor
    from .digital_twin.validation import TwinValidator, ModelValidator, AccuracyValidator
    DIGITAL_TWIN_AVAILABLE = True
except ImportError as e:
    DIGITAL_TWIN_AVAILABLE = False
    import warnings
    warnings.warn(f"Digital twin components not available: {e}")

# Testing Framework Components - Optional heavy dependencies
try:
    from .testing_frameworks.experiment_design import VirtualExperimentDesigner, ParameterSweepDesigner, DoEDesigner
    from .testing_frameworks.automated_testing import AutomatedTestRunner, TestOrchestrator, TestScheduler
    from .testing_frameworks.validation import VirtualValidator, ResultValidator, ComparisonValidator
    from .testing_frameworks.reporting import TestReportGenerator, TestVisualizer, TestDocumentation
    TESTING_FRAMEWORKS_AVAILABLE = True
except ImportError as e:
    TESTING_FRAMEWORKS_AVAILABLE = False
    import warnings
    warnings.warn(f"Testing frameworks not available: {e}")

# Cloud Integration Components - Optional heavy dependencies
try:
    from .cloud_integration.cloud_providers import CloudProviderManager, AWSProvider, AzureProvider, GCPProvider
    from .cloud_integration.distributed_computing import DistributedComputingManager, ClusterManager, JobScheduler
    from .cloud_integration.containerization import ContainerManager, DockerManager, KubernetesManager
    from .cloud_integration.serverless import ServerlessManager, LambdaManager, FunctionManager
    CLOUD_INTEGRATION_AVAILABLE = True
except ImportError as e:
    CLOUD_INTEGRATION_AVAILABLE = False
    import warnings
    warnings.warn(f"Cloud integration components not available: {e}")

__all__ = [
    # VM Management
    'VMOrchestrator',
    'VMLifecycleManager',
    'VMResourceManager',
    'VMProvisioner',
    'VMConfigurationManager',
    'VMImageManager',
    'VMStorageManager',
    'VMDataManager',
    'VMBackupManager',
    'VMSecurityManager',
    'VMIsolationManager',
    'VMAccessControl',
    
    # Simulation Engines
    'ThermalSimulator',
    'ThermalSolver',
    'ThermalAnalyzer',
    'FluidDynamicsSimulator',
    'CFDSolver',
    'FlowAnalyzer',
    'MechanicalSimulator',
    'StressSolver',
    'DeformationAnalyzer',
    'MaterialPhysicsSimulator',
    'PhaseChangeSolver',
    'MicrostructureAnalyzer',
    'MultiPhysicsSimulator',
    'PhysicsCoupler',
    'CoupledSolver',
    
    # Digital Twin
    'DigitalTwinModel',
    'ProcessTwinModel',
    'QualityTwinModel',
    'TwinSynchronizer',
    'RealTimeSync',
    'DataSyncManager',
    'TwinPredictor',
    'QualityPredictor',
    'ProcessPredictor',
    'TwinValidator',
    'ModelValidator',
    'AccuracyValidator',
    
    # Testing Frameworks
    'VirtualExperimentDesigner',
    'ParameterSweepDesigner',
    'DoEDesigner',
    'AutomatedTestRunner',
    'TestOrchestrator',
    'TestScheduler',
    'VirtualValidator',
    'ResultValidator',
    'ComparisonValidator',
    'TestReportGenerator',
    'TestVisualizer',
    'TestDocumentation',
    
    # Cloud Integration
    'CloudProviderManager',
    'AWSProvider',
    'AzureProvider',
    'GCPProvider',
    'DistributedComputingManager',
    'ClusterManager',
    'JobScheduler',
    'ContainerManager',
    'DockerManager',
    'KubernetesManager',
    'ServerlessManager',
    'LambdaManager',
    'FunctionManager',
]

# Version information
__version__ = "1.0.0"
__author__ = "PBF-LB/M Research Team"
__description__ = "Virtual environment for PBF-LB/M data pipeline"
