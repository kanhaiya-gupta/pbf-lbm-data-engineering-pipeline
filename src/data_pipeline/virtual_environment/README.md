# Virtual Environment for PBF-LB/M Data Pipeline

This module provides a comprehensive virtual testing and simulation environment for PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) processes, including virtual machine management, simulation engines, digital twin capabilities, testing frameworks, and cloud integration.

## Features

- **Virtual Machine Management**: VM orchestration, provisioning, storage, and security
- **Simulation Engines**: Thermal, fluid dynamics, mechanical, material physics, and multi-physics simulations
- **Digital Twin**: Real-time synchronization, predictive analytics, and validation
- **Testing Frameworks**: Experiment design, automated testing, validation, and reporting
- **Cloud Integration**: Cloud providers, distributed computing, containerization, and serverless computing

## Quick Start

```python
from src.data_pipeline.virtual_environment import (
    VMOrchestrator,
    ThermalSimulator,
    DigitalTwinModel,
    ExperimentDesigner,
    CloudProviderManager
)

# Initialize virtual environment components
vm_orchestrator = VMOrchestrator()
thermal_sim = ThermalSimulator()
digital_twin = DigitalTwinModel("twin_001")
experiment_designer = ExperimentDesigner("exp_001")
cloud_manager = CloudProviderManager()

# Create virtual machine
vm_id = await vm_orchestrator.create_vm("test-vm", {"cpu": 4, "memory": 8192})

# Run thermal simulation
result = await thermal_sim.simulate_thermal_behavior(
    laser_power=250,
    scan_speed=1.2,
    layer_thickness=0.06
)

# Create digital twin
digital_twin.define_process_model({
    "laser_power_profile": "constant",
    "scan_strategy": "zigzag"
})

# Design experiment
experiments = experiment_designer.create_parameter_sweep({
    "laser_power": [200, 250, 300],
    "scan_speed": [1.0, 1.2, 1.4]
})

# Create cloud provider
provider_id = await cloud_manager.create_provider(
    name="AWS Provider",
    provider_type=CloudProviderType.AWS
)
```

## Components

### Virtual Machine Management
- **Orchestration**: VM lifecycle management and scaling
- **Provisioning**: Automated VM setup and configuration
- **Storage**: Volume management and data persistence
- **Security**: Access control and network isolation

### Simulation Engines
- **Thermal Simulation**: Heat transfer and temperature distribution
- **Fluid Dynamics**: Powder flow and melt pool dynamics
- **Mechanical Simulation**: Stress, strain, and deformation analysis
- **Material Physics**: Microstructural evolution and phase transformations
- **Multi-Physics**: Coupled physics simulations

### Digital Twin
- **Twin Models**: Process and quality prediction models
- **Synchronization**: Real-time data synchronization
- **Prediction**: Quality and process parameter prediction
- **Validation**: Model accuracy and reliability assessment

### Testing Frameworks
- **Experiment Design**: Parameter sweeps, factorial designs, and DoE
- **Automated Testing**: Test case management and execution
- **Validation**: Data integrity and performance validation
- **Reporting**: Comprehensive test reports and documentation

### Cloud Integration
- **Cloud Providers**: AWS, Azure, and GCP integration
- **Distributed Computing**: Task distribution and parallel processing
- **Containerization**: Docker and Kubernetes orchestration
- **Serverless Computing**: Function deployment and event-driven execution

## Examples

See the `examples/` directory for comprehensive usage examples:
- `virtual_environment_example.py` - Complete virtual environment workflow
- `cloud_integration_example.py` - Cloud integration examples

## Dependencies

- `asyncio` - Asynchronous programming
- `docker` - Container management
- `kubernetes` - Container orchestration
- `boto3` - AWS SDK
- `azure-mgmt-compute` - Azure management
- `google-cloud-compute` - GCP management
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `sklearn` - Machine learning

## Configuration

Configure the virtual environment through environment variables or configuration files:

```bash
# Virtual Machine Settings
export VM_DEFAULT_CPU=4
export VM_DEFAULT_MEMORY=8192
export VM_DEFAULT_STORAGE=100

# Simulation Settings
export SIMULATION_TIMEOUT=3600
export SIMULATION_MEMORY_LIMIT=16Gi

# Cloud Settings
export AWS_ACCESS_KEY_ID=your_access_key
export AZURE_SUBSCRIPTION_ID=your_subscription_id
export GOOGLE_CLOUD_PROJECT=your_project_id
```

## Architecture

The virtual environment follows a modular architecture with clear separation of concerns:

```
virtual_environment/
├── vm_management/          # Virtual machine management
├── simulation_engines/     # Physics simulation engines
├── digital_twin/          # Digital twin capabilities
├── testing_frameworks/    # Testing and validation
├── cloud_integration/     # Cloud and distributed computing
└── examples/              # Usage examples
```

## Getting Started

1. **Install Dependencies**: Install required Python packages
2. **Configure Environment**: Set up environment variables
3. **Initialize Components**: Create and configure virtual environment components
4. **Run Examples**: Execute example scripts to understand usage
5. **Build Workflows**: Create custom workflows for your specific needs

## Support

For questions and support, please refer to the individual component documentation or contact the development team.
