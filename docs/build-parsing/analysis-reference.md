# Build Parser Analysis Reference

## Quick Analysis Formulas

### Energy Density Analysis
```python
# Calculate energy density for each segment
energy_density = power / velocity  # Units: J/mm²

# Identify high energy regions (potential over-melting)
high_energy_threshold = 0.3  # J/mm²
over_melted_regions = [seg for seg in scan_points 
                      if (seg['power'] / seg['velocity']) > high_energy_threshold]
```

### Heat Input Modeling
```python
# Calculate heat input per unit length
def calculate_heat_input(segment):
    length = calculate_distance(segment['start_point'], segment['end_point'])
    return (segment['power'] * segment['exposure_time']) / length  # Units: J/mm

# Apply to all segments
heat_inputs = [calculate_heat_input(seg) for seg in scan_points]
```

### Cooling Time Analysis
```python
# Calculate cooling time between segments
def calculate_cooling_time(segment):
    return segment['point_delay'] + segment['jump_delay']  # Units: μs

# Analyze cooling patterns
cooling_times = [calculate_cooling_time(seg) for seg in scan_points]
average_cooling = sum(cooling_times) / len(cooling_times)
```

### Scan Path Efficiency
```python
# Calculate total path length
total_path_length = sum(calculate_distance(seg['start_point'], seg['end_point']) 
                       for seg in scan_points)

# Calculate build volume (approximate)
build_bounds = calculate_build_bounds(scan_points)
build_volume = (build_bounds['x_max'] - build_bounds['x_min']) * \
               (build_bounds['y_max'] - build_bounds['y_min']) * \
               (build_bounds['z_max'] - build_bounds['z_min'])

# Calculate efficiency
efficiency = build_volume / total_path_length  # Units: mm²
```

## Parameter Optimization Examples

### Power Optimization
```python
# Group segments by power and analyze performance
power_groups = {}
for seg in scan_points:
    power = seg['power']
    if power not in power_groups:
        power_groups[power] = []
    power_groups[power].append(seg)

# Find optimal power range
power_performance = {}
for power, segments in power_groups.items():
    # Calculate average energy density
    avg_energy_density = sum(seg['power']/seg['velocity'] for seg in segments) / len(segments)
    power_performance[power] = avg_energy_density

# Recommend optimal power
optimal_power = min(power_performance, key=power_performance.get)
```

### Velocity Optimization
```python
# Analyze velocity distribution
velocity_distribution = {}
for seg in scan_points:
    velocity = seg['velocity']
    if velocity not in velocity_distribution:
        velocity_distribution[velocity] = 0
    velocity_distribution[velocity] += 1

# Find most common velocity (likely optimal)
optimal_velocity = max(velocity_distribution, key=velocity_distribution.get)
```

## Quality Control Checks

### Parameter Range Validation
```python
def validate_parameters(scan_points):
    issues = []
    
    for i, seg in enumerate(scan_points):
        # Power validation
        if not (0 <= seg['power'] <= 500):
            issues.append(f"Segment {i}: Invalid power {seg['power']}W")
        
        # Velocity validation
        if not (100 <= seg['velocity'] <= 2000):
            issues.append(f"Segment {i}: Invalid velocity {seg['velocity']}mm/s")
        
        # Exposure time validation
        if not (1 <= seg['exposure_time'] <= 100):
            issues.append(f"Segment {i}: Invalid exposure time {seg['exposure_time']}μs")
    
    return issues
```

### Consistency Checks
```python
def check_parameter_consistency(scan_points):
    # Group by hatch
    hatch_groups = {}
    for seg in scan_points:
        key = (seg['layer_index'], seg['hatch_index'])
        if key not in hatch_groups:
            hatch_groups[key] = []
        hatch_groups[key].append(seg)
    
    inconsistencies = []
    for (layer, hatch), segments in hatch_groups.items():
        # Check power consistency within hatch
        powers = [seg['power'] for seg in segments]
        if len(set(powers)) > 1:
            inconsistencies.append(f"Layer {layer}, Hatch {hatch}: Inconsistent power {powers}")
        
        # Check velocity consistency within hatch
        velocities = [seg['velocity'] for seg in segments]
        if len(set(velocities)) > 1:
            inconsistencies.append(f"Layer {layer}, Hatch {hatch}: Inconsistent velocity {velocities}")
    
    return inconsistencies
```

## Visualization Examples

### 3D Parameter Mapping
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_parameters(scan_points):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates and parameters
    x_coords = [seg['start_point'][0] for seg in scan_points]
    y_coords = [seg['start_point'][1] for seg in scan_points]
    z_coords = [seg['layer_index'] for seg in scan_points]
    powers = [seg['power'] for seg in scan_points]
    
    # Create 3D scatter plot colored by power
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=powers, cmap='viridis')
    plt.colorbar(scatter, label='Power (W)')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Layer')
    ax.set_title('3D Power Distribution')
    
    plt.show()
```

### Parameter Distribution Analysis
```python
def plot_parameter_distributions(scan_points):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Power distribution
    powers = [seg['power'] for seg in scan_points]
    axes[0, 0].hist(powers, bins=50, alpha=0.7)
    axes[0, 0].set_title('Power Distribution')
    axes[0, 0].set_xlabel('Power (W)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Velocity distribution
    velocities = [seg['velocity'] for seg in scan_points]
    axes[0, 1].hist(velocities, bins=50, alpha=0.7)
    axes[0, 1].set_title('Velocity Distribution')
    axes[0, 1].set_xlabel('Velocity (mm/s)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Energy density distribution
    energy_densities = [seg['power']/seg['velocity'] for seg in scan_points]
    axes[1, 0].hist(energy_densities, bins=50, alpha=0.7)
    axes[1, 0].set_title('Energy Density Distribution')
    axes[1, 0].set_xlabel('Energy Density (J/mm²)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Power vs Velocity scatter
    axes[1, 1].scatter(velocities, powers, alpha=0.5)
    axes[1, 1].set_title('Power vs Velocity')
    axes[1, 1].set_xlabel('Velocity (mm/s)')
    axes[1, 1].set_ylabel('Power (W)')
    
    plt.tight_layout()
    plt.show()
```

## Performance Metrics

### Build Time Estimation
```python
def estimate_build_time(scan_points):
    total_time = 0
    
    for seg in scan_points:
        # Calculate segment time
        length = calculate_distance(seg['start_point'], seg['end_point'])
        scan_time = length / seg['velocity']  # seconds
        
        # Add delays
        delay_time = (seg['point_delay'] + seg['jump_delay']) / 1e6  # convert μs to seconds
        
        total_time += scan_time + delay_time
    
    return total_time  # seconds
```

### Energy Consumption
```python
def calculate_energy_consumption(scan_points):
    total_energy = 0
    
    for seg in scan_points:
        length = calculate_distance(seg['start_point'], seg['end_point'])
        scan_time = length / seg['velocity']
        energy = seg['power'] * scan_time  # Joules
        total_energy += energy
    
    return total_energy  # Joules
```
