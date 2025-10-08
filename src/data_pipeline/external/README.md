# External Software Dependencies

This directory contains external software packages that are integrated into the PBF-LB/M Data Pipeline. These are specialized libraries for handling PBF-LB/M (Powder Bed Fusion - Laser Beam/Metal) machine build files and data processing.

## 📦 External Software Packages

### 1. libSLM
**Location**: `src/data_pipeline/external/libSLM/`

**Description**: A C++ library with Python bindings for reading and writing PBF-LB/M machine build files. It supports multiple file formats used in additive manufacturing.

**Supported File Formats**:
- `.mtt` - Machine Tool Technology format
- `.rea` - Realizer format  
- `.f&s` - F&S format
- `.sli` - SLI format
- `.slm` - SLM format
- `.cli` - CLI format

**Key Features**:
- Cross-platform C++ library
- Python bindings via pybind11
- Support for multiple PBF-LB/M machine formats
- Built-in translators for different file formats
- CMake build system

**Installation**: See `libSLM/README_INSTALLATION.md` for detailed installation instructions.

**Usage**: Integrated via `BuildFileParser` in the data pipeline for parsing machine build files.

### 2. PySLM
**Location**: `src/data_pipeline/external/pyslm/`

**Description**: A high-level Python library built on top of libSLM for PBF-LB/M data analysis, visualization, and processing. Provides advanced features for additive manufacturing research.

**Key Features**:
- High-level Python API for PBF-LB/M data
- 3D visualization and analysis tools
- Build time analysis
- Heatmap generation
- Support structure analysis
- Multi-threaded processing
- Parametric studies

**Installation**: 
```bash
cd src/data_pipeline/external/pyslm/
pip install -e .
```

**Usage**: Integrated via `BuildFileParser` for advanced PBF-LB/M data analysis and visualization.

## 🔧 Integration with Data Pipeline

Both external software packages are integrated into the data pipeline through:

1. **BuildFileParser** (`src/data_pipeline/processing/build_file_parser.py`)
   - Uses libSLM for low-level file parsing
   - Uses PySLM for high-level analysis and visualization
   - Provides unified interface for PBF-LB/M build file processing

2. **Voxelization Pipeline** (`src/data_pipeline/visualization/`)
   - Leverages PySLM for 3D visualization
   - Uses libSLM for process parameter extraction
   - Enables voxel-level process control and analysis

## 📋 Dependencies

### libSLM Dependencies:
- C++ compiler (GCC/Clang)
- CMake (>=3.10)
- Eigen3 library
- pybind11 (for Python bindings)
- TBB (Threading Building Blocks)

### PySLM Dependencies:
- Python 3.7+
- NumPy
- Matplotlib
- VTK (for 3D visualization)
- libSLM (as dependency)

## 🚀 Usage Examples

### Basic Build File Parsing:
```python
from data_pipeline.processing.build_file_parser import BuildFileParser

parser = BuildFileParser()
build_data = parser.parse_build_file("path/to/buildfile.mtt")
```

### Advanced Analysis with PySLM:
```python
from data_pipeline.processing.build_file_parser import BuildFileParser

parser = BuildFileParser()
if parser.pyslm_available:
    # Use PySLM for advanced analysis
    analysis_results = parser.analyze_build_parameters()
```

## 📝 License Information

- **libSLM**: See `libSLM/LICENSE` for license details
- **PySLM**: See `pyslm/LICENSE` for license details

## 🔄 Updates and Maintenance

These external packages are:
- **Version-controlled** in the repository for reproducibility
- **Built from source** to ensure compatibility
- **Integrated** into the data pipeline build process
- **Documented** with installation and usage instructions

## ⚠️ Important Notes

1. **Build Requirements**: Both packages require compilation from source
2. **Platform Compatibility**: Tested on Linux (WSL), may need adjustments for other platforms
3. **Version Compatibility**: Ensure compatible versions of dependencies
4. **Memory Usage**: These packages can be memory-intensive for large build files

## 🛠️ Troubleshooting

### Common Issues:
1. **Build Failures**: Ensure all dependencies are installed
2. **Import Errors**: Check Python path and installation
3. **Memory Issues**: Use smaller build files or increase system memory
4. **Platform Issues**: Check platform-specific build instructions

### Getting Help:
- Check individual package documentation in their respective directories
- Review installation logs for specific error messages
- Ensure all system dependencies are properly installed
