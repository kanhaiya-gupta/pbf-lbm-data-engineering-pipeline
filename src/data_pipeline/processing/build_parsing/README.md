# Build Parsing Module - PBF-LB/M Data Pipeline

## 🎯 **Mission Statement**
Create a world-class build file parsing system that leverages the existing libSLM and PySLM libraries to extract, process, and analyze PBF-LB/M machine build files with maximum efficiency and minimal reinvention.

## 🚫 **What We DON'T Do**
- ❌ **Reinvent file format parsing** - libSLM already handles this
- ❌ **Recreate 3D geometry processing** - PySLM already provides this
- ❌ **Build visualization from scratch** - PySLM has excellent visualization
- ❌ **Implement basic math operations** - Use existing libraries
- ❌ **Create new file I/O systems** - Use standard Python libraries

## ✅ **What We DO**
- ✅ **Orchestrate** libSLM and PySLM for maximum efficiency
- ✅ **Extract** specific data types (power, velocity, energy, paths)
- ✅ **Integrate** with our data pipeline architecture
- ✅ **Provide** clean, consistent APIs for our use cases
- ✅ **Bridge** between build files and our voxel/analysis systems

## 🏗️ **Architecture Principles**

### **1. Leverage Existing Libraries**
```python
# ✅ GOOD: Use libSLM for file parsing
from libSLM import slm, translators

# ✅ GOOD: Use PySLM for analysis
import pyslm
from pyslm import Slm

# ❌ BAD: Don't reimplement file parsing
# def parse_mtt_file_from_scratch():  # NO!
```

### **2. Focus on Integration, Not Implementation**
- **libSLM**: Handles all file format parsing (.mtt, .sli, .cli, etc.)
- **PySLM**: Handles 3D analysis, visualization, and advanced processing
- **Our Code**: Orchestrates, extracts specific data, integrates with pipeline

### **3. Clean Separation of Concerns**
```
Build Files → libSLM → Our Extractors → Data Pipeline → Visualization
     ↓              ↓           ↓            ↓              ↓
  .mtt/.sli    Raw parsing   Specific    Structured    Voxel/3D
  .cli/.rea    & decoding    data        data          analysis
```

## 📁 **Module Structure**

```
build_parsing/
├── __init__.py                    # Factory functions and main exports
├── base_parser.py                 # Abstract base class (minimal)
├── core/
│   ├── build_file_parser.py       # Main orchestrator (uses libSLM/PySLM)
│   ├── format_detector.py         # Auto-detect formats (leverages libSLM)
│   └── metadata_extractor.py      # Extract metadata (from libSLM output)
├── format_parsers/                # Format-specific wrappers
│   ├── eos_parser.py              # EOS wrapper (uses libSLM.translators.eos)
│   ├── mtt_parser.py              # MTT wrapper (uses libSLM.translators.mtt)
│   ├── realizer_parser.py         # Realizer wrapper (uses libSLM.translators.realizer)
│   └── slm_parser.py              # SLM wrapper (uses libSLM.translators.slmsol)
├── data_extractors/               # Extract specific data types
│   ├── power_extractor.py         # Laser power analysis (from libSLM data)
│   ├── velocity_extractor.py      # Scan velocity analysis (from libSLM data)
│   ├── path_extractor.py          # Scan path geometry (from libSLM data)
│   ├── energy_extractor.py        # Energy consumption (calculated from libSLM data)
│   └── layer_extractor.py         # Layer-specific data (from libSLM data)
└── utils/
    ├── file_utils.py              # File handling (standard Python)
    └── validation_utils.py        # Data validation (standard Python)
```

## 🔧 **Implementation Strategy**

### **Phase 1: Foundation (libSLM Integration)**
1. **`base_parser.py`** - Minimal abstract base class
2. **`build_file_parser.py`** - Main orchestrator that uses libSLM
3. **Format parsers** - Thin wrappers around libSLM translators
4. **Basic data extraction** - Extract raw data from libSLM output

### **Phase 2: Data Extraction (PySLM Integration)**
1. **Power extractor** - Use PySLM for laser power analysis
2. **Velocity extractor** - Use PySLM for scan velocity analysis
3. **Path extractor** - Use PySLM for scan path geometry
4. **Energy extractor** - Calculate energy from PySLM data

### **Phase 3: Advanced Analysis (PySLM Features)**
1. **Build time analysis** - Use PySLM's build time features
2. **Heatmap generation** - Use PySLM's visualization
3. **Support structure analysis** - Use PySLM's support features
4. **Parametric studies** - Use PySLM's parametric capabilities

## 📋 **Code Examples**

### **✅ Correct Approach - Leverage libSLM**
```python
# base_parser.py
from abc import ABC, abstractmethod
from ..external import LIBSLM_AVAILABLE, PYSLM_AVAILABLE

class BaseBuildParser(ABC):
    def __init__(self):
        if not LIBSLM_AVAILABLE:
            raise RuntimeError("libSLM required for build parsing")
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse build file using libSLM"""
        pass

# eos_parser.py
from libSLM.translators import eos

class EOSParser(BaseBuildParser):
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        # Use libSLM's EOS reader - don't reinvent!
        reader = eos.Reader()
        return reader.read(str(file_path))
```

### **✅ Correct Approach - Leverage PySLM**
```python
# power_extractor.py
import pyslm
from pyslm import Slm

class PowerExtractor:
    def __init__(self):
        if not PYSLM_AVAILABLE:
            raise RuntimeError("PySLM required for power analysis")
    
    def analyze_power_distribution(self, build_data):
        # Use PySLM's analysis capabilities
        slm = Slm()
        return slm.analyze_power_distribution(build_data)
```

### **❌ Wrong Approach - Reinventing**
```python
# ❌ DON'T DO THIS
def parse_mtt_file_manually(file_path):
    # Manually parsing MTT format - libSLM already does this!
    with open(file_path, 'rb') as f:
        # Hundreds of lines of custom parsing code...
        pass

# ❌ DON'T DO THIS  
def calculate_scan_velocity_from_scratch(coordinates, time):
    # PySLM already has velocity analysis!
    # Don't reinvent the wheel
    pass
```

## 🎯 **Success Metrics**

1. **Leverage Ratio**: >90% of functionality should use libSLM/PySLM
2. **Code Efficiency**: Minimal custom code, maximum library usage
3. **Integration Quality**: Seamless integration with our data pipeline
4. **Performance**: Fast parsing using optimized libSLM/PySLM
5. **Maintainability**: Easy to maintain by leveraging stable libraries

## 🚀 **Key Benefits**

1. **World-Class Quality**: Leverage years of libSLM/PySLM development
2. **Fast Development**: Don't reinvent, just integrate
3. **Reliability**: Use battle-tested libraries
4. **Performance**: Optimized C++ (libSLM) and Python (PySLM) code
5. **Future-Proof**: Libraries are actively maintained and updated

## 📚 **Dependencies**

- **libSLM**: C++ library with Python bindings for file parsing
- **PySLM**: High-level Python library for analysis and visualization
- **Our External Module**: Proper integration with libSLM/PySLM
- **Standard Python**: For utilities and integration code

## 🎯 **Remember**
> "Don't reinvent the wheel. Use libSLM for parsing, PySLM for analysis, and focus on integration and orchestration."

This approach ensures we build a world-class system by leveraging existing world-class libraries rather than starting from scratch.
