"""
Defect type enumeration for PBF-LB/M quality analysis.
"""

from enum import Enum


class DefectType(Enum):
    """
    Enumeration for different types of defects in PBF-LB/M processes.
    
    This enum categorizes various defects that can occur during
    powder bed fusion laser beam/metal additive manufacturing.
    """
    
    # Porosity defects
    POROSITY = "porosity"
    KEYHOLE_POROSITY = "keyhole_porosity"
    LACK_OF_FUSION = "lack_of_fusion"
    GAS_POROSITY = "gas_porosity"
    
    # Cracking defects
    HOT_CRACKING = "hot_cracking"
    COLD_CRACKING = "cold_cracking"
    STRESS_CRACKING = "stress_cracking"
    MICRO_CRACKING = "micro_cracking"
    
    # Dimensional defects
    DIMENSIONAL_DEVIATION = "dimensional_deviation"
    WARPAGE = "warpage"
    DISTORTION = "distortion"
    SHRINKAGE = "shrinkage"
    
    # Surface defects
    SURFACE_ROUGHNESS = "surface_roughness"
    SURFACE_CONTAMINATION = "surface_contamination"
    BALLING = "balling"
    SPATTER = "spatter"
    
    # Microstructural defects
    UNMELTED_POWDER = "unmelted_powder"
    INCOMPLETE_FUSION = "incomplete_fusion"
    MICROSTRUCTURAL_INHOMOGENEITY = "microstructural_inhomogeneity"
    GRAIN_BOUNDARY_DEFECTS = "grain_boundary_defects"
    
    # Process-related defects
    LAYER_SHIFT = "layer_shift"
    POWDER_BED_DEFECTS = "powder_bed_defects"
    LASER_DEFOCUS = "laser_defocus"
    SCANNING_ERRORS = "scanning_errors"
    
    # Material-related defects
    COMPOSITION_DEVIATION = "composition_deviation"
    CONTAMINATION = "contamination"
    OXIDATION = "oxidation"
    HYDROGEN_EMBRITTLEMENT = "hydrogen_embrittlement"
    
    # Quality categories
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    COSMETIC = "cosmetic"
    
    @classmethod
    def get_porosity_defects(cls):
        """Get porosity-related defect types."""
        return [
            cls.POROSITY,
            cls.KEYHOLE_POROSITY,
            cls.LACK_OF_FUSION,
            cls.GAS_POROSITY
        ]
    
    @classmethod
    def get_cracking_defects(cls):
        """Get cracking-related defect types."""
        return [
            cls.HOT_CRACKING,
            cls.COLD_CRACKING,
            cls.STRESS_CRACKING,
            cls.MICRO_CRACKING
        ]
    
    @classmethod
    def get_dimensional_defects(cls):
        """Get dimensional defect types."""
        return [
            cls.DIMENSIONAL_DEVIATION,
            cls.WARPAGE,
            cls.DISTORTION,
            cls.SHRINKAGE
        ]
    
    @classmethod
    def get_surface_defects(cls):
        """Get surface-related defect types."""
        return [
            cls.SURFACE_ROUGHNESS,
            cls.SURFACE_CONTAMINATION,
            cls.BALLING,
            cls.SPATTER
        ]
    
    @classmethod
    def get_microstructural_defects(cls):
        """Get microstructural defect types."""
        return [
            cls.UNMELTED_POWDER,
            cls.INCOMPLETE_FUSION,
            cls.MICROSTRUCTURAL_INHOMOGENEITY,
            cls.GRAIN_BOUNDARY_DEFECTS
        ]
    
    @classmethod
    def get_process_defects(cls):
        """Get process-related defect types."""
        return [
            cls.LAYER_SHIFT,
            cls.POWDER_BED_DEFECTS,
            cls.LASER_DEFOCUS,
            cls.SCANNING_ERRORS
        ]
    
    @classmethod
    def get_material_defects(cls):
        """Get material-related defect types."""
        return [
            cls.COMPOSITION_DEVIATION,
            cls.CONTAMINATION,
            cls.OXIDATION,
            cls.HYDROGEN_EMBRITTLEMENT
        ]
    
    @classmethod
    def get_quality_categories(cls):
        """Get quality category defect types."""
        return [
            cls.CRITICAL,
            cls.MAJOR,
            cls.MINOR,
            cls.COSMETIC
        ]
    
    def get_severity_level(self):
        """Get the severity level of this defect type."""
        severity_map = {
            # Critical defects
            cls.HOT_CRACKING: 4,
            cls.COLD_CRACKING: 4,
            cls.STRESS_CRACKING: 4,
            cls.CRITICAL: 4,
            
            # Major defects
            cls.KEYHOLE_POROSITY: 3,
            cls.LACK_OF_FUSION: 3,
            cls.DIMENSIONAL_DEVIATION: 3,
            cls.WARPAGE: 3,
            cls.MAJOR: 3,
            
            # Minor defects
            cls.POROSITY: 2,
            cls.GAS_POROSITY: 2,
            cls.SURFACE_ROUGHNESS: 2,
            cls.BALLING: 2,
            cls.MINOR: 2,
            
            # Cosmetic defects
            cls.SPATTER: 1,
            cls.SURFACE_CONTAMINATION: 1,
            cls.COSMETIC: 1,
        }
        return severity_map.get(self, 2)  # Default to minor
    
    def is_critical(self):
        """Check if this is a critical defect type."""
        return self.get_severity_level() >= 4
    
    def is_major(self):
        """Check if this is a major defect type."""
        return self.get_severity_level() == 3
    
    def is_minor(self):
        """Check if this is a minor defect type."""
        return self.get_severity_level() == 2
    
    def is_cosmetic(self):
        """Check if this is a cosmetic defect type."""
        return self.get_severity_level() == 1