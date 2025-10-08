"""
Quality tier enumeration for PBF-LB/M quality assessment.
"""

from enum import Enum


class QualityTier(Enum):
    """
    Enumeration for quality tiers in PBF-LB/M processes.
    
    This enum defines the quality levels that can be assigned
    to PBF processes based on various quality metrics and criteria.
    """
    
    # Premium quality tiers
    PREMIUM = "premium"
    EXCELLENT = "excellent"
    OUTSTANDING = "outstanding"
    
    # Standard quality tiers
    HIGH = "high"
    GOOD = "good"
    STANDARD = "standard"
    ACCEPTABLE = "acceptable"
    
    # Lower quality tiers
    FAIR = "fair"
    POOR = "poor"
    SUBSTANDARD = "substandard"
    
    # Rejection tiers
    UNACCEPTABLE = "unacceptable"
    REJECTED = "rejected"
    FAILED = "failed"
    
    # Special tiers
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    REQUIRES_RETEST = "requires_retest"
    
    @classmethod
    def get_premium_tiers(cls):
        """Get premium quality tiers."""
        return [
            cls.PREMIUM,
            cls.EXCELLENT,
            cls.OUTSTANDING
        ]
    
    @classmethod
    def get_standard_tiers(cls):
        """Get standard quality tiers."""
        return [
            cls.HIGH,
            cls.GOOD,
            cls.STANDARD,
            cls.ACCEPTABLE
        ]
    
    @classmethod
    def get_lower_tiers(cls):
        """Get lower quality tiers."""
        return [
            cls.FAIR,
            cls.POOR,
            cls.SUBSTANDARD
        ]
    
    @classmethod
    def get_rejection_tiers(cls):
        """Get rejection quality tiers."""
        return [
            cls.UNACCEPTABLE,
            cls.REJECTED,
            cls.FAILED
        ]
    
    @classmethod
    def get_special_tiers(cls):
        """Get special status tiers."""
        return [
            cls.PENDING,
            cls.UNDER_REVIEW,
            cls.REQUIRES_RETEST
        ]
    
    def get_numeric_score(self):
        """Get numeric score for this quality tier (0-100)."""
        score_map = {
            cls.OUTSTANDING: 100,
            cls.EXCELLENT: 95,
            cls.PREMIUM: 90,
            cls.HIGH: 85,
            cls.GOOD: 80,
            cls.STANDARD: 75,
            cls.ACCEPTABLE: 70,
            cls.FAIR: 60,
            cls.POOR: 45,
            cls.SUBSTANDARD: 30,
            cls.UNACCEPTABLE: 15,
            cls.REJECTED: 10,
            cls.FAILED: 5,
            cls.PENDING: 0,
            cls.UNDER_REVIEW: 0,
            cls.REQUIRES_RETEST: 0,
        }
        return score_map.get(self, 0)
    
    def get_grade(self):
        """Get letter grade for this quality tier."""
        grade_map = {
            cls.OUTSTANDING: "A+",
            cls.EXCELLENT: "A",
            cls.PREMIUM: "A-",
            cls.HIGH: "B+",
            cls.GOOD: "B",
            cls.STANDARD: "B-",
            cls.ACCEPTABLE: "C+",
            cls.FAIR: "C",
            cls.POOR: "D",
            cls.SUBSTANDARD: "D-",
            cls.UNACCEPTABLE: "F",
            cls.REJECTED: "F",
            cls.FAILED: "F",
            cls.PENDING: "N/A",
            cls.UNDER_REVIEW: "N/A",
            cls.REQUIRES_RETEST: "N/A",
        }
        return grade_map.get(self, "N/A")
    
    def is_premium(self):
        """Check if this is a premium quality tier."""
        return self in self.get_premium_tiers()
    
    def is_standard(self):
        """Check if this is a standard quality tier."""
        return self in self.get_standard_tiers()
    
    def is_lower(self):
        """Check if this is a lower quality tier."""
        return self in self.get_lower_tiers()
    
    def is_rejected(self):
        """Check if this is a rejection tier."""
        return self in self.get_rejection_tiers()
    
    def is_special(self):
        """Check if this is a special status tier."""
        return self in self.get_special_tiers()
    
    def is_acceptable(self):
        """Check if this tier meets minimum quality standards."""
        return self.get_numeric_score() >= 70
    
    def is_production_ready(self):
        """Check if this tier is suitable for production."""
        return self.get_numeric_score() >= 75
    
    def is_customer_ready(self):
        """Check if this tier is suitable for customer delivery."""
        return self.get_numeric_score() >= 80
    
    def requires_attention(self):
        """Check if this tier requires quality attention."""
        return self.get_numeric_score() < 70 or self.is_special()
    
    def can_upgrade_to(self, target_tier):
        """Check if upgrade to target tier is possible."""
        if self.is_special():
            return True  # Special tiers can always be upgraded
        
        return target_tier.get_numeric_score() > self.get_numeric_score()
    
    def get_improvement_suggestions(self):
        """Get improvement suggestions for this quality tier."""
        suggestions = {
            cls.OUTSTANDING: ["Maintain current standards", "Document best practices"],
            cls.EXCELLENT: ["Optimize process parameters", "Reduce minor variations"],
            cls.PREMIUM: ["Fine-tune process settings", "Improve material handling"],
            cls.HIGH: ["Review process parameters", "Check equipment calibration"],
            cls.GOOD: ["Optimize laser parameters", "Improve powder quality"],
            cls.STANDARD: ["Adjust process settings", "Review quality procedures"],
            cls.ACCEPTABLE: ["Major process review needed", "Check equipment status"],
            cls.FAIR: ["Process optimization required", "Equipment maintenance needed"],
            cls.POOR: ["Significant process changes needed", "Equipment overhaul required"],
            cls.SUBSTANDARD: ["Complete process redesign", "Equipment replacement needed"],
            cls.UNACCEPTABLE: ["Process not viable", "Major system changes required"],
            cls.REJECTED: ["Process failure", "Investigate root causes"],
            cls.FAILED: ["Process failure", "Complete system review needed"],
        }
        return suggestions.get(self, ["Review quality metrics", "Investigate issues"])