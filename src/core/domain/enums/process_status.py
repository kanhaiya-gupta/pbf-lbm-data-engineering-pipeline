"""
Process status enumeration for PBF-LB/M operations.
"""

from enum import Enum


class ProcessStatus(Enum):
    """
    Enumeration for PBF process status states.
    
    This enum defines the various states a PBF process can be in
    throughout its lifecycle from initialization to completion.
    """
    
    # Initial states
    INITIALIZED = "initialized"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    
    # Active states
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"
    
    # Completion states
    COMPLETED = "completed"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    
    # Quality states
    QUALITY_CHECK = "quality_check"
    QUALITY_PASSED = "quality_passed"
    QUALITY_FAILED = "quality_failed"
    
    # Post-processing states
    POST_PROCESSING = "post_processing"
    ANALYZING = "analyzing"
    ARCHIVING = "archiving"
    
    @classmethod
    def get_active_states(cls):
        """Get states where the process is actively running."""
        return [
            cls.PREPARING,
            cls.RUNNING,
            cls.PAUSED,
            cls.RESUMING,
            cls.QUALITY_CHECK,
            cls.POST_PROCESSING,
            cls.ANALYZING
        ]
    
    @classmethod
    def get_final_states(cls):
        """Get terminal states where the process has ended."""
        return [
            cls.COMPLETED,
            cls.SUCCESS,
            cls.FAILED,
            cls.CANCELLED,
            cls.ABORTED,
            cls.QUALITY_FAILED,
            cls.ARCHIVING
        ]
    
    @classmethod
    def get_success_states(cls):
        """Get states indicating successful completion."""
        return [
            cls.COMPLETED,
            cls.SUCCESS,
            cls.QUALITY_PASSED
        ]
    
    @classmethod
    def get_error_states(cls):
        """Get states indicating errors or failures."""
        return [
            cls.FAILED,
            cls.CANCELLED,
            cls.ABORTED,
            cls.QUALITY_FAILED
        ]
    
    def is_active(self):
        """Check if the process is in an active state."""
        return self in self.get_active_states()
    
    def is_final(self):
        """Check if the process is in a final state."""
        return self in self.get_final_states()
    
    def is_success(self):
        """Check if the process completed successfully."""
        return self in self.get_success_states()
    
    def is_error(self):
        """Check if the process ended in an error state."""
        return self in self.get_error_states()
    
    def can_transition_to(self, target_status):
        """Check if transition to target status is valid."""
        valid_transitions = {
            self.INITIALIZED: [self.PENDING, self.SCHEDULED, self.CANCELLED],
            self.PENDING: [self.SCHEDULED, self.PREPARING, self.CANCELLED],
            self.SCHEDULED: [self.PREPARING, self.CANCELLED],
            self.PREPARING: [self.RUNNING, self.FAILED, self.CANCELLED],
            self.RUNNING: [self.PAUSED, self.COMPLETED, self.FAILED, self.ABORTED],
            self.PAUSED: [self.RESUMING, self.CANCELLED, self.ABORTED],
            self.RESUMING: [self.RUNNING, self.FAILED],
            self.COMPLETED: [self.QUALITY_CHECK, self.POST_PROCESSING],
            self.QUALITY_CHECK: [self.QUALITY_PASSED, self.QUALITY_FAILED],
            self.QUALITY_PASSED: [self.SUCCESS, self.POST_PROCESSING],
            self.QUALITY_FAILED: [self.FAILED, self.ANALYZING],
            self.POST_PROCESSING: [self.ANALYZING, self.ARCHIVING],
            self.ANALYZING: [self.ARCHIVING, self.SUCCESS, self.FAILED],
            self.ARCHIVING: [self.SUCCESS],
            self.SUCCESS: [],  # Terminal state
            self.FAILED: [],   # Terminal state
            self.CANCELLED: [], # Terminal state
            self.ABORTED: [],  # Terminal state
        }
        return target_status in valid_transitions.get(self, [])