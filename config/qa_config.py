"""
Configuration settings for the QA system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ExecutorConfig:
    """Configuration for the executor agent."""
    max_retries: int = 3
    retry_delay: float = 2.0
    action_timeout: float = 10.0
    ui_settle_time: float = 1.5
    enable_validation: bool = True
    min_confidence: float = 0.3
    enable_fuzzy_matching: bool = True
    enable_semantic_matching: bool = True

@dataclass
class VerifierConfig:
    """Configuration for the verifier agent."""
    min_change_ratio: float = 0.1
    fuzzy_threshold: float = 0.7
    max_verification_time: float = 10.0
    enable_advanced_analysis: bool = True
    enable_state_tracking: bool = True
    min_confidence: float = 0.3
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'ui_change_verification': 1.0,
        'subgoal_presence_verification': 0.8,
        'state_transition_verification': 0.9,
        'element_interaction_verification': 0.7,
        'semantic_verification': 0.6
    })

@dataclass
class PlannerConfig:
    """Configuration for the planner agent."""
    enable_template_matching: bool = True
    enable_semantic_planning: bool = True
    enable_adaptive_planning: bool = True
    max_planning_time: float = 30.0
    min_confidence: float = 0.5
    enable_plan_optimization: bool = True
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'template_based_planning': 1.0,
        'semantic_planning': 0.8,
        'adaptive_planning': 0.9,
        'fallback_planning': 0.6
    })

@dataclass
class LoggerConfig:
    """Configuration for the logger."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_console: bool = True
    enable_json: bool = False

@dataclass
class UIConfig:
    """Configuration for UI parsing."""
    min_confidence: float = 0.3
    enable_fuzzy_matching: bool = True
    enable_semantic_matching: bool = True
    max_element_search_depth: int = 10

@dataclass
class QASystemConfig:
    """Main configuration for the QA system."""
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Environment settings
    env_timeout: float = 30.0
    max_consecutive_failures: int = 5
    
    # Performance settings
    enable_performance_monitoring: bool = True
    enable_statistics: bool = True
    
    # Safety settings
    enable_safety_checks: bool = True
    max_actions_per_session: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'executor': {
                'max_retries': self.executor.max_retries,
                'retry_delay': self.executor.retry_delay,
                'action_timeout': self.executor.action_timeout,
                'ui_settle_time': self.executor.ui_settle_time,
                'enable_validation': self.executor.enable_validation,
                'min_confidence': self.executor.min_confidence,
                'enable_fuzzy_matching': self.executor.enable_fuzzy_matching,
                'enable_semantic_matching': self.executor.enable_semantic_matching
            },
            'verifier': {
                'min_change_ratio': self.verifier.min_change_ratio,
                'fuzzy_threshold': self.verifier.fuzzy_threshold,
                'max_verification_time': self.verifier.max_verification_time,
                'enable_advanced_analysis': self.verifier.enable_advanced_analysis,
                'enable_state_tracking': self.verifier.enable_state_tracking,
                'min_confidence': self.verifier.min_confidence,
                'strategy_weights': self.verifier.strategy_weights
            },
            'planner': {
                'enable_template_matching': self.planner.enable_template_matching,
                'enable_semantic_planning': self.planner.enable_semantic_planning,
                'enable_adaptive_planning': self.planner.enable_adaptive_planning,
                'max_planning_time': self.planner.max_planning_time,
                'min_confidence': self.planner.min_confidence,
                'enable_plan_optimization': self.planner.enable_plan_optimization,
                'strategy_weights': self.planner.strategy_weights
            },
            'logger': {
                'log_level': self.logger.log_level,
                'log_file': self.logger.log_file,
                'enable_console': self.logger.enable_console,
                'enable_json': self.logger.enable_json
            },
            'ui': {
                'min_confidence': self.ui.min_confidence,
                'enable_fuzzy_matching': self.ui.enable_fuzzy_matching,
                'enable_semantic_matching': self.ui.enable_semantic_matching,
                'max_element_search_depth': self.ui.max_element_search_depth
            },
            'env_timeout': self.env_timeout,
            'max_consecutive_failures': self.max_consecutive_failures,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_statistics': self.enable_statistics,
            'enable_safety_checks': self.enable_safety_checks,
            'max_actions_per_session': self.max_actions_per_session
        }

# Default configuration
DEFAULT_CONFIG = QASystemConfig()

# High-performance configuration
HIGH_PERFORMANCE_CONFIG = QASystemConfig(
    executor=ExecutorConfig(
        max_retries=1,
        retry_delay=1.0,
        action_timeout=5.0,
        ui_settle_time=0.5,
        min_confidence=0.5
    ),
    verifier=VerifierConfig(
        min_change_ratio=0.05,
        fuzzy_threshold=0.8,
        max_verification_time=5.0,
        min_confidence=0.6
    ),
    planner=PlannerConfig(
        max_planning_time=15.0,
        min_confidence=0.6,
        enable_plan_optimization=True
    ),
    logger=LoggerConfig(log_level="WARNING"),
    ui=UIConfig(min_confidence=0.5)
)

# High-reliability configuration
HIGH_RELIABILITY_CONFIG = QASystemConfig(
    executor=ExecutorConfig(
        max_retries=5,
        retry_delay=3.0,
        action_timeout=15.0,
        ui_settle_time=2.0,
        min_confidence=0.2
    ),
    verifier=VerifierConfig(
        min_change_ratio=0.15,
        fuzzy_threshold=0.6,
        max_verification_time=15.0,
        min_confidence=0.4,
        enable_advanced_analysis=True,
        enable_state_tracking=True
    ),
    planner=PlannerConfig(
        max_planning_time=45.0,
        min_confidence=0.4,
        enable_plan_optimization=True,
        enable_adaptive_planning=True
    ),
    logger=LoggerConfig(log_level="DEBUG"),
    ui=UIConfig(min_confidence=0.2)
)

def get_config(config_name: str = "default") -> QASystemConfig:
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration ("default", "high_performance", "high_reliability")
        
    Returns:
        QASystemConfig: The requested configuration
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "high_performance": HIGH_PERFORMANCE_CONFIG,
        "high_reliability": HIGH_RELIABILITY_CONFIG
    }
    
    return configs.get(config_name, DEFAULT_CONFIG)
