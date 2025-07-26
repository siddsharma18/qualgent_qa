import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

class QALogger:
    """
    Enhanced structured logger for QA system with robust error handling and validation.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 log_file: Optional[str] = None,
                 enable_json: bool = True,
                 max_log_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize the QA logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Whether to enable console logging
            log_file: Path to log file
            enable_json: Whether to enable JSON logging
            max_log_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
        """
        self.log_level = self._validate_log_level(log_level)
        self.enable_console = enable_console
        self.log_file = log_file
        self.enable_json = enable_json
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        
        # Initialize logging
        self._setup_logging()
        
        # Track statistics
        self.stats = {
            'total_logs': 0,
            'error_logs': 0,
            'warning_logs': 0,
            'info_logs': 0,
            'debug_logs': 0
        }
    
    def _validate_log_level(self, level: str) -> str:
        """Validate and normalize log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level_upper = level.upper()
        
        if level_upper not in valid_levels:
            print(f"⚠️  Invalid log level '{level}', using 'INFO'")
            return 'INFO'
        
        return level_upper
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            # Create logs directory if it doesn't exist
            if self.log_file:
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=self._create_handlers()
            )
            
            self.logger = logging.getLogger('qa_system')
            
        except Exception as e:
            print(f"❌ Failed to setup logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('qa_system')
    
    def _create_handlers(self) -> List[logging.Handler]:
        """Create logging handlers."""
        handlers = []
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.log_level))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        # File handler
        if self.log_file:
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_log_size,
                    backupCount=self.backup_count
                )
                file_handler.setLevel(getattr(logging, self.log_level))
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                handlers.append(file_handler)
            except Exception as e:
                print(f"⚠️  Failed to create file handler: {e}")
        
        return handlers
    
    def _safe_log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Safely log a message with error handling."""
        try:
            log_method = getattr(self.logger, level.lower())
            
            if extra and self.enable_json:
                # Add JSON context to message
                json_context = json.dumps(extra, default=str)
                full_message = f"{message} | Context: {json_context}"
            else:
                full_message = message
            
            log_method(full_message)
            
            # Update statistics
            self.stats['total_logs'] += 1
            if level.upper() in self.stats:
                self.stats[f'{level.lower()}_logs'] += 1
            
        except Exception as e:
            # Fallback logging
            print(f"❌ Logging error: {e}")
            print(f"Original message: {message}")
    
    def info(self, component: str, message: str, **kwargs) -> None:
        """Log info message."""
        formatted_message = f"[{component}] {message}"
        self._safe_log('INFO', formatted_message, kwargs)
    
    def warning(self, component: str, message: str, **kwargs) -> None:
        """Log warning message."""
        formatted_message = f"[{component}] {message}"
        self._safe_log('WARNING', formatted_message, kwargs)
    
    def error(self, component: str, message: str, **kwargs) -> None:
        """Log error message."""
        formatted_message = f"[{component}] {message}"
        self._safe_log('ERROR', formatted_message, kwargs)
    
    def debug(self, component: str, message: str, **kwargs) -> None:
        """Log debug message."""
        formatted_message = f"[{component}] {message}"
        self._safe_log('DEBUG', formatted_message, kwargs)
    
    def critical(self, component: str, message: str, **kwargs) -> None:
        """Log critical message."""
        formatted_message = f"[{component}] {message}"
        self._safe_log('CRITICAL', formatted_message, kwargs)
    
    # Backward compatibility method
    def log(self, agent: str, action: str, data: Dict[str, Any], level: str = "INFO") -> None:
        """Backward compatibility method for old logging calls."""
        message = f"{action}: {data}"
        if level.upper() == "INFO":
            self.info(agent, message, **data)
        elif level.upper() == "WARNING":
            self.warning(agent, message, **data)
        elif level.upper() == "ERROR":
            self.error(agent, message, **data)
        elif level.upper() == "DEBUG":
            self.debug(agent, message, **data)
        elif level.upper() == "CRITICAL":
            self.critical(agent, message, **data)
    
    def log_execution_start(self, subgoal: str, ui_tree_size: int, **kwargs) -> None:
        """Log execution start with context."""
        self.info("Executor", f"Starting execution: {subgoal}", 
                 subgoal=subgoal, ui_tree_size=ui_tree_size, **kwargs)
    
    def log_execution_end(self, subgoal: str, status: str, execution_time: float, **kwargs) -> None:
        """Log execution end with results."""
        self.info("Executor", f"Execution completed: {subgoal} - {status}", 
                 subgoal=subgoal, status=status, execution_time=execution_time, **kwargs)
    
    def log_verification_start(self, subgoal: str, **kwargs) -> None:
        """Log verification start."""
        self.info("Verifier", f"Starting verification: {subgoal}", 
                 subgoal=subgoal, **kwargs)
    
    def log_verification_end(self, subgoal: str, status: str, confidence: float, **kwargs) -> None:
        """Log verification end with results."""
        self.info("Verifier", f"Verification completed: {subgoal} - {status} (confidence: {confidence:.2f})", 
                 subgoal=subgoal, status=status, confidence=confidence, **kwargs)
    
    def log_planning_start(self, goal: str, **kwargs) -> None:
        """Log planning start."""
        self.info("Planner", f"Starting planning: {goal}", 
                 goal=goal, **kwargs)
    
    def log_planning_end(self, goal: str, status: str, subgoal_count: int, **kwargs) -> None:
        """Log planning end with results."""
        self.info("Planner", f"Planning completed: {goal} - {status} ({subgoal_count} subgoals)", 
                 goal=goal, status=status, subgoal_count=subgoal_count, **kwargs)
    
    def log_supervision_start(self, test_type: str, **kwargs) -> None:
        """Log supervision start."""
        self.info("Supervisor", f"Starting supervision: {test_type}", 
                 test_type=test_type, **kwargs)
    
    def log_supervision_end(self, test_type: str, success_rate: float, **kwargs) -> None:
        """Log supervision end with results."""
        self.info("Supervisor", f"Supervision completed: {test_type} - Success rate: {success_rate:.2f}", 
                 test_type=test_type, success_rate=success_rate, **kwargs)
    
    def log_element_found(self, element_id: str, confidence: float, strategy: str) -> None:
        """Log when an element is found."""
        self.info("Executor", "Element found", 
                 element_id=element_id, confidence=confidence, strategy=strategy)
    
    def log_element_not_found(self, subgoal: str, strategies_tried: List[str]) -> None:
        """Log when no element is found."""
        self.warning("Executor", "Element not found", 
                    subgoal=subgoal, strategies_tried=strategies_tried)
    
    def log_action_executed(self, action: Dict[str, Any], result_status: str) -> None:
        """Log action execution result."""
        self.info("Executor", "Action executed", 
                 action=action, result_status=result_status)
    
    def log_retry_attempt(self, attempt: int, max_attempts: int, reason: str) -> None:
        """Log retry attempt."""
        self.warning("Executor", "Retry attempt", 
                    attempt=attempt, max_attempts=max_attempts, reason=reason)
    
    def log_json(self, data: Dict[str, Any], filename: Optional[str] = None) -> None:
        """Log data as JSON file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qa_logs_{timestamp}.json"
            
            # Ensure logs directory exists
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            filepath = os.path.join(log_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.info("Logger", f"JSON log saved: {filepath}")
            
        except Exception as e:
            self.error("Logger", f"Failed to save JSON log: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'logger_stats': self.stats.copy(),
            'log_level': self.log_level,
            'enable_console': self.enable_console,
            'log_file': self.log_file,
            'enable_json': self.enable_json
        }
    
    def reset_stats(self) -> None:
        """Reset logging statistics."""
        self.stats = {
            'total_logs': 0,
            'error_logs': 0,
            'warning_logs': 0,
            'info_logs': 0,
            'debug_logs': 0
        }
    
    def validate_log_file(self) -> bool:
        """Validate log file accessibility."""
        if not self.log_file:
            return True
        
        try:
            # Test write access
            with open(self.log_file, 'a') as f:
                f.write("")
            return True
        except Exception:
            return False
