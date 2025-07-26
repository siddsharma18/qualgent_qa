import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from utils.ui_parser import find_element_for_subgoal, UIParser
from utils.logger import QALogger

@dataclass
class ExecutionResult:
    """Result of subgoal execution."""
    status: str  # 'success', 'fail', 'retry', 'timeout'
    observation: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    attempts: int = 0
    element_id: Optional[str] = None
    confidence: float = 0.0
    execution_time: float = 0.0

class ExecutorAgent:
    """
    A robust executor agent that handles subgoal execution with multiple fallback strategies,
    retry mechanisms, and comprehensive error handling.
    """
    
    def __init__(self, 
                 env, 
                 logger: Optional[QALogger] = None,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 action_timeout: float = 10.0,
                 ui_settle_time: float = 1.5,
                 enable_validation: bool = True,
                 min_confidence: float = 0.3):
        """
        Initialize the executor agent.
        
        Args:
            env: The AndroidEnv instance
            logger: Optional logger for recording actions
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            action_timeout: Timeout for action execution in seconds
            ui_settle_time: Time to wait for UI to settle after action
            enable_validation: Whether to validate actions before execution
            min_confidence: Minimum confidence threshold for element matching
        """
        self.env = env
        self.logger = logger or QALogger()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.action_timeout = action_timeout
        self.ui_settle_time = ui_settle_time
        self.enable_validation = enable_validation
        self.min_confidence = min_confidence
        
        # Initialize UI parser with custom settings
        self.ui_parser = UIParser(min_confidence=min_confidence)
        
        # Track execution statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'retry_attempts': 0,
            'average_execution_time': 0.0
        }
    
    def execute(self, subgoal: str, ui_tree: List[Dict[str, Any]]) -> ExecutionResult:
        """
        Execute a subgoal with robust error handling and retry mechanisms.
        
        Args:
            subgoal: A string like "Turn Wi-Fi off"
            ui_tree: The current UI observation tree from env
            
        Returns:
            ExecutionResult: Detailed result of the execution
        """
        start_time = time.time()
        self.stats['total_executions'] += 1
        
        # Log execution start
        self.logger.log_execution_start(subgoal, len(ui_tree))
        
        # Validate inputs
        if not subgoal or not subgoal.strip():
            return self._create_failure_result("Empty or invalid subgoal", start_time)
        
        if not ui_tree:
            return self._create_failure_result("Empty UI tree", start_time)
        
        # Try to find the target element
        element_id = self._find_element_with_retry(subgoal, ui_tree)
        
        if element_id is None:
            return self._create_failure_result(
                f"No matching UI element found for subgoal: '{subgoal}'", 
                start_time
            )
        
        # Execute the action with retry mechanism
        result = self._execute_action_with_retry(element_id, subgoal, start_time)
        
        # Update statistics
        self._update_stats(result, time.time() - start_time)
        
        return result
    
    def _find_element_with_retry(self, subgoal: str, ui_tree: List[Dict[str, Any]]) -> Optional[str]:
        """Find element with retry mechanism using different strategies."""
        strategies = [
            lambda: self.ui_parser.find_element_for_subgoal(ui_tree, subgoal),
            lambda: self._find_element_with_variations(subgoal, ui_tree),
            lambda: self._find_element_with_context(subgoal, ui_tree)
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                element_id = strategy()
                if element_id:
                    self.logger.log_element_found(element_id, 0.8 - i * 0.2, f"strategy_{i}")
                    return element_id
            except Exception as e:
                self.logger.warning("Executor", "Element finding strategy failed", 
                                  strategy=i,
                                  error=str(e))
        
        self.logger.log_element_not_found(subgoal, [f"strategy_{i}" for i in range(len(strategies))])
        return None
    
    def _find_element_with_variations(self, subgoal: str, ui_tree: List[Dict[str, Any]]) -> Optional[str]:
        """Find element by trying variations of the subgoal."""
        variations = self._generate_subgoal_variations(subgoal)
        
        for variation in variations:
            element_id = self.ui_parser.find_element_for_subgoal(ui_tree, variation)
            if element_id:
                return element_id
        
        return None
    
    def _find_element_with_context(self, subgoal: str, ui_tree: List[Dict[str, Any]]) -> Optional[str]:
        """Find element using contextual information."""
        # Extract key terms from subgoal
        key_terms = self._extract_key_terms(subgoal)
        
        # Look for elements that contain any of the key terms
        for element in ui_tree:
            text = self._extract_element_text(element)
            if text and any(term.lower() in text.lower() for term in key_terms):
                return element.get('id')
        
        return None
    
    def _execute_action_with_retry(self, element_id: str, subgoal: str, start_time: float) -> ExecutionResult:
        """Execute action with retry mechanism."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Validate action before execution
                if self.enable_validation:
                    validation_result = self._validate_action(element_id, subgoal)
                    if not validation_result['valid']:
                        raise ValueError(validation_result['reason'])
                
                # Create action
                action = self._create_action(element_id, subgoal)
                
                # Log action execution
                self.logger.info("Executor", "Executing action", 
                               attempt=attempt + 1,
                               action=action,
                               element_id=element_id)
                
                # Execute action with timeout
                result = self._execute_action_with_timeout(action)
                
                # Wait for UI to settle
                time.sleep(self.ui_settle_time)
                
                # Validate result
                if self._validate_result(result, subgoal):
                    self.logger.log_action_executed(action, "success")
                    return ExecutionResult(
                        status="success",
                        observation=result,
                        element_id=element_id,
                        attempts=attempt + 1,
                        execution_time=time.time() - start_time
                    )
                else:
                    raise ValueError("Action did not produce expected result")
                
            except Exception as e:
                last_error = str(e)
                self.stats['retry_attempts'] += 1
                
                if attempt < self.max_retries:
                    self.logger.log_retry_attempt(attempt + 1, self.max_retries, last_error)
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error("Executor", "Max retries exceeded", 
                                    error=last_error,
                                    attempts=attempt + 1)
        
        return ExecutionResult(
            status="fail",
            reason=f"Failed after {self.max_retries + 1} attempts: {last_error}",
            element_id=element_id,
            attempts=self.max_retries + 1,
            execution_time=time.time() - start_time
        )
    
    def _execute_action_with_timeout(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Action execution timed out")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.action_timeout))
        
        try:
            result = self.env.step(action)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            raise
    
    def _create_action(self, element_id: str, subgoal: str) -> Dict[str, Any]:
        """Create an action based on the element and subgoal."""
        # Determine action type based on subgoal
        action_type = self._determine_action_type(subgoal)
        
        action = {
            "action_type": action_type,
            "element_id": element_id
        }
        
        # Add additional parameters based on action type
        if action_type == "type":
            action["text"] = self._extract_text_from_subgoal(subgoal)
        elif action_type == "scroll":
            action["direction"] = self._determine_scroll_direction(subgoal)
        
        return action
    
    def _determine_action_type(self, subgoal: str) -> str:
        """Determine the appropriate action type from the subgoal."""
        subgoal_lower = subgoal.lower()
        
        if any(word in subgoal_lower for word in ['type', 'enter', 'input', 'write']):
            return "type"
        elif any(word in subgoal_lower for word in ['scroll', 'swipe', 'move']):
            return "scroll"
        elif any(word in subgoal_lower for word in ['back', 'return', 'previous']):
            return "back"
        elif any(word in subgoal_lower for word in ['home', 'main']):
            return "home"
        else:
            return "touch"  # Default action
    
    def _validate_action(self, element_id: str, subgoal: str) -> Dict[str, Any]:
        """Validate action before execution."""
        # Check if element exists in current UI
        current_ui = self.env.get_observation() if hasattr(self.env, 'get_observation') else None
        
        if current_ui and element_id not in [elem.get('id') for elem in current_ui.get('ui_tree', [])]:
            return {
                'valid': False,
                'reason': f"Element {element_id} not found in current UI"
            }
        
        return {'valid': True}
    
    def _validate_result(self, result: Dict[str, Any], subgoal: str) -> bool:
        """Validate the result of action execution."""
        # Basic validation - check if result contains expected fields
        if not result or not isinstance(result, dict):
            return False
        
        # Check if we have a new observation
        if 'observation' not in result:
            return False
        
        # Additional validation can be added here based on subgoal type
        return True
    
    def _generate_subgoal_variations(self, subgoal: str) -> List[str]:
        """Generate variations of the subgoal for better matching."""
        variations = [subgoal]
        
        # Common variations
        if "wifi" in subgoal.lower():
            variations.extend([
                subgoal.replace("wifi", "Wi-Fi"),
                subgoal.replace("wifi", "wireless"),
                subgoal.replace("wifi", "network")
            ])
        
        if "bluetooth" in subgoal.lower():
            variations.extend([
                subgoal.replace("bluetooth", "Bluetooth"),
                subgoal.replace("bluetooth", "BT")
            ])
        
        return variations
    
    def _extract_key_terms(self, subgoal: str) -> List[str]:
        """Extract key terms from subgoal for contextual matching."""
        # Simple key term extraction
        terms = subgoal.lower().split()
        return [term for term in terms if len(term) > 2]
    
    def _extract_element_text(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract text from UI element."""
        text_fields = ['text', 'content-desc', 'label', 'title']
        for field in text_fields:
            if field in element and element[field]:
                return str(element[field])
        return None
    
    def _extract_text_from_subgoal(self, subgoal: str) -> str:
        """Extract text to type from subgoal."""
        # Simple text extraction - can be enhanced with NLP
        import re
        # Look for quoted text
        matches = re.findall(r'"([^"]*)"', subgoal)
        if matches:
            return matches[0]
        
        # Look for text after "type" or "enter"
        if "type" in subgoal.lower():
            parts = subgoal.lower().split("type")
            if len(parts) > 1:
                return parts[1].strip()
        
        return ""
    
    def _determine_scroll_direction(self, subgoal: str) -> str:
        """Determine scroll direction from subgoal."""
        subgoal_lower = subgoal.lower()
        
        if any(word in subgoal_lower for word in ['up', 'top', 'north']):
            return "up"
        elif any(word in subgoal_lower for word in ['down', 'bottom', 'south']):
            return "down"
        elif any(word in subgoal_lower for word in ['left', 'west']):
            return "left"
        elif any(word in subgoal_lower for word in ['right', 'east']):
            return "right"
        else:
            return "down"  # Default direction
    
    def _create_failure_result(self, reason: str, start_time: float) -> ExecutionResult:
        """Create a failure result."""
        return ExecutionResult(
            status="fail",
            reason=reason,
            execution_time=time.time() - start_time
        )
    
    def _update_stats(self, result: ExecutionResult, execution_time: float):
        """Update execution statistics."""
        if result.status == "success":
            self.stats['successful_executions'] += 1
        else:
            self.stats['failed_executions'] += 1
        
        # Update average execution time
        total_executions = self.stats['successful_executions'] + self.stats['failed_executions']
        if total_executions > 0:
            self.stats['average_execution_time'] = (
                (self.stats['average_execution_time'] * (total_executions - 1) + execution_time) 
                / total_executions
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset execution statistics."""
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'retry_attempts': 0,
            'average_execution_time': 0.0
        }