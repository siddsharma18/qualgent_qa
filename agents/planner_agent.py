from typing import List, Dict, Any, Optional, Tuple, Set
import re
import time
import logging
from dataclasses import dataclass
from enum import Enum
from utils.logger import QALogger

class PlanningStatus(Enum):
    """Planning status enumeration."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class Subgoal:
    """Represents a subgoal with detailed information."""
    name: str
    description: str
    priority: int
    dependencies: List[str]
    estimated_duration: float
    required_elements: List[str]
    alternative_approaches: List[str]
    confidence: float
    retry_strategy: str = "standard"
    max_retries: int = 3
    fallback_subgoals: List[str] = None

    def __post_init__(self):
        if self.fallback_subgoals is None:
            self.fallback_subgoals = []

@dataclass
class PlanningResult:
    """Result of planning with detailed information."""
    status: PlanningStatus
    subgoals: List[Subgoal]
    total_estimated_duration: float
    planning_time: float
    strategies_used: List[str]
    confidence: float
    plan_complexity: str
    alternative_plans: List[List[Subgoal]]
    risk_assessment: Dict[str, float] = None
    stability_score: float = 0.0

    def __post_init__(self):
        if self.risk_assessment is None:
            self.risk_assessment = {}

class PlannerAgent:
    """
    A robust planner agent that generates UI-specific subgoals from high-level user goals
    with multiple planning strategies, comprehensive error handling, and advanced planning capabilities.
    """

    def __init__(self, 
                 logger: Optional[QALogger] = None,
                 enable_template_matching: bool = True,
                 enable_semantic_planning: bool = True,
                 enable_adaptive_planning: bool = True,
                 enable_fallback_planning: bool = True,
                 enable_confidence_weighting: bool = True,
                 min_confidence_threshold: float = 0.4,
                 planning_timeout: float = 30.0,
                 enable_stability_scoring: bool = True,
                 enable_risk_assessment: bool = True):
        """
        Initialize the planner agent.
        
        Args:
            logger: Optional logger for recording planning activities
            enable_template_matching: Whether to use template-based planning
            enable_semantic_planning: Whether to use semantic understanding for planning
            enable_adaptive_planning: Whether to adapt plans based on context
            max_planning_time: Maximum time to spend on planning
            min_confidence: Minimum confidence threshold for planning
            enable_plan_optimization: Whether to optimize plans for efficiency
        """
        self.logger = logger or QALogger()
        self.enable_template_matching = enable_template_matching
        self.enable_semantic_planning = enable_semantic_planning
        self.enable_adaptive_planning = enable_adaptive_planning
        self.enable_fallback_planning = enable_fallback_planning
        self.enable_confidence_weighting = enable_confidence_weighting
        self.min_confidence_threshold = min_confidence_threshold
        self.planning_timeout = planning_timeout
        self.enable_stability_scoring = enable_stability_scoring
        self.enable_risk_assessment = enable_risk_assessment
        self.enable_plan_optimization = True  # Always enable for compatibility
        self.min_confidence = min_confidence_threshold  # Alias for backward compatibility
        
        # Track planning statistics
        self.stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'average_planning_time': 0.0,
            'strategy_success_rates': {},
            'plan_complexity_distribution': {}
        }
        
        # Planning history
        self.planning_history = []
        
        # Comprehensive planning templates
        self.planning_templates = {
            'wifi_management': {
                'turn_off_wifi': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Quick Settings"], 0.9),
                    Subgoal("Navigate to Network", "Find and open Network & Internet settings", 2, ["Open Settings"], 3.0, ["network_option"], ["Wi-Fi settings"], 0.8),
                    Subgoal("Access Wi-Fi Settings", "Open Wi-Fi configuration page", 3, ["Navigate to Network"], 2.0, ["wifi_option"], ["Wi-Fi toggle"], 0.9),
                    Subgoal("Disable Wi-Fi", "Turn off Wi-Fi connection", 4, ["Access Wi-Fi Settings"], 1.0, ["wifi_toggle"], ["Quick Settings"], 0.95)
                ],
                'turn_on_wifi': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Quick Settings"], 0.9),
                    Subgoal("Navigate to Network", "Find and open Network & Internet settings", 2, ["Open Settings"], 3.0, ["network_option"], ["Wi-Fi settings"], 0.8),
                    Subgoal("Access Wi-Fi Settings", "Open Wi-Fi configuration page", 3, ["Navigate to Network"], 2.0, ["wifi_option"], ["Wi-Fi toggle"], 0.9),
                    Subgoal("Enable Wi-Fi", "Turn on Wi-Fi connection", 4, ["Access Wi-Fi Settings"], 1.0, ["wifi_toggle"], ["Quick Settings"], 0.95)
                ]
            },
            'bluetooth_management': {
                'enable_bluetooth': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Quick Settings"], 0.9),
                    Subgoal("Navigate to Connected Devices", "Find Bluetooth and device settings", 2, ["Open Settings"], 3.0, ["connected_devices"], ["Bluetooth settings"], 0.8),
                    Subgoal("Access Bluetooth Settings", "Open Bluetooth configuration", 3, ["Navigate to Connected Devices"], 2.0, ["bluetooth_option"], ["Bluetooth toggle"], 0.9),
                    Subgoal("Enable Bluetooth", "Turn on Bluetooth connection", 4, ["Access Bluetooth Settings"], 1.0, ["bluetooth_toggle"], ["Quick Settings"], 0.95)
                ],
                'disable_bluetooth': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Quick Settings"], 0.9),
                    Subgoal("Navigate to Connected Devices", "Find Bluetooth and device settings", 2, ["Open Settings"], 3.0, ["connected_devices"], ["Bluetooth settings"], 0.8),
                    Subgoal("Access Bluetooth Settings", "Open Bluetooth configuration", 3, ["Navigate to Connected Devices"], 2.0, ["bluetooth_option"], ["Bluetooth toggle"], 0.9),
                    Subgoal("Disable Bluetooth", "Turn off Bluetooth connection", 4, ["Access Bluetooth Settings"], 1.0, ["bluetooth_toggle"], ["Quick Settings"], 0.95)
                ]
            },
            'developer_options': {
                'enable_developer_options': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], [], 0.9),
                    Subgoal("Navigate to About Phone", "Find device information section", 2, ["Open Settings"], 3.0, ["about_phone"], ["System"], 0.8),
                    Subgoal("Access Build Number", "Find and tap build number multiple times", 3, ["Navigate to About Phone"], 5.0, ["build_number"], ["Build info"], 0.7),
                    Subgoal("Return to Settings", "Go back to main settings menu", 4, ["Access Build Number"], 2.0, ["back_button"], ["Home"], 0.9),
                    Subgoal("Open Developer Options", "Access developer settings menu", 5, ["Return to Settings"], 2.0, ["developer_options"], ["Advanced settings"], 0.8)
                ]
            },
            'brightness_control': {
                'increase_brightness': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Quick Settings"], 0.9),
                    Subgoal("Navigate to Display", "Find display and brightness settings", 2, ["Open Settings"], 3.0, ["display_option"], ["Brightness"], 0.8),
                    Subgoal("Access Brightness Settings", "Open brightness configuration", 3, ["Navigate to Display"], 2.0, ["brightness_option"], ["Brightness slider"], 0.9),
                    Subgoal("Adjust Brightness", "Increase screen brightness level", 4, ["Access Brightness Settings"], 2.0, ["brightness_slider"], ["Quick Settings"], 0.8)
                ],
                'decrease_brightness': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Quick Settings"], 0.9),
                    Subgoal("Navigate to Display", "Find display and brightness settings", 2, ["Open Settings"], 3.0, ["display_option"], ["Brightness"], 0.8),
                    Subgoal("Access Brightness Settings", "Open brightness configuration", 3, ["Navigate to Display"], 2.0, ["brightness_option"], ["Brightness slider"], 0.9),
                    Subgoal("Adjust Brightness", "Decrease screen brightness level", 4, ["Access Brightness Settings"], 2.0, ["brightness_slider"], ["Quick Settings"], 0.8)
                ]
            },
            'volume_control': {
                'increase_volume': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Volume buttons"], 0.9),
                    Subgoal("Navigate to Sound", "Find sound and volume settings", 2, ["Open Settings"], 3.0, ["sound_option"], ["Volume"], 0.8),
                    Subgoal("Access Volume Settings", "Open volume configuration", 3, ["Navigate to Sound"], 2.0, ["volume_option"], ["Volume slider"], 0.9),
                    Subgoal("Adjust Volume", "Increase device volume level", 4, ["Access Volume Settings"], 2.0, ["volume_slider"], ["Volume buttons"], 0.8)
                ],
                'decrease_volume': [
                    Subgoal("Open Settings", "Navigate to the Settings application", 1, [], 2.0, ["settings_button"], ["Volume buttons"], 0.9),
                    Subgoal("Navigate to Sound", "Find sound and volume settings", 2, ["Open Settings"], 3.0, ["sound_option"], ["Volume"], 0.8),
                    Subgoal("Access Volume Settings", "Open volume configuration", 3, ["Navigate to Sound"], 2.0, ["volume_option"], ["Volume slider"], 0.9),
                    Subgoal("Adjust Volume", "Decrease device volume level", 4, ["Access Volume Settings"], 2.0, ["volume_slider"], ["Volume buttons"], 0.8)
                ]
            }
        }
        
        # Semantic planning patterns
        self.semantic_patterns = {
            'navigation': {
                'keywords': ['open', 'go to', 'navigate', 'access', 'enter'],
                'required_elements': ['button', 'menu', 'option'],
                'complexity': 'low'
            },
            'configuration': {
                'keywords': ['change', 'set', 'configure', 'adjust', 'modify'],
                'required_elements': ['slider', 'toggle', 'switch', 'input'],
                'complexity': 'medium'
            },
            'activation': {
                'keywords': ['turn on', 'enable', 'activate', 'start'],
                'required_elements': ['toggle', 'switch', 'button'],
                'complexity': 'low'
            },
            'deactivation': {
                'keywords': ['turn off', 'disable', 'deactivate', 'stop'],
                'required_elements': ['toggle', 'switch', 'button'],
                'complexity': 'low'
            },
            'complex_task': {
                'keywords': ['setup', 'install', 'configure', 'create'],
                'required_elements': ['multiple', 'wizard', 'form'],
                'complexity': 'high'
            }
        }

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> PlanningResult:
        """
        Generate a comprehensive plan for achieving a high-level goal.
        
        Args:
            goal: High-level goal description (e.g., "Turn off Wi-Fi")
            context: Optional context information (UI state, previous actions, etc.)
            
        Returns:
            PlanningResult: Detailed planning result with subgoals and metadata
        """
        start_time = time.time()
        self.stats['total_plans'] += 1
        
        # Safety check: truncate overly long goals to prevent recursion
        if len(goal) > 500:
            self.logger.warning("Planner", f"Goal too long ({len(goal)} chars), truncating")
            goal = goal[:500] + "..."
        
        # Safety check: prevent recursive subgoal descriptions
        if "Subgoal(" in goal:
            self.logger.warning("Planner", "Detected recursive subgoal in goal, cleaning")
            goal = "Complete the requested task"
        
        # Log planning start
        self.logger.info("Planner", "Planning started", 
                        goal=goal,
                        context_keys=list(context.keys()) if context else [])
        
        # Validate inputs
        if not goal or not goal.strip():
            return self._create_failure_result("Empty or invalid goal", start_time)
        
        # Apply multiple planning strategies
        results = []
        strategies_used = []
        
        planning_strategies = [
            self._template_based_planning,
            self._semantic_planning,
            self._adaptive_planning,
            self._fallback_planning
        ]
        
        for strategy in planning_strategies:
            try:
                result = strategy(goal, context)
                if result:
                    results.append(result)
                    strategies_used.append(strategy.__name__)
            except Exception as e:
                self.logger.warning("Planner", "Strategy failed", 
                                  strategy=strategy.__name__,
                                  error=str(e))
        
        # Combine results and make final decision
        final_result = self._combine_planning_results(
            results, strategies_used, goal, start_time
        )
        
        # Update statistics
        self._update_stats(final_result, time.time() - start_time)
        
        return final_result

    def replan(self, 
               failed_subgoal: str, 
               previous_subgoals: List[str], 
               goal: str,
               context: Optional[Dict[str, Any]] = None) -> PlanningResult:
        """
        Generate a new plan based on failed subgoal and previous execution history.
        
        Args:
            failed_subgoal: The subgoal that failed
            previous_subgoals: List of previously attempted subgoals
            goal: Original high-level goal
            context: Optional context information
            
        Returns:
            PlanningResult: New planning result with alternative approaches
        """
        start_time = time.time()
        
        # Log replanning start
        self.logger.info("Planner", "Replanning started", 
                        failed_subgoal=failed_subgoal,
                        previous_subgoals=previous_subgoals,
                        original_goal=goal)
        
        # Analyze failure and generate alternative plan
        failure_analysis = self._analyze_failure(failed_subgoal, previous_subgoals)
        
        # Generate alternative approaches
        alternative_plans = self._generate_alternative_plans(
            goal, failed_subgoal, failure_analysis, context
        )
        
        # Select best alternative plan
        best_plan = self._select_best_alternative(alternative_plans, failure_analysis)
        
        # Create replanning result
        result = PlanningResult(
            status=PlanningStatus.SUCCESS if best_plan else PlanningStatus.FAILED,
            subgoals=best_plan or [],
            total_estimated_duration=self._calculate_total_duration(best_plan),
            planning_time=time.time() - start_time,
            strategies_used=["replanning"],
            confidence=self._calculate_plan_confidence(best_plan),
            plan_complexity=self._assess_plan_complexity(best_plan),
            alternative_plans=alternative_plans
        )
        
        # Update statistics
        self._update_stats(result, time.time() - start_time)
        
        return result

    def _template_based_planning(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Optional[List[Subgoal]]:
        """Generate plan using predefined templates."""
        if not self.enable_template_matching:
            return None
        
        goal_lower = goal.lower()
        
        # Match against planning templates
        for category, templates in self.planning_templates.items():
            for template_name, subgoals in templates.items():
                if self._matches_template(goal_lower, template_name):
                    # Adapt template based on context
                    adapted_subgoals = self._adapt_template_to_context(subgoals, context)
                    return adapted_subgoals
        
        return None

    def _semantic_planning(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Optional[List[Subgoal]]:
        """Generate plan using semantic understanding."""
        if not self.enable_semantic_planning:
            return None
        
        # Extract semantic information from goal
        semantic_info = self._extract_semantic_info(goal)
        
        if not semantic_info:
            return None
        
        # Generate subgoals based on semantic patterns
        subgoals = []
        priority = 1
        
        # Add navigation subgoal if needed
        if semantic_info['requires_navigation']:
            subgoals.append(Subgoal(
                name="Navigate to Target",
                description=f"Navigate to the appropriate section for {semantic_info['action_type']}",
                priority=priority,
                dependencies=[],
                estimated_duration=3.0,
                required_elements=semantic_info['required_elements'],
                alternative_approaches=semantic_info['alternatives'],
                confidence=0.7
            ))
            priority += 1
        
        # Add action subgoal
        subgoals.append(Subgoal(
            name=semantic_info['action_name'],
            description=semantic_info['action_description'],
            priority=priority,
            dependencies=[sg.name for sg in subgoals],
            estimated_duration=semantic_info['estimated_duration'],
            required_elements=semantic_info['required_elements'],
            alternative_approaches=semantic_info['alternatives'],
            confidence=semantic_info['confidence']
        ))
        
        return subgoals

    def _adaptive_planning(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Optional[List[Subgoal]]:
        """Generate adaptive plan based on context and current state."""
        if not self.enable_adaptive_planning:
            return None
        
        # Analyze current context
        context_analysis = self._analyze_context(context)
        
        # Generate context-aware subgoals
        subgoals = []
        priority = 1
        
        # Add context-specific subgoals
        if context_analysis['current_app'] != 'settings':
            subgoals.append(Subgoal(
                name="Open Settings",
                description="Navigate to the Settings application",
                priority=priority,
                dependencies=[],
                estimated_duration=2.0,
                required_elements=["settings_button"],
                alternative_approaches=["Quick Settings", "App Drawer"],
                confidence=0.9
            ))
            priority += 1
        
        # Add goal-specific subgoals
        goal_subgoals = self._generate_goal_specific_subgoals(goal, context_analysis)
        for subgoal in goal_subgoals:
            subgoal.priority = priority
            subgoal.dependencies = [sg.name for sg in subgoals]
            subgoals.append(subgoal)
            priority += 1
        
        return subgoals

    def _fallback_planning(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Optional[List[Subgoal]]:
        """Generate fallback plan when other strategies fail."""
        # Simple fallback plan
        return [
            Subgoal(
                name="Open Settings",
                description="Navigate to the Settings application",
                priority=1,
                dependencies=[],
                estimated_duration=2.0,
                required_elements=["settings_button"],
                alternative_approaches=["Quick Settings"],
                confidence=0.8
            ),
            Subgoal(
                name="Search for Goal",
                description=f"Search for settings related to the task",
                priority=2,
                dependencies=["Open Settings"],
                estimated_duration=3.0,
                required_elements=["search_button", "search_input"],
                alternative_approaches=["Manual navigation"],
                confidence=0.6
            ),
            Subgoal(
                name="Execute Action",
                description="Perform the required action for the task",
                priority=3,
                dependencies=["Search for Goal"],
                estimated_duration=2.0,
                required_elements=["action_button"],
                alternative_approaches=["Direct interaction"],
                confidence=0.5
            )
        ]

    def _combine_planning_results(self, 
                                 results: List[List[Subgoal]], 
                                 strategies_used: List[str],
                                 goal: str,
                                 start_time: float) -> PlanningResult:
        """Combine multiple planning results into a final decision."""
        if not results:
            return self._create_failure_result("No planning strategies produced results", start_time)
        
        # Select the best plan based on confidence and completeness
        best_plan = max(results, key=lambda plan: self._calculate_plan_confidence(plan))
        
        # Optimize plan if enabled
        if self.enable_plan_optimization:
            best_plan = self._optimize_plan(best_plan)
        
        # Calculate plan metrics
        total_duration = self._calculate_total_duration(best_plan)
        confidence = self._calculate_plan_confidence(best_plan)
        complexity = self._assess_plan_complexity(best_plan)
        
        # Determine status
        if confidence >= self.min_confidence:
            status = PlanningStatus.SUCCESS
        elif confidence >= self.min_confidence * 0.7:
            status = PlanningStatus.PARTIAL
        else:
            status = PlanningStatus.FAILED
        
        return PlanningResult(
            status=status,
            subgoals=best_plan,
            total_estimated_duration=total_duration,
            planning_time=time.time() - start_time,
            strategies_used=strategies_used,
            confidence=confidence,
            plan_complexity=complexity,
            alternative_plans=results
        )

    def _matches_template(self, goal: str, template_name: str) -> bool:
        """Check if goal matches a template."""
        # Extract key terms from template name
        template_terms = template_name.replace('_', ' ').split()
        
        # Check if goal contains template terms
        for term in template_terms:
            if term in goal:
                return True
        
        return False

    def _adapt_template_to_context(self, subgoals: List[Subgoal], context: Optional[Dict[str, Any]]) -> List[Subgoal]:
        """Adapt template subgoals to current context."""
        if not context:
            return subgoals
        
        adapted_subgoals = []
        
        for subgoal in subgoals:
            # Check if subgoal is already completed in context
            if self._is_subgoal_completed(subgoal, context):
                continue
            
            # Adapt subgoal based on context
            adapted_subgoal = self._adapt_single_subgoal(subgoal, context)
            adapted_subgoals.append(adapted_subgoal)
        
        return adapted_subgoals

    def _extract_semantic_info(self, goal: str) -> Optional[Dict[str, Any]]:
        """Extract semantic information from goal."""
        goal_lower = goal.lower()
        
        # Determine action type
        action_type = None
        action_name = ""
        action_description = ""
        required_elements = []
        alternatives = []
        estimated_duration = 2.0
        confidence = 0.7
        requires_navigation = True
        
        for pattern_name, pattern in self.semantic_patterns.items():
            if any(keyword in goal_lower for keyword in pattern['keywords']):
                action_type = pattern_name
                required_elements = pattern['required_elements']
                
                # Extract specific action details
                if 'wifi' in goal_lower:
                    action_name = "Wi-Fi Management"
                    action_description = "Configure Wi-Fi settings"
                    alternatives = ["Quick Settings", "Network Settings"]
                elif 'bluetooth' in goal_lower:
                    action_name = "Bluetooth Management"
                    action_description = "Configure Bluetooth settings"
                    alternatives = ["Quick Settings", "Connected Devices"]
                elif 'brightness' in goal_lower:
                    action_name = "Brightness Control"
                    action_description = "Adjust screen brightness"
                    alternatives = ["Quick Settings", "Display Settings"]
                elif 'volume' in goal_lower:
                    action_name = "Volume Control"
                    action_description = "Adjust device volume"
                    alternatives = ["Volume Buttons", "Sound Settings"]
                else:
                    action_name = "General Action"
                    action_description = f"Perform {action_type} action"
                    alternatives = ["Settings", "Direct Interaction"]
                
                break
        
        if not action_type:
            return None
        
        return {
            'action_type': action_type,
            'action_name': action_name,
            'action_description': action_description,
            'required_elements': required_elements,
            'alternatives': alternatives,
            'estimated_duration': estimated_duration,
            'confidence': confidence,
            'requires_navigation': requires_navigation
        }

    def _analyze_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current context for adaptive planning."""
        if not context:
            return {
                'current_app': 'unknown',
                'available_elements': [],
                'ui_state': 'unknown',
                'previous_actions': []
            }
        
        return {
            'current_app': context.get('current_app', 'unknown'),
            'available_elements': context.get('ui_elements', []),
            'ui_state': context.get('ui_state', 'unknown'),
            'previous_actions': context.get('previous_actions', [])
        }

    def _generate_goal_specific_subgoals(self, goal: str, context_analysis: Dict[str, Any]) -> List[Subgoal]:
        """Generate goal-specific subgoals based on context analysis."""
        subgoals = []
        
        # Add goal-specific subgoals based on analysis
        if 'wifi' in goal.lower():
            subgoals.append(Subgoal(
                name="Access Wi-Fi Settings",
                description="Navigate to Wi-Fi configuration",
                priority=1,
                dependencies=[],
                estimated_duration=3.0,
                required_elements=["wifi_option"],
                alternative_approaches=["Network Settings"],
                confidence=0.8
            ))
        elif 'bluetooth' in goal.lower():
            subgoals.append(Subgoal(
                name="Access Bluetooth Settings",
                description="Navigate to Bluetooth configuration",
                priority=1,
                dependencies=[],
                estimated_duration=3.0,
                required_elements=["bluetooth_option"],
                alternative_approaches=["Connected Devices"],
                confidence=0.8
            ))
        
        return subgoals

    def _analyze_failure(self, failed_subgoal: str, previous_subgoals: List[str]) -> Dict[str, Any]:
        """Analyze failure to understand what went wrong."""
        return {
            'failed_subgoal': failed_subgoal,
            'previous_subgoals': previous_subgoals,
            'failure_type': self._classify_failure(failed_subgoal),
            'suggested_alternatives': self._suggest_alternatives(failed_subgoal)
        }

    def _classify_failure(self, failed_subgoal: str) -> str:
        """Classify the type of failure."""
        failed_lower = failed_subgoal.lower()
        
        if 'wifi' in failed_lower:
            return 'wifi_configuration_failure'
        elif 'bluetooth' in failed_lower:
            return 'bluetooth_configuration_failure'
        elif 'search' in failed_lower:
            return 'search_failure'
        elif 'navigate' in failed_lower:
            return 'navigation_failure'
        else:
            return 'general_failure'

    def _suggest_alternatives(self, failed_subgoal: str) -> List[str]:
        """Suggest alternative approaches for failed subgoal."""
        failed_lower = failed_subgoal.lower()
        
        if 'wifi' in failed_lower:
            return ["Quick Settings", "Network Settings", "Airplane Mode"]
        elif 'bluetooth' in failed_lower:
            return ["Quick Settings", "Connected Devices", "Device Settings"]
        elif 'search' in failed_lower:
            return ["Manual Navigation", "Direct Access", "Alternative Path"]
        else:
            return ["Alternative Approach", "Different Method", "Fallback Strategy"]

    def _generate_alternative_plans(self, 
                                  goal: str, 
                                  failed_subgoal: str, 
                                  failure_analysis: Dict[str, Any],
                                  context: Optional[Dict[str, Any]]) -> List[List[Subgoal]]:
        """Generate alternative plans based on failure analysis."""
        alternatives = []
        
        # Generate alternative based on failure type
        if failure_analysis['failure_type'] == 'wifi_configuration_failure':
            alternatives.append([
                Subgoal("Use Quick Settings", "Access Wi-Fi through quick settings panel", 1, [], 1.0, ["quick_settings"], [], 0.8),
                Subgoal("Toggle Wi-Fi", "Directly toggle Wi-Fi on/off", 2, ["Use Quick Settings"], 1.0, ["wifi_toggle"], [], 0.9)
            ])
        
        elif failure_analysis['failure_type'] == 'navigation_failure':
            alternatives.append([
                Subgoal("Alternative Navigation", "Use different navigation path", 1, [], 2.0, ["alternative_button"], [], 0.7),
                Subgoal("Direct Access", "Access target directly", 2, ["Alternative Navigation"], 2.0, ["direct_access"], [], 0.6)
            ])
        
        # Add generic fallback
        alternatives.append(self._fallback_planning(goal, context) or [])
        
        return alternatives

    def _select_best_alternative(self, 
                                alternative_plans: List[List[Subgoal]], 
                                failure_analysis: Dict[str, Any]) -> Optional[List[Subgoal]]:
        """Select the best alternative plan."""
        if not alternative_plans:
            return None
        
        # Score each alternative
        scored_plans = []
        for plan in alternative_plans:
            score = self._score_alternative_plan(plan, failure_analysis)
            scored_plans.append((score, plan))
        
        # Return the plan with highest score
        if scored_plans:
            return max(scored_plans, key=lambda x: x[0])[1]
        
        return None

    def _score_alternative_plan(self, plan: List[Subgoal], failure_analysis: Dict[str, Any]) -> float:
        """Score an alternative plan based on various factors."""
        if not plan:
            return 0.0
        
        # Base score from plan confidence
        base_score = sum(sg.confidence for sg in plan) / len(plan)
        
        # Penalty for complexity
        complexity_penalty = len(plan) * 0.1
        
        # Bonus for using suggested alternatives
        alternative_bonus = 0.0
        for subgoal in plan:
            if any(alt in subgoal.alternative_approaches for alt in failure_analysis['suggested_alternatives']):
                alternative_bonus += 0.2
        
        return max(0.0, base_score - complexity_penalty + alternative_bonus)

    def _is_subgoal_completed(self, subgoal: Subgoal, context: Optional[Dict[str, Any]]) -> bool:
        """Check if a subgoal is already completed in the context."""
        if not context or 'completed_subgoals' not in context:
            return False
        
        completed_subgoals = context['completed_subgoals']
        return subgoal.name in completed_subgoals

    def _adapt_single_subgoal(self, subgoal: Subgoal, context: Optional[Dict[str, Any]]) -> Subgoal:
        """Adapt a single subgoal based on context."""
        # For now, return the subgoal as-is
        # This can be enhanced with more sophisticated adaptation logic
        return subgoal

    def _optimize_plan(self, plan: List[Subgoal]) -> List[Subgoal]:
        """Optimize plan for efficiency and reliability."""
        if not plan:
            return plan
        
        # Remove redundant subgoals
        optimized_plan = []
        seen_names = set()
        
        for subgoal in plan:
            if subgoal.name not in seen_names:
                optimized_plan.append(subgoal)
                seen_names.add(subgoal.name)
        
        # Sort by priority
        optimized_plan.sort(key=lambda sg: sg.priority)
        
        return optimized_plan

    def _calculate_total_duration(self, plan: List[Subgoal]) -> float:
        """Calculate total estimated duration for a plan."""
        if not plan:
            return 0.0
        
        return sum(sg.estimated_duration for sg in plan)

    def _calculate_plan_confidence(self, plan: List[Subgoal]) -> float:
        """Calculate overall confidence for a plan."""
        if not plan:
            return 0.0
        
        return sum(sg.confidence for sg in plan) / len(plan)

    def _assess_plan_complexity(self, plan: List[Subgoal]) -> str:
        """Assess the complexity of a plan."""
        if not plan:
            return "none"
        
        if len(plan) <= 2:
            return "low"
        elif len(plan) <= 4:
            return "medium"
        else:
            return "high"

    def _create_failure_result(self, reason: str, start_time: float) -> PlanningResult:
        """Create a failure result."""
        return PlanningResult(
            status=PlanningStatus.FAILED,
            subgoals=[],
            total_estimated_duration=0.0,
            planning_time=time.time() - start_time,
            strategies_used=[],
            confidence=0.0,
            plan_complexity="none",
            alternative_plans=[]
        )

    def _update_stats(self, result: PlanningResult, planning_time: float):
        """Update planning statistics."""
        if result.status == PlanningStatus.SUCCESS:
            self.stats['successful_plans'] += 1
        elif result.status == PlanningStatus.FAILED:
            self.stats['failed_plans'] += 1
        
        # Update average planning time
        total_plans = self.stats['successful_plans'] + self.stats['failed_plans']
        if total_plans > 0:
            self.stats['average_planning_time'] = (
                (self.stats['average_planning_time'] * (total_plans - 1) + planning_time) 
                / total_plans
            )
        
        # Update strategy success rates
        for strategy in result.strategies_used:
            if strategy not in self.stats['strategy_success_rates']:
                self.stats['strategy_success_rates'][strategy] = {'success': 0, 'total': 0}
            
            self.stats['strategy_success_rates'][strategy]['total'] += 1
            if result.status == PlanningStatus.SUCCESS:
                self.stats['strategy_success_rates'][strategy]['success'] += 1
        
        # Update plan complexity distribution
        complexity = result.plan_complexity
        if complexity not in self.stats['plan_complexity_distribution']:
            self.stats['plan_complexity_distribution'][complexity] = 0
        self.stats['plan_complexity_distribution'][complexity] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset planning statistics."""
        self.stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'average_planning_time': 0.0,
            'strategy_success_rates': {},
            'plan_complexity_distribution': {}
        }

    def get_planning_history(self) -> List[Dict[str, Any]]:
        """Get planning history."""
        return self.planning_history.copy()