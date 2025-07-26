#!/usr/bin/env python3
"""
Full Integration Test for the robust planner, executor, and verifier agents.
This script demonstrates the complete workflow: Goal → Plan → Execute → Verify → Replan.
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.planner_agent import PlannerAgent, PlanningStatus, PlanningResult, Subgoal
from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.verifier_agent import VerifierAgent, VerificationStatus, VerificationResult
from utils.logger import QALogger
from config.qa_config import get_config

class MockEnvironment:
    """Mock environment for full integration testing."""
    
    def __init__(self):
        self.step_count = 0
        self.current_ui_state = self._create_initial_ui_state()
        self.ui_history = [self.current_ui_state.copy()]
        self.current_screen = "home"
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Mock step function that simulates UI changes."""
        self.step_count += 1
        
        # Simulate UI changes based on action
        element_id = action.get('element_id', '')
        action_type = action.get('action_type', 'touch')
        
        # Update UI state based on action
        self._simulate_ui_change(element_id, action_type)
        
        # Add to history
        self.ui_history.append(self.current_ui_state.copy())
        
        return {
            'observation': {
                'ui_tree': self.current_ui_state,
                'status': 'action_completed',
                'step_count': self.step_count
            },
            'reward': 1.0 if self._is_successful_action(element_id) else 0.5,
            'done': False
        }
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            'ui_tree': self.current_ui_state,
            'screen': self.current_screen
        }
    
    def _create_initial_ui_state(self) -> List[Dict[str, Any]]:
        """Create initial UI state."""
        return [
            {
                'id': 'wifi_switch',
                'text': 'Wi-Fi',
                'class': 'android.widget.Switch',
                'checked': True,
                'enabled': True,
                'clickable': True
            },
            {
                'id': 'wifi_label',
                'text': 'Wi-Fi is on',
                'class': 'android.widget.TextView',
                'enabled': True,
                'clickable': False
            },
            {
                'id': 'bluetooth_switch',
                'text': 'Bluetooth',
                'class': 'android.widget.Switch',
                'checked': False,
                'enabled': True,
                'clickable': True
            },
            {
                'id': 'bluetooth_label',
                'text': 'Bluetooth is off',
                'class': 'android.widget.TextView',
                'enabled': True,
                'clickable': False
            },
            {
                'id': 'settings_button',
                'text': 'Settings',
                'class': 'android.widget.Button',
                'enabled': True,
                'clickable': True
            },
            {
                'id': 'developer_options_button',
                'text': 'Developer Options',
                'class': 'android.widget.Button',
                'enabled': True,
                'clickable': True
            },
            {
                'id': 'home_button',
                'text': 'Home',
                'class': 'android.widget.Button',
                'enabled': True,
                'clickable': True
            }
        ]
    
    def _simulate_ui_change(self, element_id: str, action_type: str):
        """Simulate UI changes based on action."""
        if 'wifi' in element_id.lower():
            # Toggle WiFi
            for element in self.current_ui_state:
                if element['id'] == 'wifi_switch':
                    element['checked'] = not element['checked']
                elif element['id'] == 'wifi_label':
                    element['text'] = 'Wi-Fi is on' if element['checked'] else 'Wi-Fi is off'
        
        elif 'bluetooth' in element_id.lower():
            # Toggle Bluetooth
            for element in self.current_ui_state:
                if element['id'] == 'bluetooth_switch':
                    element['checked'] = not element['checked']
                elif element['id'] == 'bluetooth_label':
                    element['text'] = 'Bluetooth is on' if element['checked'] else 'Bluetooth is off'
        
        elif 'settings' in element_id.lower():
            # Navigate to settings
            self.current_screen = "settings"
            self.current_ui_state = [
                {
                    'id': 'settings_header',
                    'text': 'Settings',
                    'class': 'android.widget.TextView',
                    'enabled': True,
                    'clickable': False
                },
                {
                    'id': 'wifi_option',
                    'text': 'Wi-Fi',
                    'class': 'android.widget.ListItem',
                    'enabled': True,
                    'clickable': True
                },
                {
                    'id': 'bluetooth_option',
                    'text': 'Bluetooth',
                    'class': 'android.widget.ListItem',
                    'enabled': True,
                    'clickable': True
                },
                {
                    'id': 'developer_options_option',
                    'text': 'Developer Options',
                    'class': 'android.widget.ListItem',
                    'enabled': True,
                    'clickable': True
                },
                {
                    'id': 'back_button',
                    'text': 'Back',
                    'class': 'android.widget.Button',
                    'enabled': True,
                    'clickable': True
                }
            ]
        
        elif 'developer' in element_id.lower():
            # Navigate to developer options
            self.current_screen = "developer_options"
            self.current_ui_state = [
                {
                    'id': 'developer_header',
                    'text': 'Developer Options',
                    'class': 'android.widget.TextView',
                    'enabled': True,
                    'clickable': False
                },
                {
                    'id': 'usb_debugging_switch',
                    'text': 'USB Debugging',
                    'class': 'android.widget.Switch',
                    'checked': False,
                    'enabled': True,
                    'clickable': True
                },
                {
                    'id': 'usb_debugging_label',
                    'text': 'USB Debugging is off',
                    'class': 'android.widget.TextView',
                    'enabled': True,
                    'clickable': False
                },
                {
                    'id': 'back_button',
                    'text': 'Back',
                    'class': 'android.widget.Button',
                    'enabled': True,
                    'clickable': True
                }
            ]
        
        elif 'back' in element_id.lower() or 'home' in element_id.lower():
            # Go back to home
            self.current_screen = "home"
            self.current_ui_state = self._create_initial_ui_state()
    
    def _is_successful_action(self, element_id: str) -> bool:
        """Check if action was successful."""
        return any(keyword in element_id.lower() for keyword in ['wifi', 'bluetooth', 'settings', 'developer', 'back', 'home'])

class RobustAgentLoop:
    """
    A robust agent loop that coordinates planning, execution, and verification.
    """
    
    def __init__(self, env, config_name: str = "default"):
        """
        Initialize the robust agent loop.
        
        Args:
            env: The environment (AndroidEnv or mock)
            config_name: Configuration preset name
        """
        self.env = env
        self.config = get_config(config_name)
        self.logger = QALogger(log_level=self.config.logger.log_level, enable_console=True)
        
        # Initialize agents
        self.planner = PlannerAgent(
            logger=self.logger,
            enable_template_matching=self.config.planner.enable_template_matching,
            enable_semantic_planning=self.config.planner.enable_semantic_planning,
            enable_adaptive_planning=self.config.planner.enable_adaptive_planning,
            max_planning_time=self.config.planner.max_planning_time,
            min_confidence=self.config.planner.min_confidence,
            enable_plan_optimization=self.config.planner.enable_plan_optimization
        )
        
        self.executor = ExecutorAgent(
            env=env,
            logger=self.logger,
            max_retries=self.config.executor.max_retries,
            retry_delay=self.config.executor.retry_delay,
            action_timeout=self.config.executor.action_timeout,
            ui_settle_time=self.config.executor.ui_settle_time,
            enable_validation=self.config.executor.enable_validation,
            min_confidence=self.config.executor.min_confidence
        )
        
        self.verifier = VerifierAgent(
            logger=self.logger,
            min_change_ratio=self.config.verifier.min_change_ratio,
            fuzzy_threshold=self.config.verifier.fuzzy_threshold,
            max_verification_time=self.config.verifier.max_verification_time,
            enable_advanced_analysis=self.config.verifier.enable_advanced_analysis,
            enable_state_tracking=self.config.verifier.enable_state_tracking,
            min_confidence=self.config.verifier.min_confidence
        )
        
        # Track loop statistics
        self.stats = {
            'total_goals': 0,
            'successful_goals': 0,
            'failed_goals': 0,
            'total_plans': 0,
            'total_replans': 0,
            'total_executions': 0,
            'total_verifications': 0,
            'average_goal_completion_time': 0.0
        }
    
    def execute_goal(self, goal: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Execute a high-level goal using the robust agent loop.
        
        Args:
            goal: High-level goal (e.g., "Turn off Wi-Fi and enable Bluetooth")
            max_iterations: Maximum iterations for the loop
            
        Returns:
            Dict containing execution results and statistics
        """
        start_time = time.time()
        self.stats['total_goals'] += 1
        
        self.logger.info("AgentLoop", "Starting goal execution", goal=goal)
        
        # Get initial observation
        initial_obs = self.env.get_observation()
        
        # Generate initial plan
        planning_result = self.planner.plan(goal, initial_obs)
        self.stats['total_plans'] += 1
        
        if planning_result.status != PlanningStatus.SUCCESS:
            return self._create_failure_result(
                f"Planning failed: {planning_result.status.value}",
                start_time,
                planning_result
            )
        
        subgoals = planning_result.subgoals
        completed_subgoals = []
        failed_subgoals = []
        
        self.logger.info("AgentLoop", "Plan generated", 
                        subgoals_count=len(subgoals),
                        total_estimated_duration=planning_result.total_estimated_duration,
                        confidence=planning_result.confidence)
        
        iteration = 0
        while iteration < max_iterations and subgoals:
            iteration += 1
            self.logger.info("AgentLoop", f"Iteration {iteration}", 
                           remaining_subgoals=len(subgoals),
                           completed_subgoals=len(completed_subgoals))
            
            # Get current subgoal
            current_subgoal = subgoals[0]
            subgoals = subgoals[1:]  # Remove from queue
            
            # Get current observation
            current_obs = self.env.get_observation()
            
            # Execute subgoal
            self.stats['total_executions'] += 1
            execution_result = self.executor.execute(current_subgoal.name, current_obs['ui_tree'])
            
            # Get post-execution observation
            post_obs = self.env.get_observation()
            
            # Verify subgoal completion
            self.stats['total_verifications'] += 1
            verification_result = self.verifier.verify(current_subgoal.name, current_obs, post_obs)
            
            # Log results
            self.logger.info("AgentLoop", "Subgoal completed", 
                           subgoal=current_subgoal.name,
                           execution_status=execution_result.status,
                           verification_status=verification_result.status.value,
                           verification_confidence=verification_result.confidence)
            
            # Check if subgoal was successful
            if (execution_result.status == "success" and 
                verification_result.status == VerificationStatus.PASS):
                completed_subgoals.append(current_subgoal)
                self.logger.info("AgentLoop", "Subgoal successful", 
                               subgoal=current_subgoal.name)
            else:
                failed_subgoals.append(current_subgoal)
                self.logger.warning("AgentLoop", "Subgoal failed", 
                                  subgoal=current_subgoal.name,
                                  execution_reason=execution_result.reason,
                                  verification_reason=verification_result.reason)
                
                # Check if replanning is needed
                if verification_result.needs_replan:
                    self.logger.info("AgentLoop", "Replanning needed", 
                                   failed_subgoal=current_subgoal.name)
                    
                    # Generate failure context
                    failure_context = {
                        'failed_subgoal': current_subgoal.name,
                        'execution_reason': execution_result.reason,
                        'verification_reason': verification_result.reason,
                        'completed_subgoals': [sg.name for sg in completed_subgoals],
                        'remaining_subgoals': [sg.name for sg in subgoals]
                    }
                    
                    # Replan
                    replan_result = self.planner.replan(
                        goal, 
                        completed_subgoals, 
                        failed_subgoals, 
                        failure_context
                    )
                    self.stats['total_replans'] += 1
                    
                    if replan_result.status == PlanningStatus.SUCCESS:
                        # Replace remaining subgoals with new plan
                        subgoals = replan_result.subgoals
                        self.logger.info("AgentLoop", "Replan successful", 
                                       new_subgoals_count=len(subgoals))
                    else:
                        self.logger.error("AgentLoop", "Replan failed", 
                                        replan_status=replan_result.status.value)
                        break
        
        # Determine overall success
        total_subgoals = len(completed_subgoals) + len(failed_subgoals)
        success_rate = len(completed_subgoals) / max(total_subgoals, 1)
        
        if success_rate >= 0.8:  # 80% success threshold
            self.stats['successful_goals'] += 1
            final_status = "success"
        else:
            self.stats['failed_goals'] += 1
            final_status = "failed"
        
        execution_time = time.time() - start_time
        self.stats['average_goal_completion_time'] = (
            (self.stats['average_goal_completion_time'] * (self.stats['total_goals'] - 1) + execution_time) / 
            self.stats['total_goals']
        )
        
        return {
            'status': final_status,
            'goal': goal,
            'completed_subgoals': [sg.name for sg in completed_subgoals],
            'failed_subgoals': [sg.name for sg in failed_subgoals],
            'success_rate': success_rate,
            'execution_time': execution_time,
            'iterations': iteration,
            'planning_result': planning_result,
            'stats': self.stats.copy()
        }
    
    def _create_failure_result(self, reason: str, start_time: float, planning_result: Optional[PlanningResult] = None) -> Dict[str, Any]:
        """Create a failure result."""
        self.stats['failed_goals'] += 1
        return {
            'status': 'failed',
            'reason': reason,
            'execution_time': time.time() - start_time,
            'planning_result': planning_result,
            'stats': self.stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'loop_stats': self.stats,
            'planner_stats': self.planner.get_stats(),
            'executor_stats': self.executor.get_stats(),
            'verifier_stats': self.verifier.get_stats()
        }

def test_full_integration_workflow():
    """Test the complete robust agent loop."""
    print("🔄 Testing Full Robust Agent Loop")
    print("=" * 60)
    
    # Create mock environment
    mock_env = MockEnvironment()
    
    # Create robust agent loop
    agent_loop = RobustAgentLoop(mock_env, "default")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Simple WiFi Management',
            'goal': 'Turn off Wi-Fi',
            'expected_status': 'success'
        },
        {
            'name': 'Complex Settings Management',
            'goal': 'Turn off Wi-Fi and enable Bluetooth',
            'expected_status': 'success'
        },
        {
            'name': 'Navigation and Configuration',
            'goal': 'Open Settings and enable USB Debugging',
            'expected_status': 'success'
        },
        {
            'name': 'Multi-step Configuration',
            'goal': 'Turn off Wi-Fi, enable Bluetooth, and open Developer Options',
            'expected_status': 'success'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print("-" * 50)
        print(f"   Goal: {scenario['goal']}")
        
        # Execute goal
        result = agent_loop.execute_goal(scenario['goal'])
        
        # Print results
        print(f"   Status: {result['status']}")
        print(f"   Success Rate: {result['success_rate']:.1%}")
        print(f"   Execution Time: {result['execution_time']:.2f}s")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Completed Subgoals: {len(result['completed_subgoals'])}")
        print(f"   Failed Subgoals: {len(result['failed_subgoals'])}")
        
        if result['completed_subgoals']:
            print(f"   Completed: {', '.join(result['completed_subgoals'])}")
        
        if result['failed_subgoals']:
            print(f"   Failed: {', '.join(result['failed_subgoals'])}")
        
        # Check if planning was successful
        if result['planning_result']:
            plan_result = result['planning_result']
            print(f"   Planning Status: {plan_result.status.value}")
            print(f"   Planning Confidence: {plan_result.confidence:.2f}")
            print(f"   Planning Time: {plan_result.planning_time:.2f}s")
            print(f"   Strategies Used: {', '.join(plan_result.strategies_used)}")
        
        # Overall success
        expected_success = result['status'] == scenario['expected_status']
        print(f"   Expected: {'✅ MATCH' if expected_success else '❌ MISMATCH'}")
    
    # Print comprehensive statistics
    print(f"\n📊 Comprehensive Statistics")
    print("=" * 60)
    
    stats = agent_loop.get_stats()
    
    print(f"   Loop Statistics:")
    print(f"     Total Goals: {stats['loop_stats']['total_goals']}")
    print(f"     Success Rate: {stats['loop_stats']['successful_goals'] / max(stats['loop_stats']['total_goals'], 1) * 100:.1f}%")
    print(f"     Average Goal Time: {stats['loop_stats']['average_goal_completion_time']:.2f}s")
    print(f"     Total Plans: {stats['loop_stats']['total_plans']}")
    print(f"     Total Replans: {stats['loop_stats']['total_replans']}")
    print(f"     Total Executions: {stats['loop_stats']['total_executions']}")
    print(f"     Total Verifications: {stats['loop_stats']['total_verifications']}")
    
    print(f"\n   Planner Statistics:")
    planner_stats = stats['planner_stats']
    print(f"     Total Plans: {planner_stats['total_plans']}")
    print(f"     Success Rate: {planner_stats['successful_plans'] / max(planner_stats['total_plans'], 1) * 100:.1f}%")
    print(f"     Average Planning Time: {planner_stats['average_planning_time']:.2f}s")
    
    print(f"\n   Executor Statistics:")
    executor_stats = stats['executor_stats']
    print(f"     Total Executions: {executor_stats['total_executions']}")
    print(f"     Success Rate: {executor_stats['successful_executions'] / max(executor_stats['total_executions'], 1) * 100:.1f}%")
    print(f"     Average Execution Time: {executor_stats['average_execution_time']:.2f}s")
    
    print(f"\n   Verifier Statistics:")
    verifier_stats = stats['verifier_stats']
    print(f"     Total Verifications: {verifier_stats['total_verifications']}")
    print(f"     Success Rate: {verifier_stats['successful_verifications'] / max(verifier_stats['total_verifications'], 1) * 100:.1f}%")
    print(f"     Average Verification Time: {verifier_stats['average_verification_time']:.2f}s")
    
    print(f"\n✅ Full integration testing completed!")

def test_error_handling():
    """Test error handling in the robust agent loop."""
    print(f"\n🚨 Testing Error Handling")
    print("=" * 60)
    
    # Create mock environment
    mock_env = MockEnvironment()
    
    # Create robust agent loop
    agent_loop = RobustAgentLoop(mock_env)
    
    # Test with invalid goal
    print("\n📋 Test: Invalid Goal")
    print("-" * 30)
    
    result = agent_loop.execute_goal("")
    print(f"   Status: {result['status']}")
    print(f"   Reason: {result.get('reason', 'N/A')}")
    
    # Test with impossible goal
    print("\n📋 Test: Impossible Goal")
    print("-" * 30)
    
    result = agent_loop.execute_goal("Make coffee with the phone")
    print(f"   Status: {result['status']}")
    print(f"   Success Rate: {result['success_rate']:.1%}")
    
    print(f"\n✅ Error handling testing completed!")

if __name__ == "__main__":
    try:
        test_full_integration_workflow()
        test_error_handling()
    except Exception as e:
        print(f"❌ Full integration test failed with error: {e}")
        import traceback
        traceback.print_exc() 