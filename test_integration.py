#!/usr/bin/env python3
"""
Integration test for the robust executor and verifier agents.
This script demonstrates how the agents work together in a complete workflow.
"""

import sys
import os
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.verifier_agent import VerifierAgent, VerificationStatus, VerificationResult
from utils.logger import QALogger
from config.qa_config import get_config

class MockEnvironment:
    """Mock environment for integration testing."""
    
    def __init__(self):
        self.step_count = 0
        self.current_ui_state = self._create_initial_ui_state()
        self.ui_history = [self.current_ui_state.copy()]
    
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
            'ui_tree': self.current_ui_state
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
                    'id': 'back_button',
                    'text': 'Back',
                    'class': 'android.widget.Button',
                    'enabled': True,
                    'clickable': True
                }
            ]
        
        elif 'back' in element_id.lower() or 'home' in element_id.lower():
            # Go back to home
            self.current_ui_state = self._create_initial_ui_state()
    
    def _is_successful_action(self, element_id: str) -> bool:
        """Check if action was successful."""
        return 'wifi' in element_id.lower() or 'bluetooth' in element_id.lower() or 'settings' in element_id.lower()

def test_integration_workflow():
    """Test the complete integration workflow."""
    print("🔄 Testing Integration Workflow")
    print("=" * 50)
    
    # Create mock environment
    mock_env = MockEnvironment()
    
    # Create logger
    logger = QALogger(log_level="INFO", enable_console=True)
    
    # Get configuration
    config = get_config("default")
    
    # Create executor agent
    executor = ExecutorAgent(
        env=mock_env,
        logger=logger,
        max_retries=config.executor.max_retries,
        retry_delay=config.executor.retry_delay,
        action_timeout=config.executor.action_timeout,
        ui_settle_time=config.executor.ui_settle_time,
        enable_validation=config.executor.enable_validation,
        min_confidence=config.executor.min_confidence
    )
    
    # Create verifier agent
    verifier = VerifierAgent(
        logger=logger,
        min_change_ratio=config.verifier.min_change_ratio,
        fuzzy_threshold=config.verifier.fuzzy_threshold,
        max_verification_time=config.verifier.max_verification_time,
        enable_advanced_analysis=config.verifier.enable_advanced_analysis,
        enable_state_tracking=config.verifier.enable_state_tracking,
        min_confidence=config.verifier.min_confidence
    )
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'WiFi Toggle Workflow',
            'subgoal': 'Turn Wi-Fi off',
            'expected_execution': 'success',
            'expected_verification': 'pass'
        },
        {
            'name': 'Bluetooth Toggle Workflow',
            'subgoal': 'Enable Bluetooth',
            'expected_execution': 'success',
            'expected_verification': 'pass'
        },
        {
            'name': 'Settings Navigation Workflow',
            'subgoal': 'Open Settings',
            'expected_execution': 'success',
            'expected_verification': 'pass'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        # Get initial observation
        initial_obs = mock_env.get_observation()
        print(f"   Initial UI elements: {len(initial_obs['ui_tree'])}")
        
        # Execute subgoal
        print(f"   Executing: {scenario['subgoal']}")
        execution_result = executor.execute(scenario['subgoal'], initial_obs['ui_tree'])
        
        print(f"   Execution Status: {execution_result.status}")
        print(f"   Execution Attempts: {execution_result.attempts}")
        print(f"   Execution Time: {execution_result.execution_time:.2f}s")
        
        if execution_result.reason:
            print(f"   Execution Reason: {execution_result.reason}")
        
        # Get final observation
        final_obs = mock_env.get_observation()
        print(f"   Final UI elements: {len(final_obs['ui_tree'])}")
        
        # Verify subgoal completion
        print(f"   Verifying: {scenario['subgoal']}")
        verification_result = verifier.verify(scenario['subgoal'], initial_obs, final_obs)
        
        print(f"   Verification Status: {verification_result.status.value}")
        print(f"   Verification Confidence: {verification_result.confidence:.2f}")
        print(f"   Verification Reason: {verification_result.reason}")
        print(f"   Needs Replan: {verification_result.needs_replan}")
        print(f"   Strategies Used: {len(verification_result.strategies_used)}")
        
        # Check results
        execution_success = execution_result.status == "success"
        verification_success = verification_result.status == VerificationStatus.PASS
        
        print(f"   Execution: {'✅ PASS' if execution_success else '❌ FAIL'}")
        print(f"   Verification: {'✅ PASS' if verification_success else '❌ FAIL'}")
        
        # Overall workflow success
        workflow_success = execution_success and verification_success
        print(f"   Overall Workflow: {'✅ SUCCESS' if workflow_success else '❌ FAILED'}")
    
    # Print final statistics
    print(f"\n📊 Final Statistics:")
    print("-" * 30)
    
    executor_stats = executor.get_stats()
    verifier_stats = verifier.get_stats()
    
    print(f"   Executor:")
    print(f"     Total Executions: {executor_stats['total_executions']}")
    print(f"     Success Rate: {executor_stats['successful_executions'] / max(executor_stats['total_executions'], 1) * 100:.1f}%")
    print(f"     Average Execution Time: {executor_stats['average_execution_time']:.2f}s")
    
    print(f"   Verifier:")
    print(f"     Total Verifications: {verifier_stats['total_verifications']}")
    print(f"     Success Rate: {verifier_stats['successful_verifications'] / max(verifier_stats['total_verifications'], 1) * 100:.1f}%")
    print(f"     Average Verification Time: {verifier_stats['average_verification_time']:.2f}s")
    
    # Show state history
    print(f"\n📈 State History Analysis:")
    print("-" * 30)
    state_history = verifier.get_state_history()
    print(f"   Total State Changes: {len(state_history)}")
    
    for i, state in enumerate(state_history[-3:], 1):  # Show last 3 states
        print(f"   State {i}:")
        print(f"     Subgoal: {state['subgoal']}")
        print(f"     Elements: {state['prev_element_count']} → {state['curr_element_count']}")
        print(f"     Changes: {len(state['changes'])}")
    
    print(f"\n✅ Integration testing completed!")

def test_error_handling():
    """Test error handling in the integration workflow."""
    print(f"\n🚨 Testing Error Handling")
    print("=" * 50)
    
    # Create mock environment
    mock_env = MockEnvironment()
    
    # Create agents
    executor = ExecutorAgent(env=mock_env)
    verifier = VerifierAgent()
    
    # Test with invalid subgoal
    print("\n📋 Test: Invalid Subgoal")
    print("-" * 30)
    
    initial_obs = mock_env.get_observation()
    execution_result = executor.execute("", initial_obs['ui_tree'])
    
    print(f"   Execution Status: {execution_result.status}")
    print(f"   Execution Reason: {execution_result.reason}")
    
    # Test with invalid observations
    print("\n📋 Test: Invalid Observations")
    print("-" * 30)
    
    verification_result = verifier.verify("Test subgoal", {}, {})
    
    print(f"   Verification Status: {verification_result.status.value}")
    print(f"   Verification Reason: {verification_result.reason}")
    
    print(f"\n✅ Error handling testing completed!")

if __name__ == "__main__":
    try:
        test_integration_workflow()
        test_error_handling()
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc() 