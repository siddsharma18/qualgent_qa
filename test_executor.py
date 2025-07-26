#!/usr/bin/env python3
"""
Test script for the robust executor agent.
This script demonstrates the improved functionality of the executor agent.
"""

import sys
import os
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.executor_agent import ExecutorAgent, ExecutionResult
from utils.logger import QALogger
from utils.ui_parser import UIParser
from config.qa_config import get_config

class MockEnvironment:
    """Mock environment for testing the executor agent."""
    
    def __init__(self):
        self.step_count = 0
        self.mock_ui_tree = self._create_mock_ui_tree()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Mock step function."""
        self.step_count += 1
        
        # Simulate different scenarios based on action
        element_id = action.get('element_id', '')
        
        if 'wifi' in element_id.lower():
            # Simulate WiFi toggle
            return {
                'observation': {
                    'ui_tree': self.mock_ui_tree,
                    'status': 'wifi_toggled'
                },
                'reward': 1.0,
                'done': False
            }
        elif 'invalid' in element_id.lower():
            # Simulate failure
            raise ValueError("Invalid element")
        else:
            # Simulate successful action
            return {
                'observation': {
                    'ui_tree': self.mock_ui_tree,
                    'status': 'action_completed'
                },
                'reward': 0.5,
                'done': False
            }
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            'ui_tree': self.mock_ui_tree
        }
    
    def _create_mock_ui_tree(self) -> List[Dict[str, Any]]:
        """Create a mock UI tree for testing."""
        return [
            {
                'id': 'wifi_toggle',
                'text': 'Wi-Fi',
                'class': 'android.widget.Switch',
                'clickable': True,
                'content-desc': 'Wi-Fi toggle switch'
            },
            {
                'id': 'bluetooth_toggle',
                'text': 'Bluetooth',
                'class': 'android.widget.Switch',
                'clickable': True,
                'content-desc': 'Bluetooth toggle switch'
            },
            {
                'id': 'settings_button',
                'text': 'Settings',
                'class': 'android.widget.Button',
                'clickable': True,
                'content-desc': 'Settings button'
            },
            {
                'id': 'back_button',
                'text': 'Back',
                'class': 'android.widget.Button',
                'clickable': True,
                'content-desc': 'Back button'
            },
            {
                'id': 'invalid_element',
                'text': 'Invalid',
                'class': 'android.widget.TextView',
                'clickable': False,
                'content-desc': 'Invalid element'
            }
        ]

def test_executor_agent():
    """Test the executor agent with various scenarios."""
    print("🧪 Testing Robust Executor Agent")
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
    
    # Test cases
    test_cases = [
        {
            'name': 'Wi-Fi Toggle',
            'subgoal': 'Turn Wi-Fi off',
            'expected_status': 'success'
        },
        {
            'name': 'Bluetooth Toggle',
            'subgoal': 'Enable Bluetooth',
            'expected_status': 'success'
        },
        {
            'name': 'Settings Navigation',
            'subgoal': 'Open Settings',
            'expected_status': 'success'
        },
        {
            'name': 'Back Navigation',
            'subgoal': 'Go back',
            'expected_status': 'success'
        },
        {
            'name': 'Invalid Element',
            'subgoal': 'Click invalid element',
            'expected_status': 'fail'
        },
        {
            'name': 'Non-existent Element',
            'subgoal': 'Click something that does not exist',
            'expected_status': 'fail'
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"   Subgoal: {test_case['subgoal']}")
        
        try:
            result = executor.execute(test_case['subgoal'], mock_env.mock_ui_tree)
            results.append(result)
            
            print(f"   Status: {result.status}")
            print(f"   Attempts: {result.attempts}")
            print(f"   Execution Time: {result.execution_time:.2f}s")
            
            if result.reason:
                print(f"   Reason: {result.reason}")
            
            if result.status == test_case['expected_status']:
                print("   ✅ PASS")
            else:
                print(f"   ❌ FAIL (Expected: {test_case['expected_status']})")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(ExecutionResult(status="error", reason=str(e)))
    
    # Print statistics
    print(f"\n📊 Execution Statistics:")
    stats = executor.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test UI parser directly
    print(f"\n🔍 Testing UI Parser")
    print("-" * 30)
    
    ui_parser = UIParser()
    
    parser_test_cases = [
        "Turn Wi-Fi off",
        "Enable Bluetooth", 
        "Open Settings",
        "Go back",
        "Click something random"
    ]
    
    for subgoal in parser_test_cases:
        element_id = ui_parser.find_element_for_subgoal(mock_env.mock_ui_tree, subgoal)
        if element_id:
            print(f"   '{subgoal}' -> Found: {element_id}")
        else:
            print(f"   '{subgoal}' -> Not found")
    
    print(f"\n✅ Testing completed!")

def test_configurations():
    """Test different configuration presets."""
    print(f"\n⚙️  Testing Configuration Presets")
    print("=" * 50)
    
    configs = ["default", "high_performance", "high_reliability"]
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n📋 {config_name.upper()} Configuration:")
        config_dict = config.to_dict()
        
        for section, settings in config_dict.items():
            if isinstance(settings, dict):
                print(f"   {section}:")
                for key, value in settings.items():
                    print(f"     {key}: {value}")
            else:
                print(f"   {section}: {settings}")

if __name__ == "__main__":
    try:
        test_executor_agent()
        test_configurations()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 