#!/usr/bin/env python3
"""
Test script for Agent-S integration with robust QA agents.
This demonstrates the complete Agent-S architecture with our QA system.
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_s_integration import AgentSGraphSearchAgent, AgentSMessageType
from utils.logger import QALogger
from config.qa_config import get_config

class MockAndroidEnv:
    """Mock Android environment for testing Agent-S integration."""
    
    def __init__(self):
        self.current_screen = "home"
        self.wifi_enabled = True
        self.bluetooth_enabled = False
        self.settings_open = False
        
    def reset(self):
        """Reset the environment."""
        self.current_screen = "home"
        self.wifi_enabled = True
        self.bluetooth_enabled = False
        self.settings_open = False
        return self._get_observation()
    
    def step(self, action):
        """Execute an action and return observation."""
        action_type = action.get("action_type", "")
        element_id = action.get("element_id", "")
        
        if action_type == "touch":
            if "settings" in element_id.lower():
                self.settings_open = True
                self.current_screen = "settings"
            elif "wifi" in element_id.lower() and self.settings_open:
                self.wifi_enabled = not self.wifi_enabled
            elif "bluetooth" in element_id.lower() and self.settings_open:
                self.bluetooth_enabled = not self.bluetooth_enabled
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation."""
        ui_tree = []
        
        if self.current_screen == "home":
            ui_tree = [
                {"text": "Settings", "type": "button", "id": "settings_button"},
                {"text": "Wi-Fi", "type": "toggle", "id": "wifi_toggle", "state": "on" if self.wifi_enabled else "off"},
                {"text": "Bluetooth", "type": "toggle", "id": "bluetooth_toggle", "state": "on" if self.bluetooth_enabled else "off"}
            ]
        elif self.current_screen == "settings":
            ui_tree = [
                {"text": "Wi-Fi", "type": "toggle", "id": "wifi_toggle", "state": "on" if self.wifi_enabled else "off"},
                {"text": "Bluetooth", "type": "toggle", "id": "bluetooth_toggle", "state": "on" if self.bluetooth_enabled else "off"},
                {"text": "Back", "type": "button", "id": "back_button"}
            ]
        
        return {
            "ui_tree": ui_tree,
            "screenshot": b"mock_screenshot_data",
            "accessibility_tree": f"Screen: {self.current_screen}, WiFi: {self.wifi_enabled}, Bluetooth: {self.bluetooth_enabled}"
        }

def test_agent_s_integration():
    """Test the Agent-S integration with robust QA agents."""
    print("🤖 Testing Agent-S Integration with Robust QA Agents")
    print("=" * 70)
    
    # Create mock environment
    mock_env = MockAndroidEnv()
    
    # Initialize Agent-S GraphSearchAgent
    engine_params = {
        "engine_type": "mock",
        "model": "mock-llm",
        "api_key": "mock_key"
    }
    
    agent = AgentSGraphSearchAgent(
        env=mock_env,
        engine_params=engine_params,
        platform="android",
        action_space="android_env",
        observation_type="mixed"
    )
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Turn off Wi-Fi",
            "instruction": "Turn off Wi-Fi",
            "expected_result": "WiFi should be disabled"
        },
        {
            "name": "Enable Bluetooth",
            "instruction": "Enable Bluetooth",
            "expected_result": "Bluetooth should be enabled"
        },
        {
            "name": "Complex Task",
            "instruction": "Turn off Wi-Fi and enable Bluetooth",
            "expected_result": "WiFi disabled and Bluetooth enabled"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing Scenario: {scenario['name']}")
        print("-" * 50)
        
        # Reset agent and environment
        agent.reset()
        obs = mock_env.reset()
        
        print(f"Goal: {scenario['instruction']}")
        print(f"Expected: {scenario['expected_result']}")
        
        # Execute the task using Agent-S pattern
        max_iterations = 10
        trajectory = f"Task: {scenario['instruction']}\n"
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}")
            
            # Get next action from agent
            info, actions = agent.predict(scenario['instruction'], obs)
            
            print(f"   Current Subtask: {info.get('subtask', 'None')}")
            print(f"   Subtask Status: {info.get('subtask_status', 'Unknown')}")
            print(f"   Actions: {actions}")
            
            # Execute actions
            if actions and "DONE" not in actions and "FAIL" not in actions:
                for action in actions:
                    if action.startswith("touch"):
                        # Extract element ID from action
                        element_id = action.split("(")[1].split(")")[0].strip('"')
                        obs = mock_env.step({
                            "action_type": "touch",
                            "element_id": element_id
                        })
                        print(f"   Executed: {action}")
            
            # Check if task is complete
            if "DONE" in actions:
                print("   ✅ Task completed successfully!")
                break
            elif "FAIL" in actions:
                print("   ❌ Task failed!")
                break
            
            # Update trajectory
            trajectory += f"\nIteration {iteration + 1}: {info.get('subtask', 'Unknown')} - {info.get('subtask_status', 'Unknown')}"
        
        # Update narrative memory
        agent.update_narrative_memory(trajectory)
        
        # Display results
        print(f"\n📊 Final State:")
        print(f"   WiFi: {'Enabled' if mock_env.wifi_enabled else 'Disabled'}")
        print(f"   Bluetooth: {'Enabled' if mock_env.bluetooth_enabled else 'Disabled'}")
        print(f"   Screen: {mock_env.current_screen}")
        
        # Display Agent-S statistics
        print(f"\n📈 Agent-S Statistics:")
        print(f"   Total Messages: {len(agent.get_message_history())}")
        print(f"   Visual Traces: {len(agent.get_visual_traces())}")
        print(f"   Episode Trajectories: {len(agent.get_episode_trajectory())}")
        
        # Display message types
        message_types = {}
        for msg in agent.get_message_history():
            msg_type = msg.message_type.value
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        print(f"   Message Types: {message_types}")
    
    print(f"\n✅ Agent-S integration testing completed!")

def test_visual_trace_analysis():
    """Test visual trace analysis capabilities."""
    print(f"\n📸 Testing Visual Trace Analysis")
    print("=" * 50)
    
    # Create mock environment
    mock_env = MockAndroidEnv()
    
    # Initialize agent
    engine_params = {"engine_type": "mock", "model": "mock-llm"}
    agent = AgentSGraphSearchAgent(env=mock_env, engine_params=engine_params)
    
    # Execute a simple task
    agent.reset()
    obs = mock_env.reset()
    
    # Simulate some actions
    for i in range(3):
        info, actions = agent.predict("Test visual tracing", obs)
        if actions and "DONE" not in actions:
            obs = mock_env.step({"action_type": "touch", "element_id": f"element_{i}"})
    
    # Get visual traces
    visual_traces = agent.get_visual_traces()
    
    print(f"📊 Visual Trace Analysis:")
    print(f"   Total Traces: {len(visual_traces)}")
    
    for i, trace in enumerate(visual_traces[:3]):  # Show first 3 traces
        print(f"   Trace {i+1}:")
        print(f"     Event: {trace['event']}")
        print(f"     Timestamp: {trace['timestamp']}")
        print(f"     Step Count: {trace['step_count']}")
        print(f"     Current Subtask: {trace['current_subtask']}")
        print(f"     Screenshot Size: {len(trace['screenshot']) if trace['screenshot'] else 0} bytes")
    
    print(f"\n✅ Visual trace analysis testing completed!")

def test_message_history():
    """Test Agent-S message history and communication."""
    print(f"\n💬 Testing Agent-S Message History")
    print("=" * 50)
    
    # Create mock environment
    mock_env = MockAndroidEnv()
    
    # Initialize agent
    engine_params = {"engine_type": "mock", "model": "mock-llm"}
    agent = AgentSGraphSearchAgent(env=mock_env, engine_params=engine_params)
    
    # Execute a task
    agent.reset()
    obs = mock_env.reset()
    
    info, actions = agent.predict("Test message history", obs)
    
    # Get message history
    messages = agent.get_message_history()
    
    print(f"📊 Message History Analysis:")
    print(f"   Total Messages: {len(messages)}")
    
    # Group messages by type
    message_groups = {}
    for msg in messages:
        msg_type = msg.message_type.value
        if msg_type not in message_groups:
            message_groups[msg_type] = []
        message_groups[msg_type].append(msg)
    
    for msg_type, msgs in message_groups.items():
        print(f"   {msg_type.upper()} Messages: {len(msgs)}")
        for msg in msgs[:2]:  # Show first 2 messages of each type
            print(f"     {msg.sender} -> {msg.receiver}: {msg.content.get('status', 'N/A')}")
    
    print(f"\n✅ Message history testing completed!")

def test_evaluation_report():
    """Test evaluation report generation with Agent-S integration."""
    print(f"\n📋 Testing Evaluation Report Generation")
    print("=" * 50)
    
    # Create mock environment
    mock_env = MockAndroidEnv()
    
    # Initialize agent
    engine_params = {"engine_type": "mock", "model": "mock-llm"}
    agent = AgentSGraphSearchAgent(env=mock_env, engine_params=engine_params)
    
    # Execute multiple tasks to generate logs
    tasks = ["Turn off Wi-Fi", "Enable Bluetooth", "Open Settings"]
    
    for task in tasks:
        agent.reset()
        obs = mock_env.reset()
        
        # Execute task
        for _ in range(5):
            info, actions = agent.predict(task, obs)
            if actions and "DONE" not in actions:
                obs = mock_env.step({"action_type": "touch", "element_id": "test_element"})
            if "DONE" in actions or "FAIL" in actions:
                break
    
    # Generate evaluation report
    try:
        report = agent.generate_evaluation_report()
        print(f"📊 Evaluation Report Generated:")
        print(f"   Total Goals: {report.total_goals}")
        print(f"   Successful Goals: {report.successful_goals}")
        print(f"   Failed Goals: {report.failed_goals}")
        print(f"   Overall Success Rate: {report.overall_success_rate:.1%}")
        print(f"   Issues Found: {len(report.issues)}")
        print(f"   Strengths: {len(report.strengths)}")
        print(f"   Recommendations: {len(report.recommendations)}")
        
    except Exception as e:
        print(f"   ⚠️  Report generation failed: {e}")
    
    print(f"\n✅ Evaluation report testing completed!")

if __name__ == "__main__":
    try:
        test_agent_s_integration()
        test_visual_trace_analysis()
        test_message_history()
        test_evaluation_report()
    except Exception as e:
        print(f"❌ Agent-S integration test failed with error: {e}")
        import traceback
        traceback.print_exc() 