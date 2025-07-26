#!/usr/bin/env python3
"""
Test script for the robust verifier agent.
This script demonstrates the improved verification functionality.
"""

import sys
import os
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.verifier_agent import VerifierAgent, VerificationStatus, VerificationResult
from utils.logger import QALogger
from config.qa_config import get_config

def create_mock_ui_tree(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a mock UI tree for testing."""
    return elements

def create_mock_observation(ui_tree: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Create a mock observation for testing."""
    obs = {"ui_tree": ui_tree}
    obs.update(kwargs)
    return obs

def test_verifier_agent():
    """Test the verifier agent with various scenarios."""
    print("🧪 Testing Robust Verifier Agent")
    print("=" * 50)
    
    # Create logger
    logger = QALogger(log_level="INFO", enable_console=True)
    
    # Get configuration
    config = get_config("default")
    
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
    
    # Test case 1: WiFi toggle - successful
    print("\n📋 Test 1: WiFi Toggle - Successful")
    print("-" * 40)
    
    prev_ui = create_mock_ui_tree([
        {"id": "wifi_switch", "text": "Wi-Fi", "class": "android.widget.Switch", "checked": True, "enabled": True},
        {"id": "wifi_label", "text": "Wi-Fi is on", "class": "android.widget.TextView", "enabled": True},
        {"id": "settings_button", "text": "Settings", "class": "android.widget.Button", "clickable": True}
    ])
    
    curr_ui = create_mock_ui_tree([
        {"id": "wifi_switch", "text": "Wi-Fi", "class": "android.widget.Switch", "checked": False, "enabled": True},
        {"id": "wifi_label", "text": "Wi-Fi is off", "class": "android.widget.TextView", "enabled": True},
        {"id": "settings_button", "text": "Settings", "class": "android.widget.Button", "clickable": True}
    ])
    
    prev_obs = create_mock_observation(prev_ui)
    curr_obs = create_mock_observation(curr_ui)
    
    result = verifier.verify("Turn Wi-Fi off", prev_obs, curr_obs)
    print(f"   Subgoal: Turn Wi-Fi off")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reason: {result.reason}")
    print(f"   Needs Replan: {result.needs_replan}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   ✅ PASS" if result.status == VerificationStatus.PASS else "   ❌ FAIL")
    
    # Test case 2: Bluetooth toggle - successful
    print("\n📋 Test 2: Bluetooth Toggle - Successful")
    print("-" * 40)
    
    prev_ui = create_mock_ui_tree([
        {"id": "bt_switch", "text": "Bluetooth", "class": "android.widget.Switch", "checked": False, "enabled": True},
        {"id": "bt_label", "text": "Bluetooth is off", "class": "android.widget.TextView", "enabled": True}
    ])
    
    curr_ui = create_mock_ui_tree([
        {"id": "bt_switch", "text": "Bluetooth", "class": "android.widget.Switch", "checked": True, "enabled": True},
        {"id": "bt_label", "text": "Bluetooth is on", "class": "android.widget.TextView", "enabled": True}
    ])
    
    prev_obs = create_mock_observation(prev_ui)
    curr_obs = create_mock_observation(curr_ui)
    
    result = verifier.verify("Enable Bluetooth", prev_obs, curr_obs)
    print(f"   Subgoal: Enable Bluetooth")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reason: {result.reason}")
    print(f"   Needs Replan: {result.needs_replan}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   ✅ PASS" if result.status == VerificationStatus.PASS else "   ❌ FAIL")
    
    # Test case 3: Settings navigation - successful
    print("\n📋 Test 3: Settings Navigation - Successful")
    print("-" * 40)
    
    prev_ui = create_mock_ui_tree([
        {"id": "home_button", "text": "Home", "class": "android.widget.Button", "clickable": True},
        {"id": "settings_button", "text": "Settings", "class": "android.widget.Button", "clickable": True}
    ])
    
    curr_ui = create_mock_ui_tree([
        {"id": "settings_header", "text": "Settings", "class": "android.widget.TextView", "enabled": True},
        {"id": "wifi_option", "text": "Wi-Fi", "class": "android.widget.ListItem", "clickable": True},
        {"id": "bluetooth_option", "text": "Bluetooth", "class": "android.widget.ListItem", "clickable": True},
        {"id": "back_button", "text": "Back", "class": "android.widget.Button", "clickable": True}
    ])
    
    prev_obs = create_mock_observation(prev_ui)
    curr_obs = create_mock_observation(curr_ui)
    
    result = verifier.verify("Open Settings", prev_obs, curr_obs)
    print(f"   Subgoal: Open Settings")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reason: {result.reason}")
    print(f"   Needs Replan: {result.needs_replan}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   ✅ PASS" if result.status == VerificationStatus.PASS else "   ❌ FAIL")
    
    # Test case 4: No change - should fail
    print("\n📋 Test 4: No Change - Should Fail")
    print("-" * 40)
    
    prev_ui = create_mock_ui_tree([
        {"id": "wifi_switch", "text": "Wi-Fi", "class": "android.widget.Switch", "checked": True, "enabled": True},
        {"id": "settings_button", "text": "Settings", "class": "android.widget.Button", "clickable": True}
    ])
    
    curr_ui = create_mock_ui_tree([
        {"id": "wifi_switch", "text": "Wi-Fi", "class": "android.widget.Switch", "checked": True, "enabled": True},
        {"id": "settings_button", "text": "Settings", "class": "android.widget.Button", "clickable": True}
    ])
    
    prev_obs = create_mock_observation(prev_ui)
    curr_obs = create_mock_observation(curr_ui)
    
    result = verifier.verify("Turn Wi-Fi off", prev_obs, curr_obs)
    print(f"   Subgoal: Turn Wi-Fi off")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reason: {result.reason}")
    print(f"   Needs Replan: {result.needs_replan}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   ✅ PASS" if result.status == VerificationStatus.FAIL else "   ❌ FAIL")
    
    # Test case 5: Invalid observations - should error
    print("\n📋 Test 5: Invalid Observations - Should Error")
    print("-" * 40)
    
    result = verifier.verify("Turn Wi-Fi off", {}, {})
    print(f"   Subgoal: Turn Wi-Fi off")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reason: {result.reason}")
    print(f"   Needs Replan: {result.needs_replan}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   ✅ PASS" if result.status == VerificationStatus.ERROR else "   ❌ FAIL")
    
    # Test case 6: Brightness adjustment - successful
    print("\n📋 Test 6: Brightness Adjustment - Successful")
    print("-" * 40)
    
    prev_ui = create_mock_ui_tree([
        {"id": "brightness_slider", "text": "Brightness", "class": "android.widget.SeekBar", "progress": 50, "enabled": True},
        {"id": "brightness_label", "text": "Brightness: 50%", "class": "android.widget.TextView", "enabled": True}
    ])
    
    curr_ui = create_mock_ui_tree([
        {"id": "brightness_slider", "text": "Brightness", "class": "android.widget.SeekBar", "progress": 80, "enabled": True},
        {"id": "brightness_label", "text": "Brightness: 80%", "class": "android.widget.TextView", "enabled": True}
    ])
    
    prev_obs = create_mock_observation(prev_ui)
    curr_obs = create_mock_observation(curr_ui)
    
    result = verifier.verify("Increase brightness", prev_obs, curr_obs)
    print(f"   Subgoal: Increase brightness")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Reason: {result.reason}")
    print(f"   Needs Replan: {result.needs_replan}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   ✅ PASS" if result.status == VerificationStatus.PASS else "   ❌ FAIL")
    
    # Print statistics
    print(f"\n📊 Verification Statistics:")
    stats = verifier.get_stats()
    for key, value in stats.items():
        if key == 'strategy_success_rates':
            print(f"   {key}:")
            for strategy, rates in value.items():
                success_rate = rates['success'] / max(rates['total'], 1) * 100
                print(f"     {strategy}: {success_rate:.1f}% ({rates['success']}/{rates['total']})")
        else:
            print(f"   {key}: {value}")
    
    # Test state history
    print(f"\n📈 State History Analysis:")
    state_history = verifier.get_state_history()
    print(f"   Total state changes tracked: {len(state_history)}")
    
    if state_history:
        latest_state = state_history[-1]
        print(f"   Latest state change:")
        print(f"     Subgoal: {latest_state['subgoal']}")
        print(f"     Previous elements: {latest_state['prev_element_count']}")
        print(f"     Current elements: {latest_state['curr_element_count']}")
        print(f"     Changes detected: {len(latest_state['changes'])}")
    
    print(f"\n✅ Verification testing completed!")

def test_verification_strategies():
    """Test individual verification strategies."""
    print(f"\n🔍 Testing Individual Verification Strategies")
    print("=" * 50)
    
    verifier = VerifierAgent()
    
    # Test UI change verification
    print("\n📋 UI Change Verification Strategy")
    print("-" * 30)
    
    prev_ui = create_mock_ui_tree([
        {"id": "element1", "text": "Old Text", "enabled": True},
        {"id": "element2", "text": "Static Text", "enabled": True}
    ])
    
    curr_ui = create_mock_ui_tree([
        {"id": "element1", "text": "New Text", "enabled": False},
        {"id": "element2", "text": "Static Text", "enabled": True},
        {"id": "element3", "text": "New Element", "enabled": True}
    ])
    
    result = verifier._ui_change_verification("Test subgoal", prev_ui, curr_ui)
    if result:
        print(f"   Change Ratio: {result['change_ratio']:.2f}")
        print(f"   Is Significant: {result['is_significant']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Added Elements: {result['added_elements']}")
        print(f"   Removed Elements: {result['removed_elements']}")
        print(f"   State Changes: {result['state_changes']}")
    
    # Test subgoal presence verification
    print("\n📋 Subgoal Presence Verification Strategy")
    print("-" * 30)
    
    ui_tree = create_mock_ui_tree([
        {"id": "wifi_switch", "text": "Wi-Fi", "class": "android.widget.Switch"},
        {"id": "bluetooth_switch", "text": "Bluetooth", "class": "android.widget.Switch"},
        {"id": "settings_button", "text": "Settings", "class": "android.widget.Button"}
    ])
    
    result = verifier._subgoal_presence_verification("Turn Wi-Fi off", ui_tree, ui_tree)
    if result:
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Key Terms: {result['key_terms']}")
        print(f"   Matches: {len(result['matches'])}")
        for match in result['matches']:
            print(f"     - {match['text']} (score: {match['match_score']:.2f})")
    
    # Test state transition verification
    print("\n📋 State Transition Verification Strategy")
    print("-" * 30)
    
    result = verifier._state_transition_verification("Turn Wi-Fi off", prev_ui, curr_ui)
    if result:
        print(f"   Action Type: {result['action_type']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Expected Changes: {result['expected_changes']}")
        print(f"   Actual Changes: {result['actual_changes']}")
        print(f"   Matches: {result['matches']}")

def test_configurations():
    """Test different configuration presets for verifier."""
    print(f"\n⚙️  Testing Verifier Configuration Presets")
    print("=" * 50)
    
    configs = ["default", "high_performance", "high_reliability"]
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n📋 {config_name.upper()} Configuration:")
        verifier_config = config.verifier
        
        print(f"   min_change_ratio: {verifier_config.min_change_ratio}")
        print(f"   fuzzy_threshold: {verifier_config.fuzzy_threshold}")
        print(f"   max_verification_time: {verifier_config.max_verification_time}")
        print(f"   enable_advanced_analysis: {verifier_config.enable_advanced_analysis}")
        print(f"   enable_state_tracking: {verifier_config.enable_state_tracking}")
        print(f"   min_confidence: {verifier_config.min_confidence}")
        print(f"   strategy_weights:")
        for strategy, weight in verifier_config.strategy_weights.items():
            print(f"     {strategy}: {weight}")

if __name__ == "__main__":
    try:
        test_verifier_agent()
        test_verification_strategies()
        test_configurations()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 