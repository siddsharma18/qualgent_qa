#!/usr/bin/env python3
"""
Test script for the robust planner agent.
This script demonstrates the improved planning functionality.
"""

import sys
import os
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.planner_agent import PlannerAgent, PlanningStatus, PlanningResult, Subgoal
from utils.logger import QALogger
from config.qa_config import get_config

def test_planner_agent():
    """Test the planner agent with various scenarios."""
    print("🧠 Testing Robust Planner Agent")
    print("=" * 50)
    
    # Create logger
    logger = QALogger(log_level="INFO", enable_console=True)
    
    # Get configuration
    config = get_config("default")
    
    # Create planner agent
    planner = PlannerAgent(
        logger=logger,
        enable_template_matching=config.planner.enable_template_matching,
        enable_semantic_planning=config.planner.enable_semantic_planning,
        enable_adaptive_planning=config.planner.enable_adaptive_planning,
        planning_timeout=config.planner.max_planning_time,
        min_confidence_threshold=config.planner.min_confidence
    )
    
    # Test case 1: WiFi management - template matching
    print("\n📋 Test 1: WiFi Management - Template Matching")
    print("-" * 40)
    
    context = {
        'current_app': 'home',
        'ui_elements': ['settings_button', 'wifi_toggle'],
        'ui_state': 'home_screen',
        'previous_actions': []
    }
    
    result = planner.plan("Turn off Wi-Fi", context)
    
    print(f"   Goal: Turn off Wi-Fi")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Total Duration: {result.total_estimated_duration:.1f}s")
    print(f"   Plan Complexity: {result.plan_complexity}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   Number of Subgoals: {len(result.subgoals)}")
    
    for i, subgoal in enumerate(result.subgoals, 1):
        print(f"     {i}. {subgoal.name}")
        print(f"        Description: {subgoal.description}")
        print(f"        Priority: {subgoal.priority}")
        print(f"        Duration: {subgoal.estimated_duration:.1f}s")
        print(f"        Confidence: {subgoal.confidence:.2f}")
    
    print(f"   ✅ PASS" if result.status == PlanningStatus.SUCCESS else "   ❌ FAIL")
    
    # Test case 2: Bluetooth management - semantic planning
    print("\n📋 Test 2: Bluetooth Management - Semantic Planning")
    print("-" * 40)
    
    context = {
        'current_app': 'settings',
        'ui_elements': ['bluetooth_option', 'connected_devices'],
        'ui_state': 'settings_screen',
        'previous_actions': ['Open Settings']
    }
    
    result = planner.plan("Enable Bluetooth", context)
    
    print(f"   Goal: Enable Bluetooth")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Total Duration: {result.total_estimated_duration:.1f}s")
    print(f"   Plan Complexity: {result.plan_complexity}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   Number of Subgoals: {len(result.subgoals)}")
    
    for i, subgoal in enumerate(result.subgoals, 1):
        print(f"     {i}. {subgoal.name}")
        print(f"        Description: {subgoal.description}")
        print(f"        Priority: {subgoal.priority}")
        print(f"        Duration: {subgoal.estimated_duration:.1f}s")
        print(f"        Confidence: {subgoal.confidence:.2f}")
    
    print(f"   ✅ PASS" if result.status == PlanningStatus.SUCCESS else "   ❌ FAIL")
    
    # Test case 3: Developer options - complex task
    print("\n📋 Test 3: Developer Options - Complex Task")
    print("-" * 40)
    
    context = {
        'current_app': 'home',
        'ui_elements': ['settings_button', 'about_phone'],
        'ui_state': 'home_screen',
        'previous_actions': []
    }
    
    result = planner.plan("Enable Developer Options", context)
    
    print(f"   Goal: Enable Developer Options")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Total Duration: {result.total_estimated_duration:.1f}s")
    print(f"   Plan Complexity: {result.plan_complexity}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   Number of Subgoals: {len(result.subgoals)}")
    
    for i, subgoal in enumerate(result.subgoals, 1):
        print(f"     {i}. {subgoal.name}")
        print(f"        Description: {subgoal.description}")
        print(f"        Priority: {subgoal.priority}")
        print(f"        Duration: {subgoal.estimated_duration:.1f}s")
        print(f"        Confidence: {subgoal.confidence:.2f}")
    
    print(f"   ✅ PASS" if result.status == PlanningStatus.SUCCESS else "   ❌ FAIL")
    
    # Test case 4: Unknown task - fallback planning
    print("\n📋 Test 4: Unknown Task - Fallback Planning")
    print("-" * 40)
    
    context = {
        'current_app': 'home',
        'ui_elements': ['settings_button'],
        'ui_state': 'home_screen',
        'previous_actions': []
    }
    
    result = planner.plan("Configure Advanced System Settings", context)
    
    print(f"   Goal: Configure Advanced System Settings")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Total Duration: {result.total_estimated_duration:.1f}s")
    print(f"   Plan Complexity: {result.plan_complexity}")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   Number of Subgoals: {len(result.subgoals)}")
    
    for i, subgoal in enumerate(result.subgoals, 1):
        print(f"     {i}. {subgoal.name}")
        print(f"        Description: {subgoal.description}")
        print(f"        Priority: {subgoal.priority}")
        print(f"        Duration: {subgoal.estimated_duration:.1f}s")
        print(f"        Confidence: {subgoal.confidence:.2f}")
    
    print(f"   ✅ PASS" if result.status in [PlanningStatus.SUCCESS, PlanningStatus.PARTIAL] else "   ❌ FAIL")
    
    # Test case 5: Invalid goal - should fail
    print("\n📋 Test 5: Invalid Goal - Should Fail")
    print("-" * 40)
    
    result = planner.plan("", context)
    
    print(f"   Goal: (empty)")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Total Duration: {result.total_estimated_duration:.1f}s")
    print(f"   Strategies Used: {result.strategies_used}")
    print(f"   Number of Subgoals: {len(result.subgoals)}")
    
    print(f"   ✅ PASS" if result.status == PlanningStatus.FAILED else "   ❌ FAIL")
    
    # Test case 6: Replanning after failure
    print("\n📋 Test 6: Replanning After Failure")
    print("-" * 40)
    
    failed_subgoal = "Access Wi-Fi Settings"
    previous_subgoals = ["Open Settings", "Navigate to Network"]
    
    replan_result = planner.replan(failed_subgoal, previous_subgoals, "Turn off Wi-Fi", context)
    
    print(f"   Original Goal: Turn off Wi-Fi")
    print(f"   Failed Subgoal: {failed_subgoal}")
    print(f"   Previous Subgoals: {previous_subgoals}")
    print(f"   Replan Status: {replan_result.status.value}")
    print(f"   Replan Confidence: {replan_result.confidence:.2f}")
    print(f"   Alternative Plans: {len(replan_result.alternative_plans)}")
    print(f"   Number of Subgoals: {len(replan_result.subgoals)}")
    
    for i, subgoal in enumerate(replan_result.subgoals, 1):
        print(f"     {i}. {subgoal.name}")
        print(f"        Description: {subgoal.description}")
        print(f"        Priority: {subgoal.priority}")
        print(f"        Duration: {subgoal.estimated_duration:.1f}s")
        print(f"        Confidence: {subgoal.confidence:.2f}")
    
    print(f"   ✅ PASS" if replan_result.status == PlanningStatus.SUCCESS else "   ❌ FAIL")
    
    # Print statistics
    print(f"\n📊 Planning Statistics:")
    stats = planner.get_stats()
    for key, value in stats.items():
        if key == 'strategy_success_rates':
            print(f"   {key}:")
            for strategy, rates in value.items():
                success_rate = rates['success'] / max(rates['total'], 1) * 100
                print(f"     {strategy}: {success_rate:.1f}% ({rates['success']}/{rates['total']})")
        elif key == 'plan_complexity_distribution':
            print(f"   {key}:")
            for complexity, count in value.items():
                print(f"     {complexity}: {count}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n✅ Planning testing completed!")

def test_planning_strategies():
    """Test individual planning strategies."""
    print(f"\n🔍 Testing Individual Planning Strategies")
    print("=" * 50)
    
    planner = PlannerAgent()
    
    # Test template-based planning
    print("\n📋 Template-Based Planning Strategy")
    print("-" * 30)
    
    result = planner._template_based_planning("Turn off Wi-Fi")
    if result:
        print(f"   Number of Subgoals: {len(result)}")
        for i, subgoal in enumerate(result, 1):
            print(f"     {i}. {subgoal.name} (confidence: {subgoal.confidence:.2f})")
    else:
        print("   No template match found")
    
    # Test semantic planning
    print("\n📋 Semantic Planning Strategy")
    print("-" * 30)
    
    result = planner._semantic_planning("Increase brightness")
    if result:
        print(f"   Number of Subgoals: {len(result)}")
        for i, subgoal in enumerate(result, 1):
            print(f"     {i}. {subgoal.name} (confidence: {subgoal.confidence:.2f})")
    else:
        print("   No semantic match found")
    
    # Test adaptive planning
    print("\n📋 Adaptive Planning Strategy")
    print("-" * 30)
    
    context = {
        'current_app': 'settings',
        'ui_elements': ['wifi_option', 'network_option'],
        'ui_state': 'settings_screen'
    }
    
    result = planner._adaptive_planning("Turn on Wi-Fi", context)
    if result:
        print(f"   Number of Subgoals: {len(result)}")
        for i, subgoal in enumerate(result, 1):
            print(f"     {i}. {subgoal.name} (confidence: {subgoal.confidence:.2f})")
    else:
        print("   No adaptive plan generated")
    
    # Test fallback planning
    print("\n📋 Fallback Planning Strategy")
    print("-" * 30)
    
    result = planner._fallback_planning("Unknown task")
    if result:
        print(f"   Number of Subgoals: {len(result)}")
        for i, subgoal in enumerate(result, 1):
            print(f"     {i}. {subgoal.name} (confidence: {subgoal.confidence:.2f})")
    else:
        print("   No fallback plan generated")

def test_configurations():
    """Test different configuration presets for planner."""
    print(f"\n⚙️  Testing Planner Configuration Presets")
    print("=" * 50)
    
    configs = ["default", "high_performance", "high_reliability"]
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n📋 {config_name.upper()} Configuration:")
        planner_config = config.planner
        
        print(f"   enable_template_matching: {planner_config.enable_template_matching}")
        print(f"   enable_semantic_planning: {planner_config.enable_semantic_planning}")
        print(f"   enable_adaptive_planning: {planner_config.enable_adaptive_planning}")
        print(f"   max_planning_time: {planner_config.max_planning_time}")
        print(f"   min_confidence: {planner_config.min_confidence}")
        print(f"   enable_plan_optimization: {planner_config.enable_plan_optimization}")
        print(f"   strategy_weights:")
        for strategy, weight in planner_config.strategy_weights.items():
            print(f"     {strategy}: {weight}")

if __name__ == "__main__":
    try:
        test_planner_agent()
        test_planning_strategies()
        test_configurations()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 