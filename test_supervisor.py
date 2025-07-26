#!/usr/bin/env python3
"""
Test script for the robust supervisor agent.
This script demonstrates the comprehensive analysis capabilities of the SupervisorAgent.
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.supervisor_agent import SupervisorAgent, EvaluationReport, TestOutcome, Severity
from utils.logger import QALogger
from config.qa_config import get_config

def create_sample_logs() -> List[Dict[str, Any]]:
    """Create sample QA logs for testing."""
    return [
        {
            "goal": "Turn off Wi-Fi",
            "status": "success",
            "success_rate": 1.0,
            "execution_time": 2.5,
            "iterations": 1,
            "completed_subgoals": ["Open Settings", "Toggle Wi-Fi"],
            "failed_subgoals": [],
            "reason": None,
            "planning_result": {
                "status": "success",
                "planning_time": 0.1,
                "confidence": 0.9,
                "strategies_used": ["template_based_planning", "semantic_planning"]
            },
            "stats": {
                "total_replans": 0,
                "total_retries": 0,
                "planner_stats": {
                    "total_plans": 1,
                    "successful_plans": 1,
                    "average_planning_time": 0.1
                },
                "executor_stats": {
                    "total_executions": 2,
                    "successful_executions": 2,
                    "average_execution_time": 1.25,
                    "total_retries": 0
                },
                "verifier_stats": {
                    "total_verifications": 2,
                    "successful_verifications": 2,
                    "average_verification_time": 0.05
                }
            }
        },
        {
            "goal": "Enable Bluetooth and open Developer Options",
            "status": "failed",
            "success_rate": 0.5,
            "execution_time": 8.2,
            "iterations": 3,
            "completed_subgoals": ["Open Settings", "Enable Bluetooth"],
            "failed_subgoals": ["Open Developer Options"],
            "reason": "Developer Options not found",
            "planning_result": {
                "status": "success",
                "planning_time": 0.3,
                "confidence": 0.8,
                "strategies_used": ["template_based_planning", "adaptive_planning"]
            },
            "stats": {
                "total_replans": 2,
                "total_retries": 3,
                "planner_stats": {
                    "total_plans": 3,
                    "successful_plans": 3,
                    "average_planning_time": 0.3
                },
                "executor_stats": {
                    "total_executions": 5,
                    "successful_executions": 3,
                    "average_execution_time": 1.64,
                    "total_retries": 3
                },
                "verifier_stats": {
                    "total_verifications": 5,
                    "successful_verifications": 3,
                    "average_verification_time": 0.08
                }
            }
        },
        {
            "goal": "Configure device settings",
            "status": "success",
            "success_rate": 1.0,
            "execution_time": 4.1,
            "iterations": 2,
            "completed_subgoals": ["Open Settings", "Navigate to System", "Configure Display"],
            "failed_subgoals": [],
            "reason": None,
            "planning_result": {
                "status": "success",
                "planning_time": 0.2,
                "confidence": 0.85,
                "strategies_used": ["semantic_planning", "template_based_planning"]
            },
            "stats": {
                "total_replans": 0,
                "total_retries": 1,
                "planner_stats": {
                    "total_plans": 1,
                    "successful_plans": 1,
                    "average_planning_time": 0.2
                },
                "executor_stats": {
                    "total_executions": 3,
                    "successful_executions": 3,
                    "average_execution_time": 1.37,
                    "total_retries": 1
                },
                "verifier_stats": {
                    "total_verifications": 3,
                    "successful_verifications": 3,
                    "average_verification_time": 0.06
                }
            }
        },
        {
            "goal": "Make coffee with the phone",
            "status": "failed",
            "success_rate": 0.0,
            "execution_time": 1.2,
            "iterations": 1,
            "completed_subgoals": [],
            "failed_subgoals": ["Open Coffee App"],
            "reason": "Impossible goal",
            "planning_result": {
                "status": "success",
                "planning_time": 0.1,
                "confidence": 0.9,
                "strategies_used": ["fallback_planning"]
            },
            "stats": {
                "total_replans": 0,
                "total_retries": 0,
                "planner_stats": {
                    "total_plans": 1,
                    "successful_plans": 1,
                    "average_planning_time": 0.1
                },
                "executor_stats": {
                    "total_executions": 1,
                    "successful_executions": 0,
                    "average_execution_time": 1.2,
                    "total_retries": 0
                },
                "verifier_stats": {
                    "total_verifications": 1,
                    "successful_verifications": 0,
                    "average_verification_time": 0.01
                }
            }
        }
    ]

def test_supervisor_agent():
    """Test the robust supervisor agent."""
    print("🧠 Testing Robust Supervisor Agent")
    print("=" * 60)
    
    # Create sample logs
    sample_logs = create_sample_logs()
    
    # Save sample logs to file
    with open("qa_logs.json", "w") as f:
        json.dump(sample_logs, f, indent=2)
    
    print("📝 Created sample QA logs with various scenarios:")
    for i, log in enumerate(sample_logs, 1):
        print(f"   {i}. {log['goal']} - {log['status']} ({log['success_rate']:.0%} success)")
    
    # Create supervisor agent
    supervisor = SupervisorAgent(
        logger=QALogger(log_level="INFO", enable_console=True),
        enable_visual_analysis=False,
        confidence_threshold=0.7,
        flaky_threshold=0.3
    )
    
    # Test configuration
    config = get_config("default")
    
    # Evaluate logs
    print(f"\n🔍 Analyzing QA logs...")
    report = supervisor.evaluate_logs(
        log_path="qa_logs.json",
        config={"name": "default", "settings": config.__dict__},
        test_context={"test_type": "integration", "environment": "mock"}
    )
    
    # Display results
    print(f"\n📊 Evaluation Results")
    print("=" * 60)
    print(f"   Total Goals: {report.total_goals}")
    print(f"   Successful Goals: {report.successful_goals}")
    print(f"   Failed Goals: {report.failed_goals}")
    print(f"   Flaky Goals: {report.flaky_goals}")
    print(f"   Overall Success Rate: {report.overall_success_rate:.1%}")
    
    print(f"\n⏱️ Performance Metrics:")
    print(f"   Average Planning Time: {report.avg_planning_time:.2f}s")
    print(f"   Average Execution Time: {report.avg_execution_time:.2f}s")
    print(f"   Average Verification Time: {report.avg_verification_time:.2f}s")
    print(f"   Average Total Time: {report.avg_total_time:.2f}s")
    
    print(f"\n🔄 System Metrics:")
    print(f"   Total Replans: {report.total_replans}")
    print(f"   Total Retries: {report.total_retries}")
    print(f"   Average Retries per Goal: {report.avg_retries_per_goal:.1f}")
    print(f"   Average Replans per Goal: {report.avg_replans_per_goal:.1f}")
    
    print(f"\n🤖 Agent Performance:")
    print(f"   Planner Success Rate: {report.planner_performance.success_rate:.1%}")
    print(f"   Executor Success Rate: {report.executor_performance.success_rate:.1%}")
    print(f"   Verifier Success Rate: {report.verifier_performance.success_rate:.1%}")
    
    if report.flaky_subgoals:
        print(f"\n🔄 Flaky Subgoals:")
        for subgoal in report.flaky_subgoals:
            print(f"   - {subgoal}")
    
    if report.issues:
        print(f"\n🚨 Issues Detected:")
        for issue in report.issues:
            print(f"   [{issue.severity.value.upper()}] {issue.category}: {issue.description}")
            print(f"       Occurrences: {issue.occurrence_count}")
            for rec in issue.recommendations:
                print(f"       💡 {rec}")
    
    print(f"\n✅ Strengths:")
    for strength in report.strengths:
        print(f"   - {strength}")
    
    print(f"\n⚠️ Weaknesses:")
    for weakness in report.weaknesses:
        print(f"   - {weakness}")
    
    print(f"\n💡 Recommendations:")
    for rec in report.recommendations:
        print(f"   - {rec}")
    
    # Test subgoal analysis
    print(f"\n📋 Subgoal Analysis:")
    print("=" * 60)
    for name, analysis in report.subgoal_analysis.items():
        print(f"   {name}:")
        print(f"     Success Rate: {analysis.success_rate:.1%}")
        print(f"     Attempts: {analysis.total_attempts}")
        print(f"     Flaky: {'Yes' if analysis.is_flaky else 'No'}")
        if analysis.recommendations:
            print(f"     Recommendations: {', '.join(analysis.recommendations)}")
    
    # Test statistics
    stats = supervisor.get_stats()
    print(f"\n📈 Supervisor Statistics:")
    print(f"   Total Evaluations: {stats['total_evaluations']}")
    print(f"   Average Analysis Time: {stats['avg_analysis_time']:.2f}s")
    
    print(f"\n✅ Supervisor agent testing completed!")

def test_error_handling():
    """Test error handling in the supervisor agent."""
    print(f"\n🚨 Testing Error Handling")
    print("=" * 60)
    
    # Create supervisor agent
    supervisor = SupervisorAgent()
    
    # Test with non-existent log file
    print("\n📋 Test: Non-existent log file")
    print("-" * 30)
    
    report = supervisor.evaluate_logs("non_existent_logs.json")
    print(f"   Status: {'Error' if report.total_goals == 0 else 'Success'}")
    print(f"   Issues: {len(report.issues)}")
    
    # Test with invalid log format
    print("\n📋 Test: Invalid log format")
    print("-" * 30)
    
    with open("invalid_logs.json", "w") as f:
        json.dump({"invalid": "format"}, f)
    
    report = supervisor.evaluate_logs("invalid_logs.json")
    print(f"   Status: {'Error' if report.total_goals == 0 else 'Success'}")
    print(f"   Issues: {len(report.issues)}")
    
    # Clean up
    if os.path.exists("invalid_logs.json"):
        os.remove("invalid_logs.json")
    
    print(f"\n✅ Error handling testing completed!")

def test_report_formats():
    """Test different report formats."""
    print(f"\n📄 Testing Report Formats")
    print("=" * 60)
    
    # Create sample logs
    sample_logs = create_sample_logs()
    with open("qa_logs.json", "w") as f:
        json.dump(sample_logs, f, indent=2)
    
    # Create supervisor agent
    supervisor = SupervisorAgent()
    
    # Generate report
    report = supervisor.evaluate_logs("qa_logs.json")
    
    # Check if reports were generated
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_report_path = f"evaluation_report_{timestamp}.json"
    md_report_path = f"evaluation_report_{timestamp}.md"
    
    print(f"\n📋 Generated Reports:")
    print(f"   JSON Report: {json_report_path}")
    print(f"   Markdown Report: {md_report_path}")
    
    # Check if files exist
    if os.path.exists(json_report_path):
        print(f"   ✅ JSON report created successfully")
    else:
        print(f"   ❌ JSON report not found")
    
    if os.path.exists(md_report_path):
        print(f"   ✅ Markdown report created successfully")
    else:
        print(f"   ❌ Markdown report not found")
    
    print(f"\n✅ Report format testing completed!")

if __name__ == "__main__":
    try:
        test_supervisor_agent()
        test_error_handling()
        test_report_formats()
    except Exception as e:
        print(f"❌ Supervisor agent test failed with error: {e}")
        import traceback
        traceback.print_exc() 