#!/usr/bin/env python3
"""
Main script for running the robust agent loop.
This demonstrates how to use the PlannerAgent, ExecutorAgent, and VerifierAgent together.
"""

import sys
import os
import argparse
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_full_integration import RobustAgentLoop, MockEnvironment
from utils.logger import QALogger
from config.qa_config import get_config

def main():
    """Main function to run the robust agent loop."""
    parser = argparse.ArgumentParser(description="Run the robust agent loop")
    parser.add_argument("--goal", "-g", type=str, required=True,
                       help="High-level goal to achieve (e.g., 'Turn off Wi-Fi and enable Bluetooth')")
    parser.add_argument("--config", "-c", type=str, default="default",
                       choices=["default", "high_performance", "high_reliability"],
                       help="Configuration preset to use")
    parser.add_argument("--max-iterations", "-i", type=int, default=10,
                       help="Maximum iterations for the loop")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = QALogger(log_level=log_level, enable_console=True)
    
    logger.info("Main", "Starting robust agent loop", {
        "goal": args.goal,
        "config": args.config,
        "max_iterations": args.max_iterations
    })
    
    try:
        # Create mock environment (replace with real AndroidEnv in production)
        mock_env = MockEnvironment()
        
        # Create robust agent loop
        agent_loop = RobustAgentLoop(mock_env, args.config)
        
        # Execute goal
        result = agent_loop.execute_goal(args.goal, args.max_iterations)
        
        # Print results
        print("\n" + "="*60)
        print("🎯 GOAL EXECUTION RESULTS")
        print("="*60)
        print(f"Goal: {result['goal']}")
        print(f"Status: {result['status'].upper()}")
        print(f"Success Rate: {result['success_rate']:.1%}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Iterations: {result['iterations']}")
        
        print(f"\n📋 SUBGOALS:")
        print(f"  Completed ({len(result['completed_subgoals'])}):")
        for subgoal in result['completed_subgoals']:
            print(f"    ✅ {subgoal}")
        
        if result['failed_subgoals']:
            print(f"  Failed ({len(result['failed_subgoals'])}):")
            for subgoal in result['failed_subgoals']:
                print(f"    ❌ {subgoal}")
        
        # Print planning details
        if result['planning_result']:
            plan_result = result['planning_result']
            print(f"\n📝 PLANNING DETAILS:")
            print(f"  Status: {plan_result.status.value}")
            print(f"  Confidence: {plan_result.confidence:.2f}")
            print(f"  Planning Time: {plan_result.planning_time:.2f}s")
            print(f"  Strategies Used: {', '.join(plan_result.strategies_used)}")
            print(f"  Plan Complexity: {plan_result.plan_complexity}")
        
        # Print statistics
        stats = agent_loop.get_stats()
        print(f"\n📊 STATISTICS:")
        print(f"  Loop:")
        print(f"    Total Goals: {stats['loop_stats']['total_goals']}")
        print(f"    Success Rate: {stats['loop_stats']['successful_goals'] / max(stats['loop_stats']['total_goals'], 1) * 100:.1f}%")
        print(f"    Total Plans: {stats['loop_stats']['total_plans']}")
        print(f"    Total Replans: {stats['loop_stats']['total_replans']}")
        print(f"    Total Executions: {stats['loop_stats']['total_executions']}")
        print(f"    Total Verifications: {stats['loop_stats']['total_verifications']}")
        
        print(f"  Planner:")
        planner_stats = stats['planner_stats']
        print(f"    Success Rate: {planner_stats['successful_plans'] / max(planner_stats['total_plans'], 1) * 100:.1f}%")
        print(f"    Avg Planning Time: {planner_stats['average_planning_time']:.2f}s")
        
        print(f"  Executor:")
        executor_stats = stats['executor_stats']
        print(f"    Success Rate: {executor_stats['successful_executions'] / max(executor_stats['total_executions'], 1) * 100:.1f}%")
        print(f"    Avg Execution Time: {executor_stats['average_execution_time']:.2f}s")
        
        print(f"  Verifier:")
        verifier_stats = stats['verifier_stats']
        print(f"    Success Rate: {verifier_stats['successful_verifications'] / max(verifier_stats['total_verifications'], 1) * 100:.1f}%")
        print(f"    Avg Verification Time: {verifier_stats['average_verification_time']:.2f}s")
        
        # Final result
        if result['status'] == 'success':
            print(f"\n🎉 SUCCESS! Goal achieved with {result['success_rate']:.1%} success rate.")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS. Goal partially achieved with {result['success_rate']:.1%} success rate.")
        
        return 0 if result['status'] == 'success' else 1
        
    except Exception as e:
        logger.error("Main", "Error in robust agent loop", {"error": str(e)})
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
    
    
