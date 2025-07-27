#!/usr/bin/env python3
"""
Enhanced robust agent loop with anti-flaky behavior mechanisms.
This demonstrates how to use the PlannerAgent, ExecutorAgent, and VerifierAgent together
with improved stability and reliability.
"""

import sys
import os
import argparse
import time
import random
from typing import Dict, Any
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_full_integration import RobustAgentLoop, MockEnvironment
from utils.logger import QALogger
from config.qa_config import get_config

class EnhancedRobustAgentLoop(RobustAgentLoop):
    """Enhanced agent loop with anti-flaky behavior mechanisms."""
    
    def __init__(self, env, config_name: str = "default"):
        super().__init__(env, config_name)
        self.flaky_behavior_detector = FlakyBehaviorDetector()
        self.goal_execution_history = {}
        self.adaptive_retry_strategies = AdaptiveRetryStrategies()
    
    def execute_goal_with_stability_checks(self, goal: str, max_iterations: int = 10) -> Dict[str, Any]:
        """Execute goal with enhanced stability checks and anti-flaky mechanisms."""
        
        # Pre-execution stability check
        if not self._pre_execution_stability_check():
            self.logger.warning("StabilityCheck", "Environment not stable, waiting...")
            time.sleep(2.0)
        
        # Check historical performance for this goal
        goal_history = self.goal_execution_history.get(goal, [])
        if len(goal_history) >= 3:
            flaky_score = self.flaky_behavior_detector.calculate_goal_flakiness(goal_history)
            if flaky_score > 0.5:
                self.logger.warning("FlakyDetector", f"Goal '{goal}' shows flaky behavior (score: {flaky_score:.2f})")
                # Apply enhanced retry strategy for flaky goals
                return self._execute_flaky_goal_with_enhanced_strategy(goal, max_iterations)
        
        # Standard execution with monitoring
        result = self._execute_with_monitoring(goal, max_iterations)
        
        # Record execution for flaky behavior analysis
        self.goal_execution_history.setdefault(goal, []).append({
            'timestamp': time.time(),
            'success': result['status'] == 'success',
            'execution_time': result['execution_time'],
            'iterations': result['iterations']
        })
        
        return result
    
    def _pre_execution_stability_check(self) -> bool:
        """Check if environment is stable before execution."""
        try:
            # Check if environment responds properly
            obs = self.env.get_observation()
            if not obs or not obs.get('ui_tree'):
                return False
            
            # Check for expected UI elements
            ui_tree = obs.get('ui_tree', [])
            if len(ui_tree) < 2:  # Minimum expected UI elements
                return False
            
            return True
        except Exception as e:
            self.logger.error("StabilityCheck", f"Environment stability check failed: {e}")
            return False
    
    def _execute_flaky_goal_with_enhanced_strategy(self, goal: str, max_iterations: int) -> Dict[str, Any]:
        """Execute a known flaky goal with enhanced retry strategy."""
        self.logger.info("EnhancedExecution", f"Using enhanced strategy for flaky goal: {goal}")
        
        # Use adaptive retry with longer delays
        retry_delays = [2.0, 4.0, 6.0, 8.0, 10.0]
        
        for attempt in range(min(len(retry_delays), max_iterations)):
            self.logger.info("EnhancedExecution", f"Attempt {attempt + 1} for flaky goal")
            
            # Extended stability check
            if not self._extended_stability_check():
                self.logger.warning("EnhancedExecution", "Extended stability check failed, retrying...")
                time.sleep(retry_delays[attempt])
                continue
            
            # Execute with enhanced monitoring
            result = self._execute_with_monitoring(goal, max_iterations // len(retry_delays))
            
            if result['status'] == 'success':
                self.logger.info("EnhancedExecution", f"Flaky goal succeeded on attempt {attempt + 1}")
                return result
            
            # Wait with exponential backoff
            if attempt < len(retry_delays) - 1:
                delay = retry_delays[attempt]
                self.logger.info("EnhancedExecution", f"Waiting {delay}s before retry...")
                time.sleep(delay)
        
        # Final attempt with standard strategy
        return self._execute_with_monitoring(goal, max_iterations)
    
    def _extended_stability_check(self) -> bool:
        """Extended stability check for flaky scenarios."""
        checks = []
        
        for i in range(3):  # Multiple checks
            checks.append(self._pre_execution_stability_check())
            if i < 2:  # Don't wait after last check
                time.sleep(0.5)
        
        # Require at least 2 out of 3 checks to pass
        return sum(checks) >= 2
    
    def _execute_with_monitoring(self, goal: str, max_iterations: int) -> Dict[str, Any]:
        """Execute goal with enhanced monitoring and stability tracking."""
        start_time = time.time()
        
        # Monitor execution with stability tracking
        stability_scores = []
        
        result = super().execute_goal(goal, max_iterations)
        
        # Calculate stability score for this execution
        execution_time = time.time() - start_time
        stability_score = self._calculate_execution_stability_score(result, execution_time)
        
        result['stability_score'] = stability_score
        result['enhanced_execution'] = True
        
        return result
    
    def _calculate_execution_stability_score(self, result: Dict[str, Any], execution_time: float) -> float:
        """Calculate stability score for an execution."""
        score = 1.0
        
        # Penalize for failures
        if result['status'] != 'success':
            score *= 0.5
        
        # Penalize for too many iterations
        if result['iterations'] > 5:
            score *= 0.8
        
        # Penalize for very long execution times
        if execution_time > 30.0:
            score *= 0.7
        
        # Bonus for quick successful execution
        if result['status'] == 'success' and execution_time < 10.0:
            score *= 1.1
        
        return min(score, 1.0)


class FlakyBehaviorDetector:
    """Detects flaky behavior patterns in goal execution."""
    
    def calculate_goal_flakiness(self, execution_history: list) -> float:
        """Calculate flakiness score for a goal based on execution history."""
        if len(execution_history) < 3:
            return 0.0
        
        # Analyze success/failure pattern
        recent_executions = execution_history[-10:]  # Last 10 executions
        successes = [ex['success'] for ex in recent_executions]
        
        # Calculate transition rate (success -> failure -> success indicates flakiness)
        transitions = sum(1 for i in range(1, len(successes)) if successes[i] != successes[i-1])
        max_transitions = len(successes) - 1
        
        transition_rate = transitions / max_transitions if max_transitions > 0 else 0
        
        # Calculate variance in execution time
        execution_times = [ex['execution_time'] for ex in recent_executions]
        if len(execution_times) > 1:
            import statistics
            mean_time = statistics.mean(execution_times)
            std_time = statistics.stdev(execution_times)
            time_variance = std_time / mean_time if mean_time > 0 else 0
        else:
            time_variance = 0
        
        # Combine metrics
        flaky_score = (transition_rate * 0.7) + (min(time_variance, 1.0) * 0.3)
        
        return min(flaky_score, 1.0)


class AdaptiveRetryStrategies:
    """Adaptive retry strategies based on failure patterns."""
    
    def get_retry_strategy(self, goal: str, failure_history: list) -> Dict[str, Any]:
        """Get appropriate retry strategy for a goal based on failure history."""
        if "wifi" in goal.lower():
            return {
                'max_retries': 5,
                'retry_delays': [1.0, 2.0, 4.0, 6.0, 8.0],
                'pre_retry_actions': ['reset_wifi_state', 'check_network_settings']
            }
        elif "bluetooth" in goal.lower():
            return {
                'max_retries': 4,
                'retry_delays': [2.0, 4.0, 6.0, 8.0],
                'pre_retry_actions': ['reset_bluetooth_state']
            }
        else:
            return {
                'max_retries': 3,
                'retry_delays': [1.0, 2.0, 4.0],
                'pre_retry_actions': []
            }


def main():
    """Main function to run the robust agent loop."""
    parser = argparse.ArgumentParser(description="QualGent QA System - Robust Agent Loop")
    parser.add_argument("--goal", type=str, required=True, help="High-level goal to execute")
    parser.add_argument("--config", type=str, default="default", 
                       choices=["default", "high_performance", "high_reliability"],
                       help="Configuration preset to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--log-file", type=str, help="Log file path")
    args = parser.parse_args()
    
    print("QualGent QA System - Robust Agent Loop")
    print("=" * 50)
    print()
    
    try:
        # Create environment (mock for now, replace with real AndroidEnv in production)
        env = MockEnvironment()
        
        # Create robust agent loop
        agent_loop = RobustAgentLoop(env, args.config)
        
        # Execute goal
        print(f"Executing goal: {args.goal}")
        print(f"Configuration: {args.config}")
        print()
        
        start_time = time.time()
        result = agent_loop.execute_goal(args.goal)
        execution_time = time.time() - start_time
        
        # Print results
        print("Execution Results:")
        print("-" * 20)
        print(f"   Status: {result['status']}")
        print(f"   Success Rate: {result['success_rate']:.1%}")
        print(f"   Execution Time: {execution_time:.2f}s")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Completed Subgoals: {result['completed_subgoals']}")
        print(f"   Failed Subgoals: {result['failed_subgoals']}")
        
        if result['completed_subgoals_list']:
            print(f"   Completed: {', '.join(result['completed_subgoals_list'])}")
        
        if result['failed_subgoals_list']:
            print(f"   Failed: {', '.join(result['failed_subgoals_list'])}")
        
        print()
        
        # Print planning details
        if result.get('planning_status'):
            print("Planning Details:")
            print("-" * 20)
            print(f"   Status: {result['planning_status']}")
            print(f"   Confidence: {result['planning_confidence']:.2f}")
            print(f"   Planning Time: {result['planning_time']:.2f}s")
            print(f"   Strategies Used: {', '.join(result['strategies_used'])}")
            print()
        
        # Print statistics
        stats = agent_loop.get_stats()
        print("System Statistics:")
        print("-" * 20)
        print(f"   Total Goals: {stats['loop_stats']['total_goals']}")
        print(f"   Success Rate: {stats['loop_stats']['successful_goals'] / max(stats['loop_stats']['total_goals'], 1) * 100:.1f}%")
        print(f"   Average Goal Time: {stats['loop_stats']['average_goal_completion_time']:.2f}s")
        print(f"   Total Plans: {stats['loop_stats']['total_plans']}")
        print(f"   Total Replans: {stats['loop_stats']['total_replans']}")
        print()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"robust_loop_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "goal": args.goal,
                "config": args.config,
                "result": result,
                "statistics": stats,
                "timestamp": timestamp
            }, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        print()
        print("Robust agent loop execution completed!")
        
        return 0 if result['status'] == 'success' else 1
        
    except Exception as e:
        print(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
