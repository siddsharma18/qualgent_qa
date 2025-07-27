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
    """Main function to run the enhanced robust agent loop."""
    parser = argparse.ArgumentParser(description="Run the enhanced robust agent loop")
    parser.add_argument("--goal", "-g", type=str, required=True,
                       help="High-level goal to achieve (e.g., 'Turn off Wi-Fi and enable Bluetooth')")
    parser.add_argument("--config", "-c", type=str, default="default",
                       choices=["default", "high_performance", "high_reliability"],
                       help="Configuration preset to use")
    parser.add_argument("--max-iterations", "-i", type=int, default=10,
                       help="Maximum iterations for the loop")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--enhanced", "-e", action="store_true",
                       help="Use enhanced anti-flaky execution mode")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = QALogger(log_level=log_level, enable_console=True)
    
    logger.info("Main", "Starting enhanced robust agent loop", {
        "goal": args.goal,
        "config": args.config,
        "max_iterations": args.max_iterations,
        "enhanced_mode": args.enhanced
    })
    
    try:
        # Create mock environment (replace with real AndroidEnv in production)
        mock_env = MockEnvironment()
        
        # Create enhanced agent loop
        if args.enhanced:
            agent_loop = EnhancedRobustAgentLoop(mock_env, args.config)
            result = agent_loop.execute_goal_with_stability_checks(args.goal, args.max_iterations)
        else:
            agent_loop = RobustAgentLoop(mock_env, args.config)
            result = agent_loop.execute_goal(args.goal, args.max_iterations)
        
        # Print results
        print("\n" + "="*60)
        print("🎯 ENHANCED GOAL EXECUTION RESULTS")
        print("="*60)
        print(f"Goal: {result['goal']}")
        print(f"Status: {result['status'].upper()}")
        print(f"Success Rate: {result['success_rate']:.1%}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Iterations: {result['iterations']}")
        
        if result.get('stability_score'):
            print(f"Stability Score: {result['stability_score']:.2f}")
        
        if result.get('enhanced_execution'):
            print("🔧 Enhanced anti-flaky execution mode used")
        
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
        
        # Final result
        if result['status'] == 'success':
            print(f"\n🎉 SUCCESS! Goal achieved with enhanced reliability.")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS. Goal partially achieved.")
        
        return 0 if result['status'] == 'success' else 1
        
    except Exception as e:
        logger.error("Main", "Error in enhanced robust agent loop", {"error": str(e)})
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
