#!/usr/bin/env python3
"""
Comprehensive Validation Script
Tests all components of the multi-agent QA system including flaky behavior fixes
and runs the android_in_the_wild integration bonus task.
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, Any, List
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import QALogger
from agents.supervisor_agent import SupervisorAgent
from run_robust_loop import EnhancedRobustAgentLoop
from test_full_integration import RobustAgentLoop, MockEnvironment
from android_in_the_wild_integration import AndroidInTheWildAnalyzer, AndroidInTheWildEvaluator

class ComprehensiveValidator:
    """Comprehensive validator for the entire multi-agent QA system."""
    
    def __init__(self, logger: QALogger):
        self.logger = logger
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "component_tests": {},
            "flaky_behavior_tests": {},
            "android_in_wild_integration": {},
            "overall_assessment": {}
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all components."""
        self.logger.info("Validator", "Starting comprehensive validation")
        
        # 1. Test individual components
        self.logger.info("Validator", "Testing individual components...")
        self.results["component_tests"] = self._test_individual_components()
        
        # 2. Test flaky behavior fixes
        self.logger.info("Validator", "Testing flaky behavior fixes...")
        self.results["flaky_behavior_tests"] = self._test_flaky_behavior_fixes()
        
        # 3. Run android_in_the_wild integration
        self.logger.info("Validator", "Running android_in_the_wild integration...")
        self.results["android_in_wild_integration"] = self._run_android_in_wild_integration()
        
        # 4. Generate overall assessment
        self.logger.info("Validator", "Generating overall assessment...")
        self.results["overall_assessment"] = self._generate_overall_assessment()
        
        return self.results
    
    def _test_individual_components(self) -> Dict[str, Any]:
        """Test individual agent components."""
        component_results = {}
        
        # Test Planner Agent
        try:
            from agents.planner_agent import PlannerAgent
            planner = PlannerAgent()
            
            # Test basic planning
            planning_result = planner.plan("Turn off Wi-Fi", {})
            
            component_results["planner_agent"] = {
                "status": "pass" if planning_result.status.value == "success" else "fail",
                "confidence": planning_result.confidence,
                "subgoals_generated": len(planning_result.subgoals),
                "strategies_used": planning_result.strategies_used,
                "stability_features": {
                    "risk_assessment": hasattr(planning_result, 'risk_assessment'),
                    "stability_score": hasattr(planning_result, 'stability_score')
                }
            }
            
        except Exception as e:
            component_results["planner_agent"] = {"status": "error", "error": str(e)}
        
        # Test Executor Agent
        try:
            from agents.executor_agent import ExecutorAgent
            mock_env = MockEnvironment()
            executor = ExecutorAgent(mock_env)
            
            # Test basic execution
            ui_tree = [{"id": "wifi_toggle", "text": "Wi-Fi", "clickable": True}]
            execution_result = executor.execute("Turn off Wi-Fi", ui_tree)
            
            component_results["executor_agent"] = {
                "status": "pass" if execution_result.status == "success" else "fail",
                "confidence": execution_result.confidence,
                "attempts": execution_result.attempts,
                "stability_features": {
                    "retry_strategy_used": hasattr(execution_result, 'retry_strategy_used'),
                    "stability_score": hasattr(execution_result, 'stability_score'),
                    "ui_changes_detected": hasattr(execution_result, 'ui_changes_detected')
                }
            }
            
        except Exception as e:
            component_results["executor_agent"] = {"status": "error", "error": str(e)}
        
        # Test Verifier Agent
        try:
            from agents.verifier_agent import VerifierAgent
            verifier = VerifierAgent()
            
            # Test basic verification
            prev_obs = {"ui_tree": [{"id": "wifi_toggle", "checked": True}]}
            curr_obs = {"ui_tree": [{"id": "wifi_toggle", "checked": False}]}
            
            verification_result = verifier.verify("Turn off Wi-Fi", prev_obs, curr_obs)
            
            component_results["verifier_agent"] = {
                "status": "pass" if verification_result.status.value == "pass" else "fail",
                "confidence": verification_result.confidence,
                "strategies_used": verification_result.strategies_used,
                "stability_features": {
                    "stability_score": hasattr(verification_result, 'stability_score'),
                    "false_positive_risk": hasattr(verification_result, 'false_positive_risk'),
                    "alternative_interpretations": hasattr(verification_result, 'alternative_interpretations')
                }
            }
            
        except Exception as e:
            component_results["verifier_agent"] = {"status": "error", "error": str(e)}
        
        # Test Supervisor Agent
        try:
            supervisor = SupervisorAgent()
            
            # Test basic supervision (use existing logs if available)
            if os.path.exists("qa_logs.json"):
                evaluation_result = supervisor.evaluate_logs("qa_logs.json")
                
                component_results["supervisor_agent"] = {
                    "status": "pass",
                    "total_goals": evaluation_result.total_goals,
                    "success_rate": evaluation_result.overall_success_rate,
                    "flaky_detection": {
                        "flaky_goals_detected": evaluation_result.flaky_goals,
                        "flaky_subgoals": len(evaluation_result.flaky_subgoals),
                        "enhanced_detection": hasattr(supervisor, '_detect_flaky_behavior')
                    }
                }
            else:
                component_results["supervisor_agent"] = {"status": "skip", "reason": "No logs available"}
                
        except Exception as e:
            component_results["supervisor_agent"] = {"status": "error", "error": str(e)}
        
        return component_results
    
    def _test_flaky_behavior_fixes(self) -> Dict[str, Any]:
        """Test the flaky behavior fixes."""
        flaky_results = {}
        
        # Test enhanced robust loop
        try:
            mock_env = MockEnvironment()
            enhanced_loop = EnhancedRobustAgentLoop(mock_env, "default")
            
            # Test multiple executions of the same goal to check for flakiness
            test_goals = [
                "Turn off Wi-Fi",
                "Enable Bluetooth",
                "Configure device settings"
            ]
            
            for goal in test_goals:
                execution_results = []
                
                # Run the same goal multiple times
                for i in range(3):
                    try:
                        result = enhanced_loop.execute_goal_with_stability_checks(goal, max_iterations=5)
                        execution_results.append({
                            "attempt": i + 1,
                            "status": result.get("status", "unknown"),
                            "execution_time": result.get("execution_time", 0),
                            "stability_score": result.get("stability_score", 0),
                            "enhanced_execution": result.get("enhanced_execution", False)
                        })
                    except Exception as e:
                        execution_results.append({
                            "attempt": i + 1,
                            "status": "error",
                            "error": str(e)
                        })
                
                # Analyze consistency
                statuses = [r.get("status") for r in execution_results]
                success_count = statuses.count("success")
                consistency_score = success_count / len(execution_results)
                
                flaky_results[goal] = {
                    "executions": execution_results,
                    "consistency_score": consistency_score,
                    "is_stable": consistency_score >= 0.8,  # 80% success rate for stability
                    "improvements_applied": any(r.get("enhanced_execution") for r in execution_results)
                }
                
        except Exception as e:
            flaky_results["error"] = str(e)
        
        # Test flaky behavior detection
        try:
            if hasattr(enhanced_loop, 'flaky_behavior_detector'):
                detector = enhanced_loop.flaky_behavior_detector
                
                # Create mock execution history with flaky pattern
                mock_history = [
                    {"success": True, "execution_time": 5.0},
                    {"success": False, "execution_time": 12.0},
                    {"success": True, "execution_time": 4.5},
                    {"success": False, "execution_time": 15.0},
                    {"success": True, "execution_time": 5.2}
                ]
                
                flaky_score = detector.calculate_goal_flakiness(mock_history)
                
                flaky_results["flaky_detection"] = {
                    "detector_available": True,
                    "test_flaky_score": flaky_score,
                    "detection_working": flaky_score > 0.4  # Should detect flakiness
                }
        except Exception as e:
            flaky_results["flaky_detection"] = {"error": str(e)}
        
        return flaky_results
    
    def _run_android_in_wild_integration(self) -> Dict[str, Any]:
        """Run the android_in_the_wild integration bonus task."""
        integration_results = {}
        
        try:
            # Initialize analyzer
            analyzer = AndroidInTheWildAnalyzer(self.logger)
            
            # Setup dataset
            if analyzer.setup_dataset():
                integration_results["dataset_setup"] = {"status": "success", "path": analyzer.dataset_path}
                
                # Select videos for analysis
                video_ids = analyzer.select_diverse_videos(5)
                integration_results["selected_videos"] = video_ids
                
                # Analyze videos
                analysis_results = []
                for video_id in video_ids:
                    try:
                        result = analyzer.analyze_video(video_id)
                        analysis_results.append(result)
                        
                        self.logger.info("AndroidInWild", f"Analyzed {video_id}: "
                                       f"Accuracy={result.accuracy_score:.2f}, "
                                       f"Robustness={result.robustness_score:.2f}, "
                                       f"Generalization={result.generalization_score:.2f}")
                    except Exception as e:
                        self.logger.error("AndroidInWild", f"Failed to analyze {video_id}: {e}")
                
                if analysis_results:
                    # Evaluate integration
                    evaluator = AndroidInTheWildEvaluator(self.logger)
                    evaluation = evaluator.evaluate_integration(analysis_results)
                    
                    integration_results["analysis_results"] = [
                        {
                            "video_id": r.video_id,
                            "task_prompt": r.generated_task_prompt,
                            "accuracy_score": r.accuracy_score,
                            "robustness_score": r.robustness_score,
                            "generalization_score": r.generalization_score,
                            "agent_status": r.agent_reproduction_result.get("status", "unknown")
                        }
                        for r in analysis_results
                    ]
                    
                    integration_results["evaluation"] = evaluation
                    integration_results["status"] = "success"
                else:
                    integration_results["status"] = "no_results"
                    
            else:
                integration_results["dataset_setup"] = {"status": "failed"}
                integration_results["status"] = "setup_failed"
                
        except Exception as e:
            integration_results["status"] = "error"
            integration_results["error"] = str(e)
            self.logger.error("AndroidInWild", f"Integration failed: {e}")
        
        return integration_results
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall assessment of the system."""
        assessment = {
            "component_health": {},
            "flaky_behavior_status": {},
            "android_in_wild_status": {},
            "recommendations": [],
            "overall_score": 0.0
        }
        
        # Assess component health
        component_tests = self.results.get("component_tests", {})
        total_components = len(component_tests)
        passing_components = sum(1 for test in component_tests.values() 
                               if test.get("status") == "pass")
        
        if total_components > 0:
            component_health_score = passing_components / total_components
            assessment["component_health"] = {
                "score": component_health_score,
                "status": "healthy" if component_health_score >= 0.8 else "issues_detected",
                "passing_components": passing_components,
                "total_components": total_components
            }
        
        # Assess flaky behavior fixes
        flaky_tests = self.results.get("flaky_behavior_tests", {})
        stable_goals = sum(1 for goal_data in flaky_tests.values() 
                          if isinstance(goal_data, dict) and goal_data.get("is_stable", False))
        total_goals = len([k for k in flaky_tests.keys() if k != "error" and k != "flaky_detection"])
        
        if total_goals > 0:
            stability_score = stable_goals / total_goals
            assessment["flaky_behavior_status"] = {
                "score": stability_score,
                "status": "stable" if stability_score >= 0.8 else "flaky_detected",
                "stable_goals": stable_goals,
                "total_goals": total_goals,
                "detection_working": flaky_tests.get("flaky_detection", {}).get("detection_working", False)
            }
        
        # Assess android_in_the_wild integration
        integration = self.results.get("android_in_wild_integration", {})
        if integration.get("status") == "success":
            evaluation = integration.get("evaluation", {})
            metrics = evaluation.get("aggregate_metrics", {})
            overall_performance = metrics.get("overall_performance", 0.0)
            
            assessment["android_in_wild_status"] = {
                "score": overall_performance,
                "status": "success" if overall_performance >= 0.7 else "needs_improvement",
                "videos_analyzed": metrics.get("total_videos_analyzed", 0),
                "average_accuracy": metrics.get("average_accuracy", 0.0),
                "average_robustness": metrics.get("average_robustness", 0.0),
                "average_generalization": metrics.get("average_generalization", 0.0)
            }
        else:
            assessment["android_in_wild_status"] = {
                "status": "failed",
                "error": integration.get("error", "Unknown error")
            }
        
        # Generate recommendations
        if assessment["component_health"].get("score", 0) < 0.8:
            assessment["recommendations"].append("Fix failing agent components")
        
        if assessment["flaky_behavior_status"].get("score", 0) < 0.8:
            assessment["recommendations"].append("Implement additional stability improvements")
        
        if assessment["android_in_wild_status"].get("score", 0) < 0.7:
            assessment["recommendations"].append("Improve real-world scenario handling")
        
        # Calculate overall score
        scores = [
            assessment["component_health"].get("score", 0) * 0.4,
            assessment["flaky_behavior_status"].get("score", 0) * 0.3,
            assessment["android_in_wild_status"].get("score", 0) * 0.3
        ]
        assessment["overall_score"] = sum(scores)
        
        return assessment
    
    def save_results(self, output_file: str) -> None:
        """Save validation results to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info("Validator", f"Results saved to {output_file}")


def main():
    """Main function for comprehensive validation."""
    parser = argparse.ArgumentParser(description="Comprehensive Multi-Agent QA System Validation")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = QALogger(log_level=log_level, enable_console=True)
    
    logger.info("Main", "Starting comprehensive validation")
    
    try:
        # Run comprehensive validation
        validator = ComprehensiveValidator(logger)
        results = validator.run_full_validation()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"comprehensive_validation_{timestamp}.json")
        validator.save_results(output_file)
        
        # Print summary
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE VALIDATION RESULTS")
        print("="*80)
        
        # Component health
        component_health = results["overall_assessment"]["component_health"]
        print(f"\n🤖 Component Health: {component_health.get('status', 'unknown').upper()}")
        print(f"  Score: {component_health.get('score', 0):.2f}")
        print(f"  Passing: {component_health.get('passing_components', 0)}/{component_health.get('total_components', 0)}")
        
        # Flaky behavior status
        flaky_status = results["overall_assessment"]["flaky_behavior_status"]
        print(f"\n🔄 Flaky Behavior Status: {flaky_status.get('status', 'unknown').upper()}")
        print(f"  Stability Score: {flaky_status.get('score', 0):.2f}")
        print(f"  Stable Goals: {flaky_status.get('stable_goals', 0)}/{flaky_status.get('total_goals', 0)}")
        
        # Android in the Wild integration
        android_status = results["overall_assessment"]["android_in_wild_status"]
        print(f"\n📱 Android in the Wild: {android_status.get('status', 'unknown').upper()}")
        if android_status.get("score") is not None:
            print(f"  Performance Score: {android_status.get('score', 0):.2f}")
            print(f"  Videos Analyzed: {android_status.get('videos_analyzed', 0)}")
            print(f"  Average Accuracy: {android_status.get('average_accuracy', 0):.2f}")
            print(f"  Average Robustness: {android_status.get('average_robustness', 0):.2f}")
            print(f"  Average Generalization: {android_status.get('average_generalization', 0):.2f}")
        
        # Overall assessment
        overall_score = results["overall_assessment"]["overall_score"]
        print(f"\n📊 Overall System Score: {overall_score:.2f}")
        
        if overall_score >= 0.8:
            print("🎉 EXCELLENT: System is performing well!")
        elif overall_score >= 0.6:
            print("✅ GOOD: System is functional with minor issues")
        else:
            print("⚠️  NEEDS IMPROVEMENT: System requires attention")
        
        # Recommendations
        recommendations = results["overall_assessment"]["recommendations"]
        if recommendations:
            print(f"\n🔧 Recommendations:")
            for rec in recommendations:
                print(f"  💡 {rec}")
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        return 0 if overall_score >= 0.6 else 1
        
    except Exception as e:
        logger.error("Main", f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
