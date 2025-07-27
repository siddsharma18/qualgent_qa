#!/usr/bin/env python3
"""
Android in the Wild Dataset Integration
Implementation of the bonus task to integrate android_in_the_wild dataset
with our multi-agent QA system for enhanced training and evaluation.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent
from utils.logger import QALogger
from test_full_integration import RobustAgentLoop, MockEnvironment
from run_robust_loop import EnhancedRobustAgentLoop

@dataclass
class VideoAnalysisResult:
    """Result of analyzing a video from android_in_the_wild dataset."""
    video_id: str
    video_path: str
    generated_task_prompt: str
    extracted_ui_flow: List[Dict[str, Any]]
    agent_reproduction_result: Dict[str, Any]
    comparison_metrics: Dict[str, float]
    accuracy_score: float
    robustness_score: float
    generalization_score: float
    analysis_timestamp: str

@dataclass
class UIStep:
    """Represents a single UI interaction step."""
    timestamp: float
    action_type: str  # 'touch', 'swipe', 'type', 'wait'
    coordinates: Optional[Tuple[int, int]]
    element_description: str
    screen_state: Dict[str, Any]
    confidence: float

class AndroidInTheWildAnalyzer:
    """
    Analyzer for android_in_the_wild dataset videos.
    Extracts UI flows and generates task prompts for multi-agent reproduction.
    """
    
    def __init__(self, logger: Optional[QALogger] = None):
        self.logger = logger or QALogger()
        self.dataset_path = None
        self.video_analyzer = VideoUIAnalyzer()
        self.prompt_generator = TaskPromptGenerator()
        
    def setup_dataset(self, dataset_path: str = None) -> bool:
        """Setup the android_in_the_wild dataset."""
        if dataset_path and os.path.exists(dataset_path):
            self.dataset_path = dataset_path
            self.logger.info("DatasetSetup", f"Using local dataset at {dataset_path}")
            return True
        
        # Try to download/setup dataset
        return self._setup_dataset_from_source()
    
    def _setup_dataset_from_source(self) -> bool:
        """Setup dataset from online source or create mock data."""
        self.logger.info("DatasetSetup", "Setting up android_in_the_wild dataset")
        
        # For this implementation, we'll create mock video data that simulates
        # real android_in_the_wild scenarios
        mock_dataset_path = self._create_mock_dataset()
        if mock_dataset_path:
            self.dataset_path = mock_dataset_path
            return True
        
        return False
    
    def _create_mock_dataset(self) -> Optional[str]:
        """Create mock dataset for demonstration."""
        try:
            temp_dir = tempfile.mkdtemp(prefix="android_in_wild_")
            
            # Create mock video scenarios
            mock_scenarios = [
                {
                    "video_id": "settings_wifi_toggle_001",
                    "description": "User opens settings and toggles Wi-Fi on/off",
                    "ui_flow": [
                        {"action": "touch", "element": "settings_icon", "description": "Open Settings app"},
                        {"action": "touch", "element": "wifi_option", "description": "Navigate to Wi-Fi settings"},
                        {"action": "touch", "element": "wifi_toggle", "description": "Toggle Wi-Fi switch"},
                        {"action": "wait", "duration": 2.0, "description": "Wait for Wi-Fi state change"}
                    ],
                    "expected_outcome": "Wi-Fi state successfully changed"
                },
                {
                    "video_id": "bluetooth_pairing_002",
                    "description": "User enables Bluetooth and searches for devices",
                    "ui_flow": [
                        {"action": "touch", "element": "settings_icon", "description": "Open Settings app"},
                        {"action": "touch", "element": "bluetooth_option", "description": "Navigate to Bluetooth settings"},
                        {"action": "touch", "element": "bluetooth_toggle", "description": "Enable Bluetooth"},
                        {"action": "touch", "element": "scan_button", "description": "Start device scan"}
                    ],
                    "expected_outcome": "Bluetooth enabled and scanning for devices"
                },
                {
                    "video_id": "brightness_adjustment_003",
                    "description": "User adjusts screen brightness through quick settings",
                    "ui_flow": [
                        {"action": "swipe", "direction": "down", "description": "Pull down notification panel"},
                        {"action": "swipe", "direction": "down", "description": "Expand quick settings"},
                        {"action": "drag", "element": "brightness_slider", "description": "Adjust brightness slider"},
                        {"action": "touch", "element": "back", "description": "Close settings panel"}
                    ],
                    "expected_outcome": "Screen brightness successfully adjusted"
                },
                {
                    "video_id": "app_installation_004",
                    "description": "User installs an app from Play Store",
                    "ui_flow": [
                        {"action": "touch", "element": "play_store_icon", "description": "Open Play Store"},
                        {"action": "touch", "element": "search_bar", "description": "Open search"},
                        {"action": "type", "text": "calculator", "description": "Search for calculator app"},
                        {"action": "touch", "element": "first_result", "description": "Select first search result"},
                        {"action": "touch", "element": "install_button", "description": "Install the app"}
                    ],
                    "expected_outcome": "App successfully installed"
                },
                {
                    "video_id": "notification_management_005",
                    "description": "User manages notification settings",
                    "ui_flow": [
                        {"action": "swipe", "direction": "down", "description": "Pull down notification panel"},
                        {"action": "touch", "element": "notification_settings", "description": "Open notification settings"},
                        {"action": "touch", "element": "app_notifications", "description": "Select app notifications"},
                        {"action": "touch", "element": "toggle_notifications", "description": "Toggle app notifications"}
                    ],
                    "expected_outcome": "Notification settings successfully modified"
                }
            ]
            
            # Save mock scenarios
            scenarios_file = os.path.join(temp_dir, "scenarios.json")
            with open(scenarios_file, 'w') as f:
                json.dump(mock_scenarios, f, indent=2)
            
            # Create mock video files (placeholders)
            videos_dir = os.path.join(temp_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)
            
            for scenario in mock_scenarios:
                video_file = os.path.join(videos_dir, f"{scenario['video_id']}.mp4")
                # Create empty video file as placeholder
                open(video_file, 'a').close()
            
            self.logger.info("MockDataset", f"Created mock dataset at {temp_dir}")
            return temp_dir
            
        except Exception as e:
            self.logger.error("MockDataset", f"Failed to create mock dataset: {e}")
            return None
    
    def select_diverse_videos(self, count: int = 5) -> List[str]:
        """Select diverse videos from the dataset for analysis."""
        if not self.dataset_path:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
        
        scenarios_file = os.path.join(self.dataset_path, "scenarios.json")
        if not os.path.exists(scenarios_file):
            raise FileNotFoundError("Scenarios file not found in dataset")
        
        with open(scenarios_file, 'r') as f:
            scenarios = json.load(f)
        
        # Select diverse scenarios (different types of interactions)
        selected = scenarios[:min(count, len(scenarios))]
        return [s["video_id"] for s in selected]
    
    def analyze_video(self, video_id: str) -> VideoAnalysisResult:
        """Analyze a single video and extract UI flow."""
        self.logger.info("VideoAnalysis", f"Analyzing video: {video_id}")
        
        # Load scenario data
        scenarios_file = os.path.join(self.dataset_path, "scenarios.json")
        with open(scenarios_file, 'r') as f:
            scenarios = json.load(f)
        
        scenario = next((s for s in scenarios if s["video_id"] == video_id), None)
        if not scenario:
            raise ValueError(f"Video {video_id} not found in dataset")
        
        # Generate task prompt
        task_prompt = self.prompt_generator.generate_task_prompt(scenario)
        
        # Extract UI flow
        ui_flow = self._extract_ui_flow_from_scenario(scenario)
        
        # Reproduce with multi-agent system
        reproduction_result = self._reproduce_with_agents(task_prompt, ui_flow)
        
        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(ui_flow, reproduction_result)
        
        # Calculate scores
        accuracy_score = comparison_metrics.get('accuracy', 0.0)
        robustness_score = comparison_metrics.get('robustness', 0.0)
        generalization_score = comparison_metrics.get('generalization', 0.0)
        
        return VideoAnalysisResult(
            video_id=video_id,
            video_path=os.path.join(self.dataset_path, "videos", f"{video_id}.mp4"),
            generated_task_prompt=task_prompt,
            extracted_ui_flow=ui_flow,
            agent_reproduction_result=reproduction_result,
            comparison_metrics=comparison_metrics,
            accuracy_score=accuracy_score,
            robustness_score=robustness_score,
            generalization_score=generalization_score,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _extract_ui_flow_from_scenario(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract UI flow from scenario data."""
        ui_flow = []
        
        for i, step in enumerate(scenario["ui_flow"]):
            ui_step = {
                "step_id": i + 1,
                "timestamp": i * 2.0,  # Mock timestamps
                "action_type": step["action"],
                "element_description": step.get("description", ""),
                "coordinates": None,  # Would be extracted from video in real implementation
                "confidence": 0.9,  # Mock confidence
                "screen_state": {
                    "activity": "settings" if "settings" in step.get("description", "").lower() else "unknown",
                    "elements_visible": self._generate_mock_ui_elements(step)
                }
            }
            
            if step["action"] == "touch":
                ui_step["coordinates"] = (540, 960)  # Mock coordinates
            elif step["action"] == "swipe":
                ui_step["swipe_direction"] = step.get("direction", "down")
            elif step["action"] == "type":
                ui_step["text_input"] = step.get("text", "")
            
            ui_flow.append(ui_step)
        
        return ui_flow
    
    def _generate_mock_ui_elements(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mock UI elements based on step description."""
        elements = [
            {"id": "status_bar", "type": "status_bar", "bounds": [0, 0, 1080, 100]},
            {"id": "navigation_bar", "type": "navigation", "bounds": [0, 1820, 1080, 1920]}
        ]
        
        description = step.get("description", "").lower()
        
        if "settings" in description:
            elements.extend([
                {"id": "settings_title", "type": "text", "text": "Settings", "bounds": [50, 150, 200, 200]},
                {"id": "wifi_option", "type": "list_item", "text": "Wi-Fi", "bounds": [50, 300, 1030, 400]},
                {"id": "bluetooth_option", "type": "list_item", "text": "Bluetooth", "bounds": [50, 400, 1030, 500]}
            ])
        
        if "wifi" in description:
            elements.append({
                "id": "wifi_toggle", "type": "switch", "checked": True, "bounds": [900, 350, 980, 400]
            })
        
        if "bluetooth" in description:
            elements.append({
                "id": "bluetooth_toggle", "type": "switch", "checked": False, "bounds": [900, 450, 980, 500]
            })
        
        return elements
    
    def _reproduce_with_agents(self, task_prompt: str, ui_flow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reproduce the UI flow using our multi-agent system."""
        self.logger.info("AgentReproduction", f"Reproducing task: {task_prompt}")
        
        try:
            # Create enhanced agent loop
            mock_env = MockEnvironment()
            agent_loop = EnhancedRobustAgentLoop(mock_env, "default")
            
            # Execute the task
            result = agent_loop.execute_goal_with_stability_checks(task_prompt, max_iterations=15)
            
            # Add reproduction-specific metrics
            result["reproduction_metrics"] = {
                "steps_attempted": len(result.get("completed_subgoals", [])) + len(result.get("failed_subgoals", [])),
                "steps_completed": len(result.get("completed_subgoals", [])),
                "execution_fidelity": self._calculate_execution_fidelity(ui_flow, result),
                "timing_accuracy": self._calculate_timing_accuracy(ui_flow, result)
            }
            
            return result
            
        except Exception as e:
            self.logger.error("AgentReproduction", f"Failed to reproduce task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "reproduction_metrics": {
                    "steps_attempted": 0,
                    "steps_completed": 0,
                    "execution_fidelity": 0.0,
                    "timing_accuracy": 0.0
                }
            }
    
    def _calculate_execution_fidelity(self, original_flow: List[Dict[str, Any]], 
                                    agent_result: Dict[str, Any]) -> float:
        """Calculate how closely the agent execution matches the original flow."""
        if not original_flow:
            return 1.0
        
        completed_subgoals = agent_result.get("completed_subgoals", [])
        original_steps = len(original_flow)
        completed_steps = len(completed_subgoals)
        
        # Simple fidelity calculation based on completion ratio
        basic_fidelity = min(completed_steps / original_steps, 1.0) if original_steps > 0 else 0.0
        
        # Bonus for successful completion
        if agent_result.get("status") == "success":
            basic_fidelity *= 1.2
        
        return min(basic_fidelity, 1.0)
    
    def _calculate_timing_accuracy(self, original_flow: List[Dict[str, Any]], 
                                 agent_result: Dict[str, Any]) -> float:
        """Calculate timing accuracy between original and reproduced flow."""
        original_duration = max([step.get("timestamp", 0) for step in original_flow], default=0)
        agent_duration = agent_result.get("execution_time", 0)
        
        if original_duration == 0 or agent_duration == 0:
            return 0.5  # Neutral score if timing data unavailable
        
        # Calculate timing similarity (closer durations get higher scores)
        timing_ratio = min(original_duration, agent_duration) / max(original_duration, agent_duration)
        return timing_ratio
    
    def _calculate_comparison_metrics(self, ui_flow: List[Dict[str, Any]], 
                                    reproduction_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive comparison metrics."""
        metrics = {}
        
        # Accuracy: How well did the agent reproduce the exact steps?
        reproduction_metrics = reproduction_result.get("reproduction_metrics", {})
        steps_attempted = reproduction_metrics.get("steps_attempted", 0)
        steps_completed = reproduction_metrics.get("steps_completed", 0)
        
        if len(ui_flow) > 0:
            metrics["accuracy"] = min(steps_completed / len(ui_flow), 1.0)
        else:
            metrics["accuracy"] = 1.0
        
        # Robustness: How stable was the execution?
        stability_score = reproduction_result.get("stability_score", 0.5)
        success_rate = reproduction_result.get("success_rate", 0.0)
        metrics["robustness"] = (stability_score * 0.6) + (success_rate * 0.4)
        
        # Generalization: How well did the agent adapt to the task?
        execution_fidelity = reproduction_metrics.get("execution_fidelity", 0.0)
        timing_accuracy = reproduction_metrics.get("timing_accuracy", 0.0)
        metrics["generalization"] = (execution_fidelity * 0.7) + (timing_accuracy * 0.3)
        
        # Overall score
        metrics["overall"] = (metrics["accuracy"] * 0.4) + (metrics["robustness"] * 0.3) + (metrics["generalization"] * 0.3)
        
        return metrics


class VideoUIAnalyzer:
    """Analyzes video frames to extract UI interactions."""
    
    def extract_ui_flow(self, video_path: str) -> List[UIStep]:
        """Extract UI interaction flow from video."""
        # In a real implementation, this would use computer vision
        # to analyze video frames and detect UI interactions
        return []


class TaskPromptGenerator:
    """Generates natural language task prompts from UI flows."""
    
    def generate_task_prompt(self, scenario: Dict[str, Any]) -> str:
        """Generate a natural language task prompt from scenario data."""
        description = scenario.get("description", "")
        ui_flow = scenario.get("ui_flow", [])
        
        if "wifi" in description.lower():
            return "Turn Wi-Fi on or off using the Settings app"
        elif "bluetooth" in description.lower():
            return "Enable Bluetooth and search for nearby devices"
        elif "brightness" in description.lower():
            return "Adjust screen brightness using quick settings"
        elif "app" in description.lower() and "install" in description.lower():
            return "Install a calculator app from the Play Store"
        elif "notification" in description.lower():
            return "Manage notification settings for an app"
        else:
            # Generate generic prompt based on UI flow
            actions = [step.get("description", "") for step in ui_flow]
            return f"Complete the following sequence: {' -> '.join(actions)}"


class AndroidInTheWildEvaluator:
    """Evaluator for android_in_the_wild integration results."""
    
    def __init__(self, logger: Optional[QALogger] = None):
        self.logger = logger or QALogger()
    
    def evaluate_integration(self, analysis_results: List[VideoAnalysisResult]) -> Dict[str, Any]:
        """Evaluate the overall integration performance."""
        if not analysis_results:
            return {"error": "No analysis results provided"}
        
        # Calculate aggregate metrics
        accuracy_scores = [r.accuracy_score for r in analysis_results]
        robustness_scores = [r.robustness_score for r in analysis_results]
        generalization_scores = [r.generalization_score for r in analysis_results]
        
        aggregate_metrics = {
            "total_videos_analyzed": len(analysis_results),
            "average_accuracy": sum(accuracy_scores) / len(accuracy_scores),
            "average_robustness": sum(robustness_scores) / len(robustness_scores),
            "average_generalization": sum(generalization_scores) / len(generalization_scores),
            "overall_performance": (
                sum(accuracy_scores) + sum(robustness_scores) + sum(generalization_scores)
            ) / (3 * len(analysis_results))
        }
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if aggregate_metrics["average_accuracy"] > 0.8:
            strengths.append("High accuracy in task reproduction")
        elif aggregate_metrics["average_accuracy"] < 0.6:
            weaknesses.append("Low accuracy in task reproduction")
        
        if aggregate_metrics["average_robustness"] > 0.8:
            strengths.append("Robust execution across different scenarios")
        elif aggregate_metrics["average_robustness"] < 0.6:
            weaknesses.append("Inconsistent robustness across scenarios")
        
        if aggregate_metrics["average_generalization"] > 0.8:
            strengths.append("Good generalization to diverse UI patterns")
        elif aggregate_metrics["average_generalization"] < 0.6:
            weaknesses.append("Poor generalization to diverse UI patterns")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(aggregate_metrics, analysis_results)
        
        return {
            "aggregate_metrics": aggregate_metrics,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "detailed_results": [
                {
                    "video_id": r.video_id,
                    "task_prompt": r.generated_task_prompt,
                    "accuracy": r.accuracy_score,
                    "robustness": r.robustness_score,
                    "generalization": r.generalization_score,
                    "agent_status": r.agent_reproduction_result.get("status", "unknown")
                }
                for r in analysis_results
            ]
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float], 
                                results: List[VideoAnalysisResult]) -> List[str]:
        """Generate recommendations for improving the multi-agent system."""
        recommendations = []
        
        if metrics["average_accuracy"] < 0.7:
            recommendations.append("Improve element detection accuracy in the Executor Agent")
            recommendations.append("Enhance UI parsing strategies for complex layouts")
        
        if metrics["average_robustness"] < 0.7:
            recommendations.append("Implement more robust retry mechanisms")
            recommendations.append("Add better error recovery strategies")
        
        if metrics["average_generalization"] < 0.7:
            recommendations.append("Train agents on more diverse UI patterns")
            recommendations.append("Improve semantic understanding in the Planner Agent")
        
        # Analyze failure patterns
        failed_results = [r for r in results if r.agent_reproduction_result.get("status") != "success"]
        if len(failed_results) > len(results) * 0.3:  # More than 30% failures
            recommendations.append("Investigate common failure patterns and add specific handling")
        
        return recommendations


def _safe_serialize(obj) -> Any:
    """Safely serialize an object to JSON-compatible format."""
    try:
        # Try to JSON encode to test if it's serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # If it can't be serialized, convert to string or dict
        if hasattr(obj, '__dict__'):
            return {k: _safe_serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [_safe_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _safe_serialize(v) for k, v in obj.items()}
        else:
            return str(obj)


def _serialize_agent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert agent result to JSON-serializable format."""
    if not result:
        return {}
    
    serialized = {}
    for key, value in result.items():
        try:
            if key == "planning_result":
                # Convert PlanningResult to dict
                if hasattr(value, '__dict__'):
                    planning_dict = {}
                    for attr_name, attr_value in value.__dict__.items():
                        if attr_name == "subgoals":
                            # Convert subgoals to simple list
                            planning_dict[attr_name] = [
                                {
                                    "name": sg.name,
                                    "description": sg.description,
                                    "priority": sg.priority,
                                    "confidence": sg.confidence
                                } if hasattr(sg, 'name') else str(sg)
                                for sg in attr_value
                            ]
                        elif attr_name == "status":
                            # Convert enum to string
                            planning_dict[attr_name] = str(attr_value)
                        else:
                            planning_dict[attr_name] = _safe_serialize(attr_value)
                    serialized[key] = planning_dict
                else:
                    serialized[key] = str(value)
            else:
                serialized[key] = _safe_serialize(value)
        except Exception as e:
            serialized[key] = f"<serialization_error: {str(e)}>"
    
    return serialized


def main():
    """Main function to run Android in the Wild integration."""
    parser = argparse.ArgumentParser(description="Android in the Wild Integration")
    parser.add_argument("--num-videos", type=int, default=5, help="Number of videos to analyze")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    print("Android in the Wild Integration - QualGent QA System")
    print("=" * 60)
    print()
    
    try:
        # Create analyzer
        analyzer = AndroidInTheWildAnalyzer()
        
        # Run analysis
        print(f"Analyzing {args.num_videos} Android in the Wild scenarios...")
        print()
        
        results = analyzer.analyze_scenarios(args.num_videos, verbose=args.verbose)
        
        # Print results
        print("Analysis Results:")
        print("-" * 30)
        print(f"   Videos Analyzed: {len(results)}")
        print(f"   Average Accuracy: {analyzer.calculate_average_accuracy(results):.2f}")
        print(f"   Average Robustness: {analyzer.calculate_average_robustness(results):.2f}")
        print(f"   Average Generalization: {analyzer.calculate_average_generalization(results):.2f}")
        print(f"   Overall Performance: {analyzer.calculate_overall_performance(results):.2f}")
        print()
        
        # Print detailed results
        if args.verbose:
            print("Detailed Results:")
            print("-" * 20)
            for i, result in enumerate(results, 1):
                print(f"   Scenario {i}: {result['scenario_name']}")
                print(f"     Accuracy: {result['accuracy']:.2f}")
                print(f"     Robustness: {result['robustness']:.2f}")
                print(f"     Generalization: {result['generalization']:.2f}")
                print(f"     Performance: {result['performance']:.2f}")
                print()
        
        # Generate report
        report = analyzer.generate_report(results)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"android_in_the_wild_report_{timestamp}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {report_filename}")
        print()
        print("Android in the Wild integration completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
