#!/usr/bin/env python3
"""
Android World Task Runner
Integrates our Agent-S system with android_world tasks for comprehensive QA testing.
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Union, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_s_integration import AgentSGraphSearchAgent
from agents.supervisor_agent import SupervisorAgent
from utils.logger import QALogger
from config.qa_config import get_config

# Import android_world components with better error handling
ANDROID_WORLD_AVAILABLE = False
env_launcher = None
registry = None
task_eval = None

try:
    from android_world import env_launcher
    from android_world import registry
    from android_world.task_evals import task_eval
    ANDROID_WORLD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  android_world not available: {e}")
    print("Using mock environment for testing.")
except Exception as e:
    print(f"⚠️  Error importing android_world: {e}")
    print("Using mock environment for testing.")

class MockAndroidEnv:
    """Enhanced mock Android environment for testing when android_world is not available."""
    
    def __init__(self):
        self.current_screen = "home"
        self.wifi_enabled = True
        self.bluetooth_enabled = False
        self.settings_open = False
        self.step_count = 0
        self.max_steps = 50
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        self.current_screen = "home"
        self.wifi_enabled = True
        self.bluetooth_enabled = False
        self.settings_open = False
        self.step_count = 0
        return self._get_observation()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return observation."""
        self.step_count += 1
        action_type = action.get("action_type", "")
        element_id = action.get("element_id", "")
        
        # Simulate realistic UI interactions
        if action_type == "touch":
            if "settings" in element_id.lower():
                self.settings_open = True
                self.current_screen = "settings"
            elif "wifi" in element_id.lower() and self.settings_open:
                self.wifi_enabled = not self.wifi_enabled
            elif "bluetooth" in element_id.lower() and self.settings_open:
                self.bluetooth_enabled = not self.bluetooth_enabled
            elif "back" in element_id.lower() and self.settings_open:
                self.settings_open = False
                self.current_screen = "home"
        
        # Add some randomness to simulate real environment
        if self.step_count > self.max_steps:
            return {"observation": self._get_observation(), "done": True, "reward": 0.0}
        
        return {"observation": self._get_observation(), "done": False, "reward": 1.0}
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        ui_tree = []
        
        if self.current_screen == "home":
            ui_tree = [
                {"text": "Settings", "type": "button", "id": "settings_button", "clickable": True},
                {"text": "Wi-Fi", "type": "toggle", "id": "wifi_toggle", "state": "on" if self.wifi_enabled else "off", "clickable": True},
                {"text": "Bluetooth", "type": "toggle", "id": "bluetooth_toggle", "state": "on" if self.bluetooth_enabled else "off", "clickable": True}
            ]
        elif self.current_screen == "settings":
            ui_tree = [
                {"text": "Wi-Fi", "type": "toggle", "id": "wifi_toggle", "state": "on" if self.wifi_enabled else "off", "clickable": True},
                {"text": "Bluetooth", "type": "toggle", "id": "bluetooth_toggle", "state": "on" if self.bluetooth_enabled else "off", "clickable": True},
                {"text": "Back", "type": "button", "id": "back_button", "clickable": True}
            ]
        
        return {
            "ui_tree": ui_tree,
            "screenshot": b"mock_screenshot_data",
            "accessibility_tree": f"Screen: {self.current_screen}, WiFi: {self.wifi_enabled}, Bluetooth: {self.bluetooth_enabled}"
        }

class AndroidWorldTaskRunner:
    """
    Enhanced task runner that integrates our Agent-S system with android_world tasks.
    Includes robust error handling, fallback mechanisms, and comprehensive logging.
    """
    
    def __init__(self, 
                 console_port: int = 5554,
                 perform_emulator_setup: bool = False,
                 adb_path: Optional[str] = None,
                 config_name: str = "default"):
        """
        Initialize the Android World task runner.
        
        Args:
            console_port: Emulator console port
            perform_emulator_setup: Whether to perform emulator setup
            adb_path: Path to ADB executable
            config_name: Configuration preset to use
        """
        self.console_port = console_port
        self.perform_emulator_setup = perform_emulator_setup
        self.adb_path = adb_path
        self.config_name = config_name
        
        # Initialize components
        self.config = get_config(config_name)
        self.logger = QALogger(log_level=self.config.logger.log_level, enable_console=True)
        
        # Initialize environment
        self.env = None
        self.task_registry = None
        self.agent = None
        self.initialization_successful = False
        
        # Initialize environment with fallback
        self._initialize_environment()
    
    def _initialize_environment(self) -> None:
        """Initialize environment with robust fallback mechanisms."""
        if ANDROID_WORLD_AVAILABLE:
            self._initialize_android_world()
        else:
            self._initialize_mock_environment()
    
    def _initialize_android_world(self) -> None:
        """Initialize android_world environment with enhanced error handling."""
        try:
            self.logger.info("AndroidWorld", "Initializing Android World environment")
            
            # Validate android_world components
            if not all([env_launcher, registry, task_eval]):
                raise ImportError("android_world components not properly imported")
            
            # Load and setup environment
            self.env = env_launcher.load_and_setup_env(
                console_port=self.console_port,
                emulator_setup=self.perform_emulator_setup,
                adb_path=self.adb_path
            )
            
            # Initialize task registry
            self.task_registry = registry.TaskRegistry()
            aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
            
            if not aw_registry:
                raise RuntimeError("Failed to get Android World task registry")
            
            # Initialize Agent-S agent
            engine_params = {
                "engine_type": "mock",  # Replace with actual LLM engine
                "model": "mock-llm",
                "api_key": "mock_key"
            }
            
            self.agent = AgentSGraphSearchAgent(
                env=self.env,
                engine_params=engine_params,
                platform="android",
                action_space="android_env",
                observation_type="mixed"
            )
            
            self.initialization_successful = True
            self.logger.info("AndroidWorld", "Environment initialized successfully")
            
        except Exception as e:
            self.logger.error("AndroidWorld", f"Failed to initialize Android World: {e}")
            self.logger.info("AndroidWorld", "Falling back to mock environment")
            self._initialize_mock_environment()
    
    def _initialize_mock_environment(self) -> None:
        """Initialize mock environment for testing."""
        self.logger.info("MockEnv", "Initializing mock environment")
        
        try:
            self.env = MockAndroidEnv()
            
            # Initialize Agent-S agent with mock environment
            engine_params = {
                "engine_type": "mock",
                "model": "mock-llm",
                "api_key": "mock_key"
            }
            
            self.agent = AgentSGraphSearchAgent(
                env=self.env,
                engine_params=engine_params,
                platform="android",
                action_space="android_env",
                observation_type="mixed"
            )
            
            self.initialization_successful = True
            self.logger.info("MockEnv", "Mock environment initialized successfully")
            
        except Exception as e:
            self.logger.error("MockEnv", f"Failed to initialize mock environment: {e}")
            self.initialization_successful = False
    
    def run_task(self, task_name: str, max_steps: int = 50) -> Dict[str, Any]:
        """
        Run a specific android_world task with enhanced error handling.
        
        Args:
            task_name: Name of the task to run
            max_steps: Maximum number of steps
            
        Returns:
            Task execution results
        """
        if not self.initialization_successful:
            return self._create_failure_result(f"Environment not properly initialized", task_name)
        
        self.logger.info("TaskRunner", f"Starting task: {task_name}")
        
        try:
            # Reset environment and agent
            obs = self.env.reset()
            self.agent.reset()
            
            # Initialize task if using android_world
            task = None
            goal = f"Execute task: {task_name}"
            
            if ANDROID_WORLD_AVAILABLE and self.task_registry:
                try:
                    aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
                    if task_name in aw_registry:
                        task_class = aw_registry[task_name]
                        params = task_class.generate_random_params()
                        task = task_class(params)
                        task.initialize_task(self.env)
                        goal = task.goal
                    else:
                        self.logger.warning("TaskRunner", f"Task {task_name} not found in registry")
                except Exception as e:
                    self.logger.warning("TaskRunner", f"Failed to initialize task {task_name}: {e}")
            
            # Execute task using Agent-S pattern
            trajectory = f"Task: {task_name}\nGoal: {goal}\n"
            step_results = []
            
            for step in range(max_steps):
                self.logger.info("TaskRunner", f"Step {step + 1}/{max_steps}")
                
                try:
                    # Get next action from agent
                    info, actions = self.agent.predict(goal, obs)
                    
                    # Record step information
                    step_info = {
                        "step": step + 1,
                        "subtask": info.get("subtask"),
                        "subtask_status": info.get("subtask_status"),
                        "actions": actions,
                        "info": info
                    }
                    step_results.append(step_info)
                    
                    # Execute actions
                    if actions and "DONE" not in actions and "FAIL" not in actions:
                        for action in actions:
                            if hasattr(self.env, 'step'):
                                obs = self.env.step({"action_type": "touch", "element_id": action})
                            else:
                                # Mock environment
                                obs = self.env.step({"action_type": "touch", "element_id": "mock_element"})
                    
                    # Check completion
                    if "DONE" in actions:
                        self.logger.info("TaskRunner", "Task completed successfully")
                        break
                    elif "FAIL" in actions:
                        self.logger.warning("TaskRunner", "Task failed")
                        break
                    
                    # Update trajectory
                    trajectory += f"\nStep {step + 1}: {info.get('subtask', 'Unknown')} - {info.get('subtask_status', 'Unknown')}"
                    
                except Exception as e:
                    self.logger.error("TaskRunner", f"Error in step {step + 1}: {e}")
                    break
            
            # Update narrative memory
            try:
                self.agent.update_narrative_memory(trajectory)
            except Exception as e:
                self.logger.warning("TaskRunner", f"Failed to update narrative memory: {e}")
            
            # Evaluate task success if using android_world
            success_score = 0.0
            if ANDROID_WORLD_AVAILABLE and task:
                try:
                    success_score = task.is_successful(self.env)
                    task.tear_down(self.env)
                except Exception as e:
                    self.logger.warning("TaskRunner", f"Failed to evaluate task success: {e}")
            
            # Compile results
            results = {
                "task_name": task_name,
                "goal": goal,
                "success_score": success_score,
                "total_steps": len(step_results),
                "completed": "DONE" in actions if actions else False,
                "failed": "FAIL" in actions if actions else False,
                "step_results": step_results,
                "trajectory": trajectory,
                "message_count": len(self.agent.get_message_history()),
                "visual_trace_count": len(self.agent.get_visual_traces()),
                "environment_type": "android" if ANDROID_WORLD_AVAILABLE else "mock"
            }
            
            self.logger.info("TaskRunner", f"Task {task_name} completed with score: {success_score}")
            return results
            
        except Exception as e:
            self.logger.error("TaskRunner", f"Failed to run task {task_name}: {e}")
            return self._create_failure_result(str(e), task_name)
    
    def _create_failure_result(self, reason: str, task_name: str) -> Dict[str, Any]:
        """Create a standardized failure result."""
        return {
            "task_name": task_name,
            "goal": f"Execute task: {task_name}",
            "success_score": 0.0,
            "total_steps": 0,
            "completed": False,
            "failed": True,
            "error": reason,
            "step_results": [],
            "trajectory": f"Task: {task_name}\nError: {reason}",
            "message_count": 0,
            "visual_trace_count": 0,
            "environment_type": "android" if ANDROID_WORLD_AVAILABLE else "mock"
        }
    
    def run_task_suite(self, task_names: List[str], max_steps: int = 50) -> Dict[str, Any]:
        """
        Run a suite of android_world tasks with enhanced error handling.
        
        Args:
            task_names: List of task names to run
            max_steps: Maximum number of steps per task
            
        Returns:
            Suite execution results
        """
        self.logger.info("TaskRunner", f"Starting task suite with {len(task_names)} tasks")
        
        suite_results = {
            "suite_name": "android_world_qa_suite",
            "total_tasks": len(task_names),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_success_score": 0.0,
            "task_results": [],
            "suite_statistics": {},
            "environment_type": "android" if ANDROID_WORLD_AVAILABLE else "mock"
        }
        
        total_success_score = 0.0
        
        for i, task_name in enumerate(task_names):
            self.logger.info("TaskRunner", f"Running task {i+1}/{len(task_names)}: {task_name}")
            
            try:
                task_result = self.run_task(task_name, max_steps)
                suite_results["task_results"].append(task_result)
                
                if task_result["completed"]:
                    suite_results["completed_tasks"] += 1
                elif task_result["failed"]:
                    suite_results["failed_tasks"] += 1
                
                total_success_score += task_result["success_score"]
                
            except Exception as e:
                self.logger.error("TaskRunner", f"Failed to run task {task_name}: {e}")
                suite_results["failed_tasks"] += 1
                suite_results["task_results"].append({
                    "task_name": task_name,
                    "error": str(e),
                    "completed": False,
                    "failed": True,
                    "success_score": 0.0
                })
        
        # Calculate suite statistics
        if suite_results["task_results"]:
            suite_results["average_success_score"] = total_success_score / len(suite_results["task_results"])
        
        # Generate evaluation report
        try:
            supervisor = SupervisorAgent(logger=self.logger)
            evaluation_report = supervisor.evaluate_logs(
                log_path="qa_logs.json",
                config={"name": "android_world_suite", "settings": self.config.__dict__},
                test_context={
                    "test_type": "android_world_suite",
                    "environment": "android" if ANDROID_WORLD_AVAILABLE else "mock",
                    "suite_results": suite_results
                }
            )
            suite_results["evaluation_report"] = evaluation_report.__dict__
        except Exception as e:
            self.logger.error("TaskRunner", f"Failed to generate evaluation report: {e}")
        
        self.logger.info("TaskRunner", f"Task suite completed. Success rate: {suite_results['completed_tasks']}/{suite_results['total_tasks']}")
        return suite_results
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available android_world tasks."""
        if ANDROID_WORLD_AVAILABLE and self.task_registry:
            try:
                aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
                return list(aw_registry.keys())
            except Exception as e:
                self.logger.warning("TaskRunner", f"Failed to get available tasks: {e}")
        
        # Return mock task names
        return [
            "SystemWifiTurnOff",
            "SystemWifiTurnOn", 
            "SystemBluetoothTurnOff",
            "SystemBluetoothTurnOn",
            "ClockStopWatchRunning",
            "CameraTakePhoto",
            "ContactsAddContact",
            "SimpleSmsSend"
        ]
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            if self.env and hasattr(self.env, 'close'):
                self.env.close()
        except Exception as e:
            self.logger.warning("TaskRunner", f"Error during cleanup: {e}")

def main():
    """Main function for running Android World tasks."""
    parser = argparse.ArgumentParser(description="Run Android World tasks with Agent-S integration")
    parser.add_argument("--task", type=str, help="Specific task to run")
    parser.add_argument("--tasks", nargs="+", help="List of tasks to run")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per task")
    parser.add_argument("--console-port", type=int, default=5554, help="Emulator console port")
    parser.add_argument("--perform-emulator-setup", action="store_true", help="Perform emulator setup")
    parser.add_argument("--adb-path", type=str, help="Path to ADB executable")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--config", type=str, default="default", help="Configuration preset to use")
    
    args = parser.parse_args()
    
    # Initialize task runner
    runner = AndroidWorldTaskRunner(
        console_port=args.console_port,
        perform_emulator_setup=args.perform_emulator_setup,
        adb_path=args.adb_path,
        config_name=args.config
    )
    
    try:
        if args.list_tasks:
            # List available tasks
            tasks = runner.get_available_tasks()
            print("📋 Available Android World Tasks:")
            print("=" * 50)
            for i, task in enumerate(tasks, 1):
                print(f"   {i:2d}. {task}")
            return
        
        if args.task:
            # Run single task
            print(f"🎯 Running single task: {args.task}")
            result = runner.run_task(args.task, args.max_steps)
            print(f"✅ Task completed with score: {result['success_score']:.2f}")
            
        elif args.tasks:
            # Run task suite
            print(f"🏃 Running task suite with {len(args.tasks)} tasks")
            results = runner.run_task_suite(args.tasks, args.max_steps)
            
            print(f"\n📊 Task Suite Results:")
            print(f"   Total Tasks: {results['total_tasks']}")
            print(f"   Completed: {results['completed_tasks']}")
            print(f"   Failed: {results['failed_tasks']}")
            print(f"   Average Success Score: {results['average_success_score']:.2f}")
            print(f"   Environment: {results['environment_type']}")
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved to: {args.output}")
        
        else:
            # Run default task suite
            default_tasks = [
                "SystemWifiTurnOff",
                "SystemBluetoothTurnOn",
                "ClockStopWatchRunning"
            ]
            print(f"🏃 Running default task suite")
            results = runner.run_task_suite(default_tasks, args.max_steps)
            
            print(f"\n📊 Default Task Suite Results:")
            print(f"   Total Tasks: {results['total_tasks']}")
            print(f"   Completed: {results['completed_tasks']}")
            print(f"   Failed: {results['failed_tasks']}")
            print(f"   Average Success Score: {results['average_success_score']:.2f}")
            print(f"   Environment: {results['environment_type']}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Task execution interrupted by user")
    except Exception as e:
        print(f"❌ Error during task execution: {e}")
    finally:
        runner.close()

if __name__ == "__main__":
    main() 