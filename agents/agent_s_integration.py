#!/usr/bin/env python3
"""
Agent-S Integration Module
Bridges our robust QA agents with Agent-S's messaging structure and architecture.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from agents.planner_agent import PlannerAgent, PlanningResult, Subgoal
from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.verifier_agent import VerifierAgent, VerificationResult
from agents.supervisor_agent import SupervisorAgent, EvaluationReport
from utils.logger import QALogger
from config.qa_config import get_config

logger = logging.getLogger("agent_s_integration")

class AgentSMessageType(Enum):
    """Message types for Agent-S communication."""
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    SUPERVISION = "supervision"
    REPLANNING = "replanning"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class AgentSMessage:
    """Structured message for Agent-S communication."""
    message_type: AgentSMessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: float
    message_id: str
    correlation_id: Optional[str] = None

class AgentSGraphSearchAgent:
    """
    Agent-S compatible GraphSearchAgent that uses our robust QA agents.
    Implements the Agent-S messaging structure and execution framework.
    """
    
    def __init__(self, 
                 env,
                 engine_params: Dict,
                 platform: str = "android",
                 action_space: str = "android_env",
                 observation_type: str = "mixed",
                 search_engine: Optional[str] = None,
                 memory_root_path: str = os.getcwd(),
                 memory_folder_name: str = "qa_kb",
                 kb_release_tag: str = "v1.0.0"):
        """
        Initialize the Agent-S GraphSearchAgent.
        
        Args:
            env: AndroidEnv instance
            engine_params: LLM engine parameters
            platform: Platform (android)
            action_space: Action space type
            observation_type: Observation type
            search_engine: Search engine for knowledge retrieval
            memory_root_path: Path to memory directory
            memory_folder_name: Name of memory folder
            kb_release_tag: Knowledge base release tag
        """
        self.env = env
        self.engine_params = engine_params
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.search_engine = search_engine
        self.memory_root_path = memory_root_path
        self.memory_folder_name = memory_folder_name
        self.kb_release_tag = kb_release_tag
        
        # Initialize knowledge base path
        self.local_kb_path = os.path.join(memory_root_path, memory_folder_name)
        os.makedirs(self.local_kb_path, exist_ok=True)
        
        # Initialize robust QA agents
        self.config = get_config("default")
        self.logger = QALogger(log_level=self.config.logger.log_level, enable_console=True)
        
        self.planner = PlannerAgent(
            logger=self.logger,
            enable_template_matching=self.config.planner.enable_template_matching,
            enable_semantic_planning=self.config.planner.enable_semantic_planning,
            enable_adaptive_planning=self.config.planner.enable_adaptive_planning,
            max_planning_time=self.config.planner.max_planning_time,
            min_confidence=self.config.planner.min_confidence,
            enable_plan_optimization=self.config.planner.enable_plan_optimization
        )
        
        self.executor = ExecutorAgent(
            env=env,
            logger=self.logger,
            max_retries=self.config.executor.max_retries,
            retry_delay=self.config.executor.retry_delay,
            action_timeout=self.config.executor.action_timeout,
            ui_settle_time=self.config.executor.ui_settle_time,
            enable_validation=self.config.executor.enable_validation,
            min_confidence=self.config.executor.min_confidence
        )
        
        self.verifier = VerifierAgent(
            logger=self.logger,
            min_change_ratio=self.config.verifier.min_change_ratio,
            fuzzy_threshold=self.config.verifier.fuzzy_threshold,
            max_verification_time=self.config.verifier.max_verification_time,
            enable_advanced_analysis=self.config.verifier.enable_advanced_analysis,
            enable_state_tracking=self.config.verifier.enable_state_tracking,
            min_confidence=self.config.verifier.min_confidence
        )
        
        self.supervisor = SupervisorAgent(
            logger=self.logger,
            enable_visual_analysis=True,
            confidence_threshold=0.7,
            flaky_threshold=0.3
        )
        
        # Agent-S state variables
        self.requires_replan: bool = True
        self.needs_next_subtask: bool = True
        self.step_count: int = 0
        self.turn_count: int = 0
        self.failure_feedback: str = ""
        self.should_send_action: bool = False
        self.completed_tasks: List[Subgoal] = []
        self.current_subtask: Optional[Subgoal] = None
        self.subtasks: List[Subgoal] = []
        self.search_query: str = ""
        self.subtask_status: str = "Start"
        
        # Message history for Agent-S compatibility
        self.message_history: List[AgentSMessage] = []
        self.episode_trajectory: List[Dict[str, Any]] = []
        
        # Visual trace support
        self.visual_traces: List[Dict[str, Any]] = []
        self.enable_visual_tracing = True
    
    def reset(self) -> None:
        """Reset agent state and initialize components."""
        # Reset Agent-S state variables
        self.requires_replan = True
        self.needs_next_subtask = True
        self.step_count = 0
        self.turn_count = 0
        self.failure_feedback = ""
        self.should_send_action = False
        self.completed_tasks = []
        self.current_subtask = None
        self.subtasks = []
        self.search_query = ""
        self.subtask_status = "Start"
        
        # Reset message history
        self.message_history = []
        self.episode_trajectory = []
        self.visual_traces = []
        
        # Reset environment
        if self.env:
            obs = self.env.reset()
            self._capture_visual_trace("reset", obs)
    
    def reset_executor_state(self) -> None:
        """Reset executor and step counter."""
        self.step_count = 0
    
    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """
        Predict next UI action sequence using Agent-S pattern.
        
        Args:
            instruction: Natural language instruction
            observation: Current UI state observation
            
        Returns:
            Tuple of (agent info dict, list of actions)
        """
        # Initialize info dictionaries
        planner_info = {}
        executor_info = {}
        verifier_info = {}
        actions = []
        
        # Capture visual trace
        self._capture_visual_trace("predict", observation)
        
        # Agent-S execution loop
        while not self.should_send_action:
            self.subtask_status = "In"
            
            # Planning phase
            if self.requires_replan:
                logger.info("AgentS", "(RE)PLANNING...")
                planner_info, self.subtasks = self._planning_phase(instruction, observation)
                self.requires_replan = False
                
                # Send planning message
                self._send_message(
                    AgentSMessageType.PLANNING,
                    "planner",
                    "executor",
                    planner_info
                )
            
            # Execution phase
            if self.needs_next_subtask:
                logger.info("AgentS", "GETTING NEXT SUBTASK...")
                if self.subtasks:
                    self.current_subtask = self.subtasks.pop(0)
                    logger.info("AgentS", f"NEXT SUBTASK: {self.current_subtask.name}")
                    self.needs_next_subtask = False
                    self.subtask_status = "Start"
                else:
                    # All subtasks completed
                    self._send_message(
                        AgentSMessageType.SUCCESS,
                        "planner",
                        "supervisor",
                        {"status": "all_subtasks_completed"}
                    )
                    actions = ["DONE"]
                    break
            
            # Execute current subtask
            executor_info, actions = self._execution_phase(instruction, observation)
            self.step_count += 1
            
            # Verification phase
            if actions and "DONE" not in actions and "FAIL" not in actions:
                verifier_info = self._verification_phase(observation)
                
                # Send verification message
                self._send_message(
                    AgentSMessageType.VERIFICATION,
                    "verifier",
                    "planner",
                    verifier_info
                )
                
                # Check if verification failed
                if verifier_info.get("status") == "FAIL":
                    self.requires_replan = True
                    self.failure_feedback = f"Verification failed: {verifier_info.get('reason', 'Unknown error')}"
                    self.needs_next_subtask = True
                    self.reset_executor_state()
                    
                    # Send replanning message
                    self._send_message(
                        AgentSMessageType.REPLANNING,
                        "verifier",
                        "planner",
                        {"reason": self.failure_feedback}
                    )
                    
                    if self.subtasks:
                        self.should_send_action = False
                        continue
            
            # Handle execution results
            if "FAIL" in actions:
                self.requires_replan = True
                self.failure_feedback = f"Execution failed for subtask: {self.current_subtask.name if self.current_subtask else 'Unknown'}"
                self.needs_next_subtask = True
                self.reset_executor_state()
                
                # Send error message
                self._send_message(
                    AgentSMessageType.ERROR,
                    "executor",
                    "planner",
                    {"reason": self.failure_feedback}
                )
                
                if self.subtasks:
                    self.should_send_action = False
                    continue
                    
            elif "DONE" in actions:
                self.requires_replan = False
                if self.current_subtask:
                    self.completed_tasks.append(self.current_subtask)
                self.needs_next_subtask = True
                if self.subtasks:
                    self.should_send_action = False
                self.subtask_status = "Done"
                self.reset_executor_state()
            
            self.turn_count += 1
        
        # Reset for next iteration
        self.should_send_action = False
        
        # Combine info dictionaries
        info = {
            **planner_info,
            **executor_info,
            **verifier_info,
            "subtask": self.current_subtask.name if self.current_subtask else None,
            "subtask_info": self.current_subtask.description if self.current_subtask else None,
            "subtask_status": self.subtask_status,
            "step_count": self.step_count,
            "turn_count": self.turn_count,
            "completed_tasks": [task.name for task in self.completed_tasks],
            "remaining_tasks": [task.name for task in self.subtasks]
        }
        
        return info, actions
    
    def _planning_phase(self, instruction: str, observation: Dict) -> Tuple[Dict, List[Subgoal]]:
        """Execute planning phase using our robust planner."""
        try:
            planning_result = self.planner.plan(instruction)
            
            if planning_result.status.value == "success":
                return {
                    "planning_status": "success",
                    "planning_time": planning_result.planning_time,
                    "confidence": planning_result.confidence,
                    "strategies_used": planning_result.strategies_used,
                    "total_estimated_duration": planning_result.total_estimated_duration
                }, planning_result.subgoals
            else:
                return {
                    "planning_status": "failed",
                    "reason": f"Planning failed: {planning_result.status.value}"
                }, []
                
        except Exception as e:
            logger.error(f"Planning phase error: {e}")
            return {
                "planning_status": "error",
                "reason": str(e)
            }, []
    
    def _execution_phase(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Execute current subtask using our robust executor."""
        if not self.current_subtask:
            return {"execution_status": "error", "reason": "No current subtask"}, ["FAIL"]
        
        try:
            # Get UI tree from observation
            ui_tree = observation.get("ui_tree", [])
            if not ui_tree:
                ui_tree = self._extract_ui_tree(observation)
            
            # Execute the subtask
            execution_result = self.executor.execute(self.current_subtask.name, ui_tree)
            
            if execution_result.success:
                return {
                    "execution_status": "success",
                    "execution_time": execution_result.execution_time,
                    "retry_count": execution_result.retry_count,
                    "confidence": execution_result.confidence
                }, ["DONE"]
            else:
                return {
                    "execution_status": "failed",
                    "reason": execution_result.reason,
                    "execution_time": execution_result.execution_time,
                    "retry_count": execution_result.retry_count
                }, ["FAIL"]
                
        except Exception as e:
            logger.error(f"Execution phase error: {e}")
            return {
                "execution_status": "error",
                "reason": str(e)
            }, ["FAIL"]
    
    def _verification_phase(self, observation: Dict) -> Dict:
        """Verify subtask completion using our robust verifier."""
        if not self.current_subtask:
            return {"verification_status": "error", "reason": "No current subtask"}
        
        try:
            # Get UI tree from observation
            ui_tree = observation.get("ui_tree", [])
            if not ui_tree:
                ui_tree = self._extract_ui_tree(observation)
            
            # Verify the subtask
            verification_result = self.verifier.verify(
                self.current_subtask.name,
                ui_tree,
                self.current_subtask.description
            )
            
            return {
                "verification_status": verification_result.status.value,
                "confidence": verification_result.confidence,
                "reason": verification_result.reason,
                "verification_time": verification_result.verification_time,
                "strategies_used": verification_result.strategies_used
            }
            
        except Exception as e:
            logger.error(f"Verification phase error: {e}")
            return {
                "verification_status": "error",
                "reason": str(e)
            }
    
    def _extract_ui_tree(self, observation: Dict) -> List[Dict[str, Any]]:
        """Extract UI tree from observation."""
        # This is a simplified extraction - in practice, you'd parse the actual UI tree
        ui_tree = []
        
        # Extract from accessibility tree if available
        if "accessibility_tree" in observation:
            # Parse accessibility tree into UI elements
            # This is a placeholder - implement actual parsing
            ui_tree = [{"text": "placeholder", "type": "element"}]
        
        return ui_tree
    
    def _capture_visual_trace(self, event: str, observation: Dict) -> None:
        """Capture visual trace for analysis."""
        if not self.enable_visual_tracing:
            return
        
        trace_entry = {
            "timestamp": time.time(),
            "event": event,
            "step_count": self.step_count,
            "turn_count": self.turn_count,
            "current_subtask": self.current_subtask.name if self.current_subtask else None,
            "screenshot": observation.get("screenshot"),
            "accessibility_tree": observation.get("accessibility_tree"),
            "ui_tree": observation.get("ui_tree", [])
        }
        
        self.visual_traces.append(trace_entry)
    
    def _send_message(self, 
                     message_type: AgentSMessageType,
                     sender: str,
                     receiver: str,
                     content: Dict[str, Any]) -> None:
        """Send a message in Agent-S format."""
        message = AgentSMessage(
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=time.time(),
            message_id=f"msg_{len(self.message_history)}",
            correlation_id=f"episode_{self.turn_count}"
        )
        
        self.message_history.append(message)
        
        # Log message
        logger.info("AgentS", f"Agent-S Message: {sender} -> {receiver} [{message_type.value}]")
    
    def update_narrative_memory(self, trajectory: str) -> None:
        """Update narrative memory with episode trajectory."""
        self.episode_trajectory.append({
            "trajectory": trajectory,
            "timestamp": time.time(),
            "message_count": len(self.message_history),
            "visual_trace_count": len(self.visual_traces)
        })
    
    def update_episodic_memory(self, info: Dict, subtask_trajectory: str) -> str:
        """Update episodic memory with subtask information."""
        # This would integrate with Agent-S's knowledge base
        return subtask_trajectory
    
    def get_visual_traces(self) -> List[Dict[str, Any]]:
        """Get captured visual traces."""
        return self.visual_traces.copy()
    
    def get_message_history(self) -> List[AgentSMessage]:
        """Get message history."""
        return self.message_history.copy()
    
    def get_episode_trajectory(self) -> List[Dict[str, Any]]:
        """Get episode trajectory."""
        return self.episode_trajectory.copy()
    
    def generate_evaluation_report(self, log_path: str = "qa_logs.json") -> EvaluationReport:
        """Generate evaluation report using supervisor."""
        return self.supervisor.evaluate_logs(
            log_path=log_path,
            config={"name": "agent_s_integration", "settings": self.config.__dict__},
            test_context={
                "test_type": "agent_s_integration",
                "environment": "android",
                "visual_traces": self.visual_traces,
                "message_history": [msg.__dict__ for msg in self.message_history]
            }
        )

class AgentSManager:
    """
    Agent-S Manager component that handles planning and coordination.
    """
    
    def __init__(self, engine_params: Dict, local_kb_path: str):
        self.engine_params = engine_params
        self.local_kb_path = local_kb_path
        self.logger = QALogger()
    
    def get_action_queue(self, instruction: str, observation: Dict, failure_feedback: str = "") -> Tuple[Dict, List]:
        """Generate action queue using our robust planner."""
        # This would integrate with our PlannerAgent
        # For now, return a simple structure
        return {
            "planning_status": "success",
            "search_query": instruction
        }, []

class AgentSWorker:
    """
    Agent-S Worker component that handles execution.
    """
    
    def __init__(self, engine_params: Dict, env, local_kb_path: str):
        self.engine_params = engine_params
        self.env = env
        self.local_kb_path = local_kb_path
        self.logger = QALogger()
    
    def generate_next_action(self, instruction: str, subtask: str, obs: Dict) -> Tuple[Dict, List]:
        """Generate next action using our robust executor."""
        # This would integrate with our ExecutorAgent
        # For now, return a simple structure
        return {
            "execution_status": "success",
            "execution_plan": f"Execute: {subtask}"
        }, ["DONE"]
    
    def reset(self) -> None:
        """Reset worker state."""
        pass 