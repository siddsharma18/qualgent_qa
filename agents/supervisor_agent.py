#!/usr/bin/env python3
"""
Robust SupervisorAgent for QA system evaluation and analysis.
Simulates a human QA supervisor/reviewer with comprehensive analysis capabilities.
"""

import json
import statistics
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os

from utils.logger import QALogger

class TestOutcome(Enum):
    """Test outcome classification."""
    PASSED = "passed"
    FAILED = "failed"
    FLAKY = "flaky"
    TIMEOUT = "timeout"
    ERROR = "error"

class Severity(Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SubgoalAnalysis:
    """Analysis of a specific subgoal performance."""
    name: str
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    success_rate: float
    avg_execution_time: float
    avg_verification_confidence: float
    retry_count: int
    replan_count: int
    is_flaky: bool
    common_failure_reasons: List[str]
    recommendations: List[str]

@dataclass
class AgentPerformance:
    """Performance metrics for each agent."""
    agent_name: str
    total_operations: int
    success_rate: float
    avg_operation_time: float
    error_count: int
    timeout_count: int
    strategy_usage: Dict[str, int]
    confidence_scores: List[float]

@dataclass
class Issue:
    """Represents an issue found during analysis."""
    id: str
    severity: Severity
    category: str
    description: str
    affected_components: List[str]
    recommendations: List[str]
    occurrence_count: int
    first_seen: str
    last_seen: str

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    # Basic metrics
    total_goals: int
    successful_goals: int
    failed_goals: int
    flaky_goals: int
    overall_success_rate: float
    
    # Timing metrics
    avg_planning_time: float
    avg_execution_time: float
    avg_verification_time: float
    avg_total_time: float
    
    # Performance metrics
    total_replans: int
    total_retries: int
    avg_retries_per_goal: float
    avg_replans_per_goal: float
    
    # Agent performance
    planner_performance: AgentPerformance
    executor_performance: AgentPerformance
    verifier_performance: AgentPerformance
    
    # Subgoal analysis
    subgoal_analysis: Dict[str, SubgoalAnalysis]
    flaky_subgoals: List[str]
    
    # Issues and recommendations
    issues: List[Issue]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    # Test context
    test_duration: float
    test_timestamp: str
    configuration_used: str
    
    # Optional visual data
    ui_traces: Optional[List[Dict]] = None
    visual_annotations: Optional[List[Dict]] = None

class SupervisorAgent:
    """
    A robust supervisor agent that analyzes QA system performance
    and generates comprehensive evaluation reports.
    """
    
    def __init__(self, 
                 logger: Optional[QALogger] = None,
                 enable_visual_analysis: bool = False,
                 confidence_threshold: float = 0.7,
                 flaky_threshold: float = 0.3):
        """
        Initialize the supervisor agent.
        
        Args:
            logger: Optional logger for recording analysis
            enable_visual_analysis: Whether to analyze UI traces
            confidence_threshold: Minimum confidence for passing tests
            flaky_threshold: Threshold for detecting flaky behavior
        """
        self.logger = logger or QALogger()
        self.enable_visual_analysis = enable_visual_analysis
        self.confidence_threshold = confidence_threshold
        self.flaky_threshold = flaky_threshold
        
        # Analysis statistics
        self.stats = {
            'total_evaluations': 0,
            'total_issues_found': 0,
            'avg_analysis_time': 0.0
        }
    
    def evaluate_logs(self, 
                     log_path: str = "qa_logs.json",
                     config: Optional[Dict] = None,
                     test_context: Optional[Dict] = None) -> EvaluationReport:
        """
        Evaluate QA logs and generate a comprehensive report.
        
        Args:
            log_path: Path to the QA logs file
            config: Optional configuration used during testing
            test_context: Optional test context information
            
        Returns:
            EvaluationReport: Comprehensive evaluation report
        """
        start_time = time.time()
        self.stats['total_evaluations'] += 1
        
        self.logger.info("Supervisor", "Starting log evaluation", {
            "log_path": log_path,
            "enable_visual_analysis": self.enable_visual_analysis
        })
        
        # Load and validate logs
        logs = self._load_logs(log_path)
        if not logs:
            return self._create_error_report("Failed to load logs")
        
        # Analyze logs
        analysis_result = self._analyze_logs(logs, config, test_context)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(analysis_result, start_time)
        
        # Save report
        self._save_report(report)
        
        # Update statistics
        analysis_time = time.time() - start_time
        self.stats['avg_analysis_time'] = (
            (self.stats['avg_analysis_time'] * (self.stats['total_evaluations'] - 1) + analysis_time) / 
            self.stats['total_evaluations']
        )
        
        self.logger.info("Supervisor", "Evaluation complete", {
            "total_goals": report.total_goals,
            "success_rate": report.overall_success_rate,
            "issues_found": len(report.issues),
            "analysis_time": analysis_time
        })
        
        return report
    
    def _load_logs(self, log_path: str) -> List[Dict]:
        """Load and validate QA logs."""
        try:
            if not os.path.exists(log_path):
                self.logger.warning("Supervisor", "Log file not found", {"path": log_path})
                return []
            
            with open(log_path, "r") as f:
                logs = json.load(f)
            
            if not isinstance(logs, list):
                logs = [logs]
            
            # Validate log structure
            validated_logs = []
            for log in logs:
                if self._validate_log_entry(log):
                    validated_logs.append(log)
                else:
                    self.logger.warning("Supervisor", "Invalid log entry", {"log": log})
            
            return validated_logs
            
        except Exception as e:
            self.logger.error("Supervisor", "Failed to load logs", {"error": str(e)})
            return []
    
    def _validate_log_entry(self, log: Dict) -> bool:
        """Validate a log entry has required fields."""
        required_fields = ['status', 'goal']
        return all(field in log for field in required_fields)
    
    def _analyze_logs(self, logs: List[Dict], config: Optional[Dict], test_context: Optional[Dict]) -> Dict[str, Any]:
        """Perform comprehensive log analysis."""
        analysis = {
            'goals': self._analyze_goals(logs),
            'subgoals': self._analyze_subgoals(logs),
            'agents': self._analyze_agent_performance(logs),
            'issues': self._detect_issues(logs),
            'timing': self._analyze_timing(logs),
            'patterns': self._detect_patterns(logs),
            'config': config,
            'context': test_context
        }
        
        return analysis
    
    def _analyze_goals(self, logs: List[Dict]) -> Dict[str, Any]:
        """Analyze goal-level performance."""
        total_goals = len(logs)
        successful_goals = 0
        failed_goals = 0
        flaky_goals = 0
        
        goal_details = []
        
        for log in logs:
            status = log.get('status', 'unknown')
            success_rate = log.get('success_rate', 0.0)
            
            if status == 'success':
                successful_goals += 1
            elif status == 'failed':
                failed_goals += 1
            
            # Detect flaky goals (partial success)
            if 0.0 < success_rate < 1.0:
                flaky_goals += 1
            
            goal_details.append({
                'goal': log.get('goal', 'Unknown'),
                'status': status,
                'success_rate': success_rate,
                'execution_time': log.get('execution_time', 0.0),
                'iterations': log.get('iterations', 0),
                'completed_subgoals': len(log.get('completed_subgoals', [])),
                'failed_subgoals': len(log.get('failed_subgoals', []))
            })
        
        return {
            'total': total_goals,
            'successful': successful_goals,
            'failed': failed_goals,
            'flaky': flaky_goals,
            'success_rate': successful_goals / max(total_goals, 1),
            'details': goal_details
        }
    
    def _analyze_subgoals(self, logs: List[Dict]) -> Dict[str, SubgoalAnalysis]:
        """Analyze subgoal performance across all logs."""
        subgoal_stats = {}
        
        for log in logs:
            # Analyze completed subgoals
            for subgoal in log.get('completed_subgoals', []):
                if subgoal not in subgoal_stats:
                    subgoal_stats[subgoal] = {
                        'total_attempts': 0,
                        'successful_attempts': 0,
                        'failed_attempts': 0,
                        'execution_times': [],
                        'verification_confidences': [],
                        'retry_counts': [],
                        'replan_counts': [],
                        'failure_reasons': []
                    }
                
                stats = subgoal_stats[subgoal]
                stats['total_attempts'] += 1
                stats['successful_attempts'] += 1
            
            # Analyze failed subgoals
            for subgoal in log.get('failed_subgoals', []):
                if subgoal not in subgoal_stats:
                    subgoal_stats[subgoal] = {
                        'total_attempts': 0,
                        'successful_attempts': 0,
                        'failed_attempts': 0,
                        'execution_times': [],
                        'verification_confidences': [],
                        'retry_counts': [],
                        'replan_counts': [],
                        'failure_reasons': []
                    }
                
                stats = subgoal_stats[subgoal]
                stats['total_attempts'] += 1
                stats['failed_attempts'] += 1
        
        # Convert to SubgoalAnalysis objects
        subgoal_analysis = {}
        for subgoal, stats in subgoal_stats.items():
            success_rate = stats['successful_attempts'] / max(stats['total_attempts'], 1)
            
            # Detect flaky behavior
            is_flaky = (stats['successful_attempts'] > 0 and 
                       stats['failed_attempts'] > 0 and
                       success_rate > self.flaky_threshold and
                       success_rate < (1 - self.flaky_threshold))
            
            subgoal_analysis[subgoal] = SubgoalAnalysis(
                name=subgoal,
                total_attempts=stats['total_attempts'],
                successful_attempts=stats['successful_attempts'],
                failed_attempts=stats['failed_attempts'],
                success_rate=success_rate,
                avg_execution_time=statistics.mean(stats['execution_times']) if stats['execution_times'] else 0.0,
                avg_verification_confidence=statistics.mean(stats['verification_confidences']) if stats['verification_confidences'] else 0.0,
                retry_count=sum(stats['retry_counts']),
                replan_count=sum(stats['replan_counts']),
                is_flaky=is_flaky,
                common_failure_reasons=list(set(stats['failure_reasons'])),
                recommendations=self._generate_subgoal_recommendations(stats, is_flaky)
            )
        
        return subgoal_analysis
    
    def _analyze_agent_performance(self, logs: List[Dict]) -> Dict[str, AgentPerformance]:
        """Analyze performance of each agent."""
        planner_stats = {'operations': 0, 'successes': 0, 'times': [], 'errors': 0}
        executor_stats = {'operations': 0, 'successes': 0, 'times': [], 'errors': 0}
        verifier_stats = {'operations': 0, 'successes': 0, 'times': [], 'errors': 0}
        
        for log in logs:
            # Planner analysis
            planning_result = log.get('planning_result', {})
            if planning_result:
                planner_stats['operations'] += 1
                if planning_result.get('status') == 'success':
                    planner_stats['successes'] += 1
                planner_stats['times'].append(planning_result.get('planning_time', 0))
            
            # Executor analysis
            executor_stats['operations'] += 1
            if log.get('status') == 'success':
                executor_stats['successes'] += 1
            executor_stats['times'].append(log.get('execution_time', 0))
            
            # Verifier analysis
            verifier_stats['operations'] += len(log.get('completed_subgoals', [])) + len(log.get('failed_subgoals', []))
            verifier_stats['successes'] += len(log.get('completed_subgoals', []))
        
        return {
            'planner': AgentPerformance(
                agent_name='PlannerAgent',
                total_operations=planner_stats['operations'],
                success_rate=planner_stats['successes'] / max(planner_stats['operations'], 1),
                avg_operation_time=statistics.mean(planner_stats['times']) if planner_stats['times'] else 0.0,
                error_count=planner_stats['errors'],
                timeout_count=0,
                strategy_usage={},
                confidence_scores=[]
            ),
            'executor': AgentPerformance(
                agent_name='ExecutorAgent',
                total_operations=executor_stats['operations'],
                success_rate=executor_stats['successes'] / max(executor_stats['operations'], 1),
                avg_operation_time=statistics.mean(executor_stats['times']) if executor_stats['times'] else 0.0,
                error_count=executor_stats['errors'],
                timeout_count=0,
                strategy_usage={},
                confidence_scores=[]
            ),
            'verifier': AgentPerformance(
                agent_name='VerifierAgent',
                total_operations=verifier_stats['operations'],
                success_rate=verifier_stats['successes'] / max(verifier_stats['operations'], 1),
                avg_operation_time=0.0,
                error_count=verifier_stats['errors'],
                timeout_count=0,
                strategy_usage={},
                confidence_scores=[]
            )
        }
    
    def _detect_issues(self, logs: List[Dict]) -> List[Issue]:
        """Detect issues and problems in the test execution."""
        issues = []
        issue_patterns = {}
        
        for log in logs:
            # Detect repeated failures
            if log.get('status') == 'failed':
                failure_reason = log.get('reason', 'Unknown failure')
                if failure_reason not in issue_patterns:
                    issue_patterns[failure_reason] = {
                        'count': 0,
                        'first_seen': datetime.now().isoformat(),
                        'affected_goals': []
                    }
                issue_patterns[failure_reason]['count'] += 1
                issue_patterns[failure_reason]['affected_goals'].append(log.get('goal', 'Unknown'))
            
            # Detect high retry counts
            total_retries = log.get('stats', {}).get('total_retries', 0)
            if total_retries > 5:
                issues.append(Issue(
                    id=f"high_retries_{len(issues)}",
                    severity=Severity.MEDIUM,
                    category="Performance",
                    description=f"High retry count ({total_retries}) for goal: {log.get('goal')}",
                    affected_components=['ExecutorAgent'],
                    recommendations=["Investigate UI element finding strategies", "Consider increasing timeout values"],
                    occurrence_count=1,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat()
                ))
            
            # Detect repeated replans
            total_replans = log.get('stats', {}).get('total_replans', 0)
            if total_replans > 3:
                issues.append(Issue(
                    id=f"repeated_replans_{len(issues)}",
                    severity=Severity.HIGH,
                    category="Planning",
                    description=f"Repeated replanning ({total_replans}) for goal: {log.get('goal')}",
                    affected_components=['PlannerAgent'],
                    recommendations=["Improve planning accuracy", "Add more planning strategies"],
                    occurrence_count=1,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat()
                ))
        
        # Convert patterns to issues
        for reason, pattern in issue_patterns.items():
            if pattern['count'] > 1:
                severity = Severity.HIGH if pattern['count'] > 3 else Severity.MEDIUM
                issues.append(Issue(
                    id=f"repeated_failure_{len(issues)}",
                    severity=severity,
                    category="Reliability",
                    description=f"Repeated failure: {reason}",
                    affected_components=['All'],
                    recommendations=["Investigate root cause", "Add error handling"],
                    occurrence_count=pattern['count'],
                    first_seen=pattern['first_seen'],
                    last_seen=datetime.now().isoformat()
                ))
        
        return issues
    
    def _analyze_timing(self, logs: List[Dict]) -> Dict[str, float]:
        """Analyze timing patterns."""
        planning_times = []
        execution_times = []
        verification_times = []
        total_times = []
        
        for log in logs:
            planning_result = log.get('planning_result', {})
            planning_times.append(planning_result.get('planning_time', 0))
            execution_times.append(log.get('execution_time', 0))
            total_times.append(log.get('execution_time', 0))
        
        return {
            'avg_planning_time': statistics.mean(planning_times) if planning_times else 0.0,
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0.0,
            'avg_verification_time': statistics.mean(verification_times) if verification_times else 0.0,
            'avg_total_time': statistics.mean(total_times) if total_times else 0.0,
            'min_planning_time': min(planning_times) if planning_times else 0.0,
            'max_planning_time': max(planning_times) if planning_times else 0.0,
            'min_execution_time': min(execution_times) if execution_times else 0.0,
            'max_execution_time': max(execution_times) if execution_times else 0.0
        }
    
    def _detect_patterns(self, logs: List[Dict]) -> Dict[str, Any]:
        """Detect patterns in test execution."""
        patterns = {
            'common_failure_goals': {},
            'successful_patterns': [],
            'flaky_patterns': [],
            'performance_bottlenecks': []
        }
        
        for log in logs:
            goal = log.get('goal', 'Unknown')
            status = log.get('status', 'unknown')
            success_rate = log.get('success_rate', 0.0)
            
            if status == 'failed':
                patterns['common_failure_goals'][goal] = patterns['common_failure_goals'].get(goal, 0) + 1
            elif status == 'success' and success_rate == 1.0:
                patterns['successful_patterns'].append(goal)
            elif 0.0 < success_rate < 1.0:
                patterns['flaky_patterns'].append(goal)
        
        return patterns
    
    def _generate_comprehensive_report(self, analysis: Dict[str, Any], start_time: float) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        goals_analysis = analysis['goals']
        subgoals_analysis = analysis['subgoals']
        agents_analysis = analysis['agents']
        issues = analysis['issues']
        timing = analysis['timing']
        patterns = analysis['patterns']
        
        # Identify flaky subgoals
        flaky_subgoals = [
            name for name, analysis in subgoals_analysis.items() 
            if analysis.is_flaky
        ]
        
        # Generate strengths and weaknesses
        strengths = self._identify_strengths(goals_analysis, agents_analysis, timing)
        weaknesses = self._identify_weaknesses(goals_analysis, issues, patterns)
        recommendations = self._generate_recommendations(analysis)
        
        return EvaluationReport(
            total_goals=goals_analysis['total'],
            successful_goals=goals_analysis['successful'],
            failed_goals=goals_analysis['failed'],
            flaky_goals=goals_analysis['flaky'],
            overall_success_rate=goals_analysis['success_rate'],
            avg_planning_time=timing['avg_planning_time'],
            avg_execution_time=timing['avg_execution_time'],
            avg_verification_time=timing['avg_verification_time'],
            avg_total_time=timing['avg_total_time'],
            total_replans=sum(log.get('stats', {}).get('total_replans', 0) for log in analysis.get('logs', [])),
            total_retries=sum(log.get('stats', {}).get('total_retries', 0) for log in analysis.get('logs', [])),
            avg_retries_per_goal=sum(log.get('stats', {}).get('total_retries', 0) for log in analysis.get('logs', [])) / max(goals_analysis['total'], 1),
            avg_replans_per_goal=sum(log.get('stats', {}).get('total_replans', 0) for log in analysis.get('logs', [])) / max(goals_analysis['total'], 1),
            planner_performance=agents_analysis['planner'],
            executor_performance=agents_analysis['executor'],
            verifier_performance=agents_analysis['verifier'],
            subgoal_analysis=subgoals_analysis,
            flaky_subgoals=flaky_subgoals,
            issues=issues,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            test_duration=time.time() - start_time,
            test_timestamp=datetime.now().isoformat(),
            configuration_used=analysis.get('config', {}).get('name', 'default') if analysis.get('config') else 'default'
        )
    
    def _identify_strengths(self, goals_analysis: Dict, agents_analysis: Dict, timing: Dict) -> List[str]:
        """Identify system strengths."""
        strengths = []
        
        if goals_analysis['success_rate'] > 0.8:
            strengths.append("High overall success rate")
        
        if agents_analysis['planner'].success_rate > 0.9:
            strengths.append("Excellent planning accuracy")
        
        if agents_analysis['executor'].success_rate > 0.9:
            strengths.append("Reliable execution performance")
        
        if timing['avg_execution_time'] < 5.0:
            strengths.append("Fast execution times")
        
        if timing['avg_planning_time'] < 1.0:
            strengths.append("Efficient planning")
        
        return strengths
    
    def _identify_weaknesses(self, goals_analysis: Dict, issues: List[Issue], patterns: Dict) -> List[str]:
        """Identify system weaknesses."""
        weaknesses = []
        
        if goals_analysis['success_rate'] < 0.7:
            weaknesses.append("Low success rate")
        
        if goals_analysis['flaky'] > 0:
            weaknesses.append(f"Flaky behavior detected in {goals_analysis['flaky']} goals")
        
        if len(issues) > 5:
            weaknesses.append("Multiple issues detected")
        
        if patterns['common_failure_goals']:
            weaknesses.append("Repeated failures in specific goals")
        
        return weaknesses
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        goals_analysis = analysis['goals']
        issues = analysis['issues']
        patterns = analysis['patterns']
        
        if goals_analysis['success_rate'] < 0.8:
            recommendations.append("Improve overall system reliability")
        
        if goals_analysis['flaky'] > 0:
            recommendations.append("Investigate and fix flaky behavior")
        
        if len(issues) > 3:
            recommendations.append("Address critical issues before production deployment")
        
        if patterns['common_failure_goals']:
            recommendations.append("Focus testing on commonly failing goals")
        
        if not recommendations:
            recommendations.append("System performing well - consider expanding test coverage")
        
        return recommendations
    
    def _generate_subgoal_recommendations(self, stats: Dict, is_flaky: bool) -> List[str]:
        """Generate recommendations for specific subgoals."""
        recommendations = []
        
        if is_flaky:
            recommendations.append("Investigate flaky behavior")
        
        if stats['failed_attempts'] > stats['successful_attempts']:
            recommendations.append("Improve subgoal execution strategy")
        
        if stats['retry_counts'] and sum(stats['retry_counts']) > 3:
            recommendations.append("Reduce retry frequency")
        
        return recommendations
    
    def _save_report(self, report: EvaluationReport):
        """Save the evaluation report to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = f"evaluation_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self._report_to_dict(report), f, indent=2)
        
        # Save Markdown report
        md_path = f"evaluation_report_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(self._generate_markdown_report(report))
        
        self.logger.info("Supervisor", "Reports saved", {
            "json_path": json_path,
            "markdown_path": md_path
        })
    
    def _report_to_dict(self, report: EvaluationReport) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_goals": report.total_goals,
            "successful_goals": report.successful_goals,
            "failed_goals": report.failed_goals,
            "flaky_goals": report.flaky_goals,
            "overall_success_rate": report.overall_success_rate,
            "avg_planning_time": report.avg_planning_time,
            "avg_execution_time": report.avg_execution_time,
            "avg_verification_time": report.avg_verification_time,
            "avg_total_time": report.avg_total_time,
            "total_replans": report.total_replans,
            "total_retries": report.total_retries,
            "avg_retries_per_goal": report.avg_retries_per_goal,
            "avg_replans_per_goal": report.avg_replans_per_goal,
            "planner_performance": {
                "agent_name": report.planner_performance.agent_name,
                "total_operations": report.planner_performance.total_operations,
                "success_rate": report.planner_performance.success_rate,
                "avg_operation_time": report.planner_performance.avg_operation_time,
                "error_count": report.planner_performance.error_count
            },
            "executor_performance": {
                "agent_name": report.executor_performance.agent_name,
                "total_operations": report.executor_performance.total_operations,
                "success_rate": report.executor_performance.success_rate,
                "avg_operation_time": report.executor_performance.avg_operation_time,
                "error_count": report.executor_performance.error_count
            },
            "verifier_performance": {
                "agent_name": report.verifier_performance.agent_name,
                "total_operations": report.verifier_performance.total_operations,
                "success_rate": report.verifier_performance.success_rate,
                "avg_operation_time": report.verifier_performance.avg_operation_time,
                "error_count": report.verifier_performance.error_count
            },
            "flaky_subgoals": report.flaky_subgoals,
            "issues": [
                {
                    "id": issue.id,
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "description": issue.description,
                    "affected_components": issue.affected_components,
                    "recommendations": issue.recommendations,
                    "occurrence_count": issue.occurrence_count
                }
                for issue in report.issues
            ],
            "strengths": report.strengths,
            "weaknesses": report.weaknesses,
            "recommendations": report.recommendations,
            "test_duration": report.test_duration,
            "test_timestamp": report.test_timestamp,
            "configuration_used": report.configuration_used
        }
    
    def _generate_markdown_report(self, report: EvaluationReport) -> str:
        """Generate a human-readable Markdown report."""
        md = f"""# QA System Evaluation Report

**Generated:** {report.test_timestamp}  
**Configuration:** {report.configuration_used}  
**Test Duration:** {report.test_duration:.2f}s

## 📊 Executive Summary

- **Total Goals:** {report.total_goals}
- **Successful Goals:** {report.successful_goals}
- **Failed Goals:** {report.failed_goals}
- **Flaky Goals:** {report.flaky_goals}
- **Overall Success Rate:** {report.overall_success_rate:.1%}

## ⏱️ Performance Metrics

| Metric | Value |
|--------|-------|
| Average Planning Time | {report.avg_planning_time:.2f}s |
| Average Execution Time | {report.avg_execution_time:.2f}s |
| Average Verification Time | {report.avg_verification_time:.2f}s |
| Average Total Time | {report.avg_total_time:.2f}s |
| Total Replans | {report.total_replans} |
| Total Retries | {report.total_retries} |

## 🤖 Agent Performance

### Planner Agent
- **Success Rate:** {report.planner_performance.success_rate:.1%}
- **Total Operations:** {report.planner_performance.total_operations}
- **Average Time:** {report.planner_performance.avg_operation_time:.2f}s

### Executor Agent
- **Success Rate:** {report.executor_performance.success_rate:.1%}
- **Total Operations:** {report.executor_performance.total_operations}
- **Average Time:** {report.executor_performance.avg_operation_time:.2f}s

### Verifier Agent
- **Success Rate:** {report.verifier_performance.success_rate:.1%}
- **Total Operations:** {report.verifier_performance.total_operations}
- **Average Time:** {report.verifier_performance.avg_operation_time:.2f}s

## 🎯 Strengths

"""
        
        for strength in report.strengths:
            md += f"- {strength}\n"
        
        md += "\n## ⚠️ Weaknesses\n\n"
        
        for weakness in report.weaknesses:
            md += f"- {weakness}\n"
        
        if report.issues:
            md += "\n## 🚨 Issues\n\n"
            for issue in report.issues:
                md += f"### {issue.severity.value.upper()}: {issue.category}\n"
                md += f"**Description:** {issue.description}\n"
                md += f"**Occurrences:** {issue.occurrence_count}\n"
                md += f"**Recommendations:**\n"
                for rec in issue.recommendations:
                    md += f"- {rec}\n"
                md += "\n"
        
        if report.flaky_subgoals:
            md += "\n## 🔄 Flaky Subgoals\n\n"
            for subgoal in report.flaky_subgoals:
                md += f"- {subgoal}\n"
        
        md += "\n## 💡 Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- {rec}\n"
        
        return md
    
    def _create_error_report(self, reason: str) -> EvaluationReport:
        """Create an error report when analysis fails."""
        return EvaluationReport(
            total_goals=0,
            successful_goals=0,
            failed_goals=0,
            flaky_goals=0,
            overall_success_rate=0.0,
            avg_planning_time=0.0,
            avg_execution_time=0.0,
            avg_verification_time=0.0,
            avg_total_time=0.0,
            total_replans=0,
            total_retries=0,
            avg_retries_per_goal=0.0,
            avg_replans_per_goal=0.0,
            planner_performance=AgentPerformance("PlannerAgent", 0, 0.0, 0.0, 0, 0, {}, []),
            executor_performance=AgentPerformance("ExecutorAgent", 0, 0.0, 0.0, 0, 0, {}, []),
            verifier_performance=AgentPerformance("VerifierAgent", 0, 0.0, 0.0, 0, 0, {}, []),
            subgoal_analysis={},
            flaky_subgoals=[],
            issues=[Issue("error", Severity.CRITICAL, "System", reason, [], [], 1, "", "")],
            strengths=[],
            weaknesses=[reason],
            recommendations=["Fix the error and re-run analysis"],
            test_duration=0.0,
            test_timestamp=datetime.now().isoformat(),
            configuration_used="unknown"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get supervisor statistics."""
        return self.stats.copy()