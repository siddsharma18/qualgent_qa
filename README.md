# QualGent QA System

A robust Android automation system with intelligent agents for executing UI tasks.

## Features

### Robust Executor Agent
- **Multiple Element Finding Strategies**: Exact text matching, fuzzy matching, semantic matching, keyword matching, and action-based matching
- **Retry Mechanisms**: Configurable retry attempts with exponential backoff
- **Timeout Protection**: Action execution timeouts to prevent hanging
- **Comprehensive Logging**: Structured logging with multiple output formats
- **Action Validation**: Pre and post-execution validation
- **Statistics Tracking**: Performance monitoring and execution statistics
- **Error Handling**: Graceful error handling with detailed error messages

### Robust Verifier Agent
- **Multiple Verification Strategies**: UI change verification, subgoal presence verification, state transition verification, element interaction verification, and semantic verification
- **Confidence Scoring**: Each verification includes a confidence score with weighted strategy combination
- **State Tracking**: Comprehensive UI state change tracking and history analysis
- **Advanced Analysis**: Semantic understanding of subgoals and expected state changes
- **Flexible Thresholds**: Configurable confidence thresholds and verification parameters
- **Performance Monitoring**: Detailed verification statistics and strategy success rates

### Robust Planner Agent
- **Multiple Planning Strategies**: Template-based planning, semantic planning, adaptive planning, and fallback planning
- **Structured Subgoals**: Detailed subgoal representation with priority, dependencies, and confidence scores
- **Replanning Capabilities**: Dynamic replanning when subgoals fail with failure analysis
- **Plan Optimization**: Automatic plan optimization and alternative plan generation
- **Comprehensive Statistics**: Detailed planning statistics and strategy performance tracking
- **Context Awareness**: Semantic understanding of goals and UI context

### Full Robust Loop Integration
- **Complete Workflow**: Goal → Plan → Execute → Verify → Replan
- **Coordinated Agents**: Seamless integration of PlannerAgent, ExecutorAgent, and VerifierAgent
- **Failure Recovery**: Automatic replanning when subgoals fail
- **Success Tracking**: Comprehensive success rate calculation and goal completion monitoring
- **Performance Monitoring**: Detailed statistics across all agents
- **Flexible Configuration**: Easy switching between performance and reliability presets

### Robust Supervisor Agent
- **Comprehensive Analysis**: Analyzes full execution logs, goal/subgoal success/failure, confidence scores
- **Issue Detection**: Identifies flaky subgoals, repeated replans, verifier disagreements
- **Performance Metrics**: Computes goal success rate, subgoal pass rate, timing metrics, retry counts
- **Structured Reporting**: Generates JSON and Markdown reports with strengths, weaknesses, recommendations
- **Visual Analysis**: Optional UI trace analysis and visual annotations
- **Error Classification**: Classifies outcomes as passed, failed, flaky, timeout, or error

### Agent-S Integration
- **Modular Messaging Structure**: Implements Agent-S's GraphSearchAgent, Manager, Worker architecture
- **Structured Communication**: Agent-S message types for planning, execution, verification, supervision
- **Visual Trace Support**: Comprehensive visual trace capture and analysis
- **Dynamic Replanning**: Agent-S compatible replanning logic with failure feedback
- **Knowledge Base Integration**: Agent-S knowledge base and episodic memory support
- **Android World Integration**: Full integration with android_world tasks (settings_wifi, clock_alarm, email_search)

### UI Parser
- **Multi-Strategy Element Finding**: 5 different strategies for finding UI elements
- **Confidence Scoring**: Each match includes a confidence score
- **Keyword Mapping**: Extensive mapping of UI element keywords and variations
- **Action Recognition**: Automatic detection of action types from subgoals
- **Fallback Mechanisms**: Multiple fallback strategies when primary matching fails

### Configuration System
- **Preset Configurations**: Default, high-performance, and high-reliability presets
- **Flexible Settings**: Configurable timeouts, retry counts, confidence thresholds
- **Runtime Configuration**: Easy configuration switching

## Project Structure

```
qualgent_qa/
├── agents/
│   ├── executor_agent.py      # Robust executor agent
│   ├── planner_agent.py       # Robust planner agent
│   ├── supervisor_agent.py    # Supervision agent
│   └── verifier_agent.py      # Robust verifier agent
├── utils/
│   ├── logger.py              # Structured logging system
│   └── ui_parser.py           # Multi-strategy UI parser
├── config/
│   └── qa_config.py           # Configuration management
├── env/
│   ├── android_env_wrapper.py # Android environment wrapper
│   ├── simple_task.textproto  # Sample task configuration
│   └── README.md              # Environment setup guide
├── logs/                      # Log files directory
├── requirements.txt           # Python dependencies
├── test_executor.py           # Test script for executor agent
├── test_verifier.py           # Test script for verifier agent
├── test_planner.py            # Test script for planner agent
├── test_integration.py        # Test script for executor-verifier integration
├── test_full_integration.py   # Test script for full robust loop
├── test_supervisor.py         # Test script for supervisor agent
├── test_agent_s_integration.py # Test script for Agent-S integration
├── run_robust_loop.py         # Main script for running robust loop
└── run_android_world_tasks.py # Android World task runner
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd qualgent_qa
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Android environment** (optional):
   ```bash
   # Follow the setup guide in env/README.md
   ```

## Quick Start

### Full Robust Loop Usage

```python
from test_full_integration import RobustAgentLoop, MockEnvironment

# Create environment (replace with real AndroidEnv in production)
env = MockEnvironment()

# Create robust agent loop
agent_loop = RobustAgentLoop(env, "default")

# Execute a high-level goal
result = agent_loop.execute_goal("Turn off Wi-Fi and enable Bluetooth")

# Check results
print(f"Status: {result['status']}")
print(f"Success Rate: {result['success_rate']:.1%}")
print(f"Completed Subgoals: {result['completed_subgoals']}")
```

### Command Line Usage

```bash
# Run with a specific goal
python run_robust_loop.py --goal "Turn off Wi-Fi and enable Bluetooth"

# Use high-performance configuration
python run_robust_loop.py --goal "Open Settings and enable USB Debugging" --config high_performance

# Enable verbose logging
python run_robust_loop.py --goal "Configure device settings" --verbose
```

### Individual Agent Usage

```python
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.planner_agent import PlannerAgent
from utils.logger import QALogger
from config.qa_config import get_config

# Create logger
logger = QALogger(log_level="INFO")

# Get configuration
config = get_config("default")

# Create executor agent
executor = ExecutorAgent(
    env=your_android_env,
    logger=logger,
    max_retries=config.executor.max_retries,
    retry_delay=config.executor.retry_delay,
    action_timeout=config.executor.action_timeout,
    ui_settle_time=config.executor.ui_settle_time,
    enable_validation=config.executor.enable_validation,
    min_confidence=config.executor.min_confidence
)

# Execute a subgoal
result = executor.execute("Turn Wi-Fi off", ui_tree)
print(f"Status: {result.status}")
print(f"Attempts: {result.attempts}")
print(f"Execution Time: {result.execution_time:.2f}s")
```

### Testing

Run the test scripts to verify the robust agent system:

```bash
# Test the full integration workflow
python test_full_integration.py

# Test individual components
python test_executor.py
python test_verifier.py
python test_planner.py
python test_integration.py
python test_supervisor.py

# Test Agent-S integration
python test_agent_s_integration.py

# Run Android World tasks
python run_android_world_tasks.py --list-tasks
python run_android_world_tasks.py --task SystemWifiTurnOff
python run_android_world_tasks.py --tasks SystemWifiTurnOff SystemBluetoothTurnOn ClockStopWatchRunning

# Test with specific goals
python run_robust_loop.py --goal "Turn off Wi-Fi"
python run_robust_loop.py --goal "Configure device settings" --config high_performance --verbose
```

## Configuration

### Available Presets

1. **Default Configuration**: Balanced performance and reliability
2. **High Performance**: Faster execution with fewer retries
3. **High Reliability**: Maximum reliability with more retries and longer timeouts

### Custom Configuration

```python
from config.qa_config import QASystemConfig, ExecutorConfig

# Create custom configuration
custom_config = QASystemConfig(
    executor=ExecutorConfig(
        max_retries=5,
        retry_delay=1.5,
        action_timeout=8.0,
        ui_settle_time=1.0,
        min_confidence=0.4
    )
)
```

## Key Improvements Made

### 1. **Robust Element Finding**
- **Multiple Strategies**: 5 different strategies for finding UI elements
- **Confidence Scoring**: Each match includes a confidence score
- **Fallback Mechanisms**: Automatic fallback when primary strategy fails
- **Keyword Mapping**: Extensive mapping of UI element keywords

### 2. **Robust Verification System**
- **Multiple Verification Strategies**: 5 different verification strategies with weighted combination
- **Confidence Scoring**: Each verification includes a confidence score with configurable thresholds
- **State Tracking**: Comprehensive UI state change tracking and history analysis
- **Advanced Analysis**: Semantic understanding of subgoals and expected state changes
- **Flexible Thresholds**: Configurable confidence thresholds and verification parameters

### 3. **Robust Planning System**
- **Multiple Planning Strategies**: Template-based, semantic, adaptive, and fallback planning
- **Structured Subgoals**: Detailed subgoal representation with priority and dependencies
- **Replanning Capabilities**: Dynamic replanning when subgoals fail with failure analysis
- **Plan Optimization**: Automatic plan optimization and alternative plan generation
- **Context Awareness**: Semantic understanding of goals and UI context

### 4. **Full Robust Loop Integration**
- **Complete Workflow**: Goal → Plan → Execute → Verify → Replan
- **Coordinated Agents**: Seamless integration of all three agents
- **Failure Recovery**: Automatic replanning when subgoals fail
- **Success Tracking**: Comprehensive success rate calculation
- **Performance Monitoring**: Detailed statistics across all agents

### 5. **Robust Supervisor Analysis**
- **Comprehensive Log Analysis**: Analyzes execution logs, confidence scores, and performance metrics
- **Issue Detection**: Identifies flaky behavior, repeated failures, and performance bottlenecks
- **Structured Reporting**: Generates detailed JSON and Markdown reports with actionable insights
- **Agent Performance Analysis**: Individual performance metrics for planner, executor, and verifier
- **Recommendation Engine**: Provides specific recommendations for system improvement

### 6. **Retry and Error Handling**
- **Configurable Retries**: Up to 5 retry attempts with configurable delays
- **Timeout Protection**: Action execution timeouts to prevent hanging
- **Graceful Degradation**: Continues operation even when some strategies fail
- **Detailed Error Reporting**: Comprehensive error messages and logging

### 7. **Action Validation**
- **Pre-execution Validation**: Validates actions before execution
- **Post-execution Validation**: Validates results after execution
- **Element State Verification**: Checks if elements exist in current UI
- **Safety Checks**: Prevents unsafe or invalid actions

### 8. **Comprehensive Logging**
- **Structured Logging**: JSON and text format support
- **Multiple Outputs**: Console and file logging
- **Performance Tracking**: Execution time and statistics
- **Debug Information**: Detailed debug information for troubleshooting

### 9. **Performance Monitoring**
- **Execution Statistics**: Tracks success/failure rates
- **Performance Metrics**: Average execution times
- **Retry Tracking**: Monitors retry attempts and reasons
- **Real-time Monitoring**: Live performance monitoring

### 10. **Configuration Management**
- **Preset Configurations**: Ready-to-use configurations
- **Flexible Settings**: Easy customization of all parameters
- **Runtime Switching**: Change configurations at runtime
- **Validation**: Configuration validation and error checking

## Performance Features

### Statistics Tracking
```python
# Get execution statistics
stats = executor.get_stats()
print(f"Success Rate: {stats['successful_executions'] / stats['total_executions'] * 100:.1f}%")
print(f"Average Execution Time: {stats['average_execution_time']:.2f}s")
print(f"Retry Rate: {stats['retry_attempts'] / stats['total_executions'] * 100:.1f}%")
```

### Logging Options
```python
# Console logging
logger = QALogger(log_level="INFO", enable_console=True)

# File logging
logger = QALogger(log_level="DEBUG", log_file="execution.log")

# JSON logging
logger = QALogger(log_level="INFO", enable_json=True)
```

## Element Finding Strategies

1. **Exact Text Match**: Perfect text matches
2. **Fuzzy Text Match**: Similar text using string similarity
3. **Semantic Match**: Keyword-based semantic matching
4. **Keyword Match**: Specific keyword matching
5. **Action-based Match**: Context-aware action matching

## Verification Strategies

1. **UI Change Verification**: Detects significant changes in UI tree structure
2. **Subgoal Presence Verification**: Checks if subgoal text appears in current UI
3. **State Transition Verification**: Validates expected state changes based on action type
4. **Element Interaction Verification**: Analyzes interactive elements and their relevance
5. **Semantic Verification**: Uses semantic understanding to verify subgoal completion

## Safety Features

- **Action Validation**: Prevents invalid actions
- **Timeout Protection**: Prevents hanging operations
- **Error Recovery**: Graceful error handling
- **State Verification**: Validates UI state changes
- **Retry Limits**: Prevents infinite retry loops

## Monitoring and Debugging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General execution information
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical error messages

### Performance Metrics
- Total executions
- Success/failure rates
- Average execution time
- Retry attempts
- Element finding success rates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
1. Check the documentation
2. Review the test examples
3. Open an issue on GitHub
4. Contact the development team
