# Weco Copilot Tools Instructions

## Overview

This document provides comprehensive instructions for using the Weco AI-driven code optimization tools within GitHub Copilot or other AI assistant environments. Weco systematically optimizes code using evaluation-based feedback to improve performance across various domains including GPU kernels, ML models, and prompt engineering.

## Quick Start

### Prerequisites

1. **Install Weco**: `pip install weco>=0.2.18`
2. **Set API Keys**: At least one LLM provider API key is required:
   ```bash
   export OPENAI_API_KEY="your_key_here"        # For GPT models
   export ANTHROPIC_API_KEY="your_key_here"     # For Claude models  
   export GEMINI_API_KEY="your_key_here"        # For Gemini models (free tier available)
   ```

### Basic Usage Patterns

#### 1. Interactive Setup (Recommended for Beginners)
```
weco_interactive_setup(project_path=".")
```
- Launches AI-powered analysis of your codebase
- Provides tailored optimization suggestions  
- Guides you through complete setup process
- Generates evaluation scripts automatically

#### 2. Direct Optimization (Advanced Users)
```
weco_run_optimization(
    source="optimize.py",
    eval_command="python evaluate.py --device cpu", 
    metric="speedup",
    goal="maximize",
    steps=15
)
```

#### 3. Start with Examples
```
weco_create_example(example_type="hello-kernel-world")
# Then navigate to the created directory and run optimization
```

## Tool Reference

### Core Optimization Tools

#### `weco_run_optimization`
**Purpose**: Execute AI-driven iterative code optimization

**Key Parameters**:
- `source`: File to optimize (e.g., "model.py", "kernel.cu")  
- `eval_command`: Command that evaluates performance and prints target metric
- `metric`: Performance metric name (speedup, accuracy, loss, etc.)
- `goal`: "maximize" or "minimize" the metric
- `steps`: Number of optimization iterations (default: 100)
- `additional_instructions`: Natural language guidance for optimization

**Critical Requirements**:
- Evaluation command MUST print the target metric value to stdout/stderr
- Use version control (Git) as Weco modifies source files directly
- Ensure evaluation script is robust and handles code changes gracefully

**Example Workflows**:

*GPU Kernel Optimization*:
```python
weco_run_optimization(
    source="gpu_kernel.py",
    eval_command="python benchmark.py --kernel gpu_kernel.py --device cuda",
    metric="throughput", 
    goal="maximize",
    steps=50,
    additional_instructions="Optimize memory access patterns and reduce bank conflicts"
)
```

*ML Model Optimization*:
```python  
weco_run_optimization(
    source="model.py",
    eval_command="python train_and_evaluate.py",
    metric="validation_accuracy",
    goal="maximize", 
    steps=30,
    additional_instructions="Focus on architecture improvements while maintaining model interpretability"
)
```

*Prompt Engineering*:
```python
weco_run_optimization(
    source="prompts.py", 
    eval_command="python eval_prompts.py",
    metric="success_rate",
    goal="maximize",
    steps=20,
    additional_instructions="Improve clarity and reduce ambiguity in prompt formulation"
)
```

#### `weco_analyze_codebase`
**Purpose**: AI-powered analysis to identify optimization opportunities

**Use Cases**:
- Discover performance bottlenecks automatically
- Get specific optimization recommendations  
- Understand codebase structure for optimization planning

**Example**:
```python
analysis = weco_analyze_codebase(
    project_path="./my_ml_project",
    focus_area="ml_models"
)
# Returns structured analysis with optimization suggestions
```

#### `weco_generate_evaluation`  
**Purpose**: Generate evaluation scripts and identify metrics automatically

**Benefits**:
- Creates proper evaluation harnesses
- Identifies relevant performance metrics
- Ensures compatibility with optimization workflow

**Example**:
```python
weco_generate_evaluation(
    project_path=".",
    optimization_type="gpu_kernel", 
    target_file="conv2d.py"
)
```

### Utility Tools

#### `weco_interactive_setup`
**Purpose**: Guided, AI-assisted project setup

**When to Use**:
- First-time users learning Weco
- Complex projects needing analysis
- When unsure about optimization strategy

#### `weco_authenticate`  
**Purpose**: Manage Weco service authentication for enhanced features

**Actions**:
- `login`: Secure device authentication flow
- `logout`: Clear stored credentials  
- `status`: Check current authentication state

**Benefits of Authentication**:
- Save optimization runs to dashboard
- Track progress across sessions
- Enhanced support and analytics

#### `weco_check_status`
**Purpose**: Monitor optimization progress and retrieve results

**Example**:
```python
status = weco_check_status(run_id="abc123-def456")
# Returns current progress, metrics, and completion status
```

#### `weco_list_models`
**Purpose**: View available LLM models and their requirements

**Supported Providers**:
- **OpenAI**: gpt-5, o3-pro, o4-mini, gpt-4o (most capable, higher cost)
- **Anthropic**: claude-sonnet-4-0, claude-3.5-sonnet (balanced performance)  
- **Google**: gemini-2.5-pro (free tier available, good for experimentation)

#### `weco_create_example`
**Purpose**: Create template projects for learning and experimentation

**Available Examples**:
- `hello-kernel-world`: Simple PyTorch optimization
- `cuda`: CUDA kernel optimization with custom kernels
- `triton`: Triton GPU kernel examples
- `spaceship-titanic`: Complete ML pipeline optimization
- `prompt`: LLM prompt engineering examples

## Best Practices

### Evaluation Script Requirements

Your evaluation command is critical for successful optimization. It must:

1. **Print Target Metric**: Clearly output the metric name and value
   ```python
   print(f"speedup: {speedup_value}")
   # or
   print(f"Final validation accuracy = {accuracy}")
   ```

2. **Handle Code Changes**: Be robust to modifications in the source file
3. **Exit Cleanly**: Return appropriate exit codes (0 for success)
4. **Be Deterministic**: Minimize randomness for consistent comparisons

### Optimization Guidelines

1. **Start Small**: Begin with 10-15 steps to test the pipeline
2. **Version Control**: Always use Git to track changes
3. **Clear Instructions**: Provide specific optimization guidance
4. **Appropriate Metrics**: Choose metrics that align with your goals
5. **Timeout Setting**: Set reasonable evaluation timeouts for long-running processes

### Performance Expectations

Based on the AIDE algorithm research:
- **Simple optimizations**: Improvements often visible within 10-20 steps
- **Complex problems**: May require 50-100+ steps for significant gains  
- **Research-level tasks**: Can take hundreds of steps (hours of compute)

The algorithm shows progressive improvement over time, often surpassing human expert performance on challenging benchmarks with sufficient compute budget.

### Model Selection Strategy

**For experimentation/learning**:
- Use `gemini-2.5-pro` (free tier)
- Good performance for most optimization tasks

**For production/important projects**:
- Use `gpt-4o` or `o3-pro` (highest capability)
- Better reasoning and code generation quality

**For balanced cost/performance**:
- Use `claude-3.5-sonnet` or `o4-mini`
- Good optimization results at moderate cost

## Common Workflows

### Workflow 1: GPU Kernel Optimization

1. **Create or identify kernel code** that needs optimization
2. **Analyze**: `weco_analyze_codebase(focus_area="gpu_kernels")`
3. **Generate evaluation**: `weco_generate_evaluation(optimization_type="gpu_kernel")`
4. **Optimize**: `weco_run_optimization()` with performance metrics
5. **Monitor**: Use `weco_check_status()` for long-running optimizations

### Workflow 2: ML Model Development

1. **Start with baseline model** and training script
2. **Interactive setup**: `weco_interactive_setup()` for guided analysis
3. **Focus on metrics**: accuracy, AUC, F1-score, etc.
4. **Iterative improvement**: Run optimization with validation-based evaluation
5. **Track experiments**: Use authentication for run history

### Workflow 3: Prompt Engineering

1. **Define prompt templates** and evaluation criteria
2. **Create evaluation harness** that tests prompt effectiveness
3. **Optimize prompts**: Focus on success rate, relevance, format adherence
4. **A/B testing**: Compare optimized vs. original prompts

## Troubleshooting

### Common Issues

**Evaluation fails**:
- Check that eval command runs independently
- Verify metric is printed to stdout/stderr
- Ensure dependencies are installed

**No improvements found**:
- Increase number of steps
- Provide more specific optimization instructions
- Check if evaluation metric is appropriate
- Verify evaluation script isn't too restrictive

**API errors**:
- Check API key validity and quotas
- Verify internet connection
- Try different model if one provider is down

**Authentication issues**:
- Use `weco_authenticate(action="logout")` then re-login
- Check browser for device authentication flow
- Verify account permissions

### Getting Help

1. **Check logs**: Review `.runs/` directory for detailed execution logs
2. **Enable detailed logging**: Use `save_logs=True` in optimization calls
3. **Community**: Visit GitHub issues for common problems
4. **Documentation**: Check https://docs.weco.ai/ for latest updates

## Advanced Features

### Custom Evaluation Environments

For complex evaluation setups:
```python
weco_run_optimization(
    source="distributed_model.py",
    eval_command="mpirun -np 4 python eval_distributed.py",
    eval_timeout=3600,  # 1 hour timeout
    metric="throughput",
    goal="maximize"
)
```

### Multi-Metric Optimization

While Weco optimizes single metrics, you can create composite metrics:
```python
# In your evaluation script
composite_score = 0.7 * accuracy + 0.3 * (1.0 / latency)
print(f"composite_score: {composite_score}")
```

### Integration with CI/CD

Weco can be integrated into automated workflows:
```yaml
# GitHub Actions example
- name: Optimize model
  run: |
    weco_run_optimization(
      source="model.py", 
      eval_command="python benchmark.py",
      metric="throughput",
      goal="maximize", 
      steps=20
    )
```

## Examples and Templates

### Example 1: Simple PyTorch Optimization
```python
# Create example project
weco_create_example(example_type="hello-kernel-world")

# Navigate to created directory
# Run optimization
weco_run_optimization(
    source="optimize.py",
    eval_command="python evaluate.py --solution-path optimize.py --device cpu",
    metric="speedup",
    goal="maximize", 
    steps=15,
    additional_instructions="Fuse operations while maintaining numerical accuracy"
)
```

### Example 2: Interactive ML Project Setup
```python
# Start with analysis and guided setup
weco_interactive_setup(project_path="./ml_project")

# The chatbot will:
# - Analyze your codebase
# - Suggest optimization opportunities  
# - Generate evaluation scripts
# - Configure optimization parameters
# - Provide exact commands to run
```

### Example 3: CUDA Kernel Development
```python
# Create CUDA example
weco_create_example(example_type="cuda", output_dir="./cuda_project")

# Optimize CUDA kernel
weco_run_optimization(
    source="kernel.cu",
    eval_command="nvcc kernel.cu -o kernel && ./kernel",
    metric="gflops", 
    goal="maximize",
    model="gpt-4o",  # Use powerful model for CUDA
    additional_instructions="Optimize memory coalescing and shared memory usage"
)
```

This comprehensive toolset enables systematic, AI-driven code optimization across diverse domains while maintaining flexibility for different project needs and complexity levels.