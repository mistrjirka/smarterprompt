# SmarterPrompt with LangChain - Thinking Models Guide

## Quick Start with Thinking Models

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Ollama with Thinking Models
```bash
# Install and start Ollama
ollama serve

# Pull thinking models
ollama pull deepseek-r1:3b      # DeepSeek R1 (thinking model)
ollama pull deepseek-r1:7b      # Larger version
ollama pull qwen2.5:7b          # Fast model for judge

# Optional: Other models
ollama pull llama3              # General purpose
ollama pull gemma2:9b           # Google's model
```

### 3. Run the Application
```bash
python app.py
```

## Using Thinking Models

### What are Thinking Models?
Thinking models like DeepSeek R1 show their reasoning process before giving final answers. LangChain automatically:
- Captures the thinking/reasoning content
- Strips it from the final output when passing to other models
- Handles the "chain of thought" process transparently

### Recommended Setup

**For Complex Tasks (Main AI):**
- Provider: `ollama`
- Model: `deepseek-r1:3b` or `deepseek-r1:7b`
- These will show reasoning in the interface but pass clean output to judge

**For Fast Evaluation (Judge AI):**
- Provider: `ollama`
- Model: `qwen2.5:7b` or `llama3`
- Fast models work well for evaluation tasks

### Model Selection

The UI now includes:
- **Provider dropdown**: Choose between Ollama and OpenAI
- **Model dropdown**: Automatically populated with available models
- **Refresh Models button**: Updates the list from your Ollama installation

### LangGraph Workflow Benefits

1. **Automatic Reasoning**: Thinking models work transparently
2. **Clean Output**: Reasoning is stripped when passing between models  
3. **Visual Workflow**: See exactly how data flows through the system
4. **Smart Iteration**: Automatic refinement decisions based on judge feedback

## Popular Model Combinations

### 1. All Thinking (Slower but Thorough)
- **Main**: `deepseek-r1:7b` (shows reasoning)
- **Judge**: `deepseek-r1:3b` (evaluates with reasoning)

### 2. Hybrid (Recommended)
- **Main**: `deepseek-r1:3b` (thinking for main task)
- **Judge**: `qwen2.5:7b` (fast evaluation)

### 3. OpenAI Integration
- **Main**: `o1-mini` (OpenAI thinking model)
- **Judge**: `gpt-4o-mini` (fast OpenAI evaluation)

## Graph Visualization

The workflow graph shows:
- **Blue boxes**: Processing nodes (Main AI, Judge AI, etc.)
- **Decision diamonds**: Where the system decides to refine or finalize
- **Arrows**: Data flow between components

You can view the graph in three formats:
- **Image**: Visual PNG diagram
- **Mermaid**: Code for mermaid.live rendering  
- **ASCII**: Text-based representation

## Troubleshooting

### Models Not Loading?
```bash
# Check if Ollama is running
ollama list

# If not running
ollama serve

# Pull missing models
ollama pull deepseek-r1:3b
```

### No Models in Dropdown?
1. Click "Refresh Models" button
2. Check Ollama is running on localhost:11434
3. Verify models are installed with `ollama list`

### Reasoning Not Showing?
- Thinking models automatically show reasoning in the chat
- The reasoning is stripped when passed to the judge
- This is normal behavior handled by LangChain

## Advanced Configuration

### Custom Ollama URL
Set environment variable:
```bash
export OLLAMA_BASE_URL="http://your-server:11434"
```

### OpenAI Models
Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

Then select "openai" as provider and choose from available models.

## Model Performance Tips

1. **Use thinking models for main tasks** that need reasoning
2. **Use fast models for evaluation** to speed up the process
3. **Monitor the graph** to understand workflow efficiency
4. **Experiment with different combinations** for your use case

The system now handles all the complexity of thinking models automatically while giving you full control over the workflow!