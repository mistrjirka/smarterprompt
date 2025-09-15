# SmarterPrompt with LangChain/LangGraph

This is an updated version of SmarterPrompt that uses LangChain and LangGraph for improved workflow management and visualization.

## New Features

### 🔗 LangChain Integration
- Replaced custom providers with LangChain's `ChatOpenAI` and `ChatOllama`
- Better support for different model types and configurations
- Automatic handling of model-specific constraints

### 🧠 Thinking Models Support
- Native support for OpenAI's o1-preview and o1-mini models
- Automatic handling of reasoning/thinking content
- Special configuration for o1 models (no system messages, different parameter handling)

### 📊 Graph Visualization
- Visual representation of the review workflow using LangGraph
- Multiple visualization formats:
  - **PNG Image**: Visual graph display
  - **Mermaid Code**: Copy-paste to mermaid.live for rendering
  - **ASCII**: Text-based graph representation

### 🔄 Workflow Types
- **Original**: The classic ReviewLoop workflow
- **LangGraph**: New state-based workflow with better visualization and control

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Basic Usage
The application supports both the original workflow and the new LangGraph workflow. You can switch between them in the UI.

### Thinking Models Configuration
To use OpenAI's o1 models:

1. Set your `OPENAI_API_KEY` environment variable
2. In the UI, check "Use Thinking Models (o1)"
3. Select "openai" as provider and the model will auto-update to "o1-mini"

### Config File
Update your `config.yaml`:

```yaml
main_ai:
  provider: "openai"  # or "ollama"
  model: "o1-mini"    # for thinking models
  temperature: 0.2    # ignored for o1 models
  max_tokens: 2000
  stop: []

judge_ai:
  provider: "ollama"
  model: "qwen2.5:7b"
  temperature: 0.1
  max_tokens: 1500
  stop: []

openai:
  base_url: "https://api.openai.com/v1"
  organization: null

ollama:
  base_url: "http://localhost:11434"
```

## Key Differences

### Original vs LangGraph Workflow

**Original Workflow:**
- Manual step-by-step execution
- Linear progression: Main AI → Judge → Refine
- Manual refinement control

**LangGraph Workflow:**
- Automated state-based execution
- Conditional logic for refinement decisions
- Built-in iteration limits and logic
- Visual workflow representation

### Thinking Models Handling

The system automatically:
- Removes system messages for o1 models (not supported)
- Converts them to user message prefixes
- Handles reasoning content internally
- Strips thinking parts when passing between models

## Graph Visualization

The workflow graph shows:
- **Nodes**: Main AI, Judge AI, Refine, Finalize
- **Edges**: Flow between steps
- **Conditional Logic**: When to refine vs finalize
- **Current Position**: Where the workflow is in execution

### Visualization Formats

1. **Image**: Direct PNG display in the UI
2. **Mermaid**: Code you can copy to mermaid.live
3. **ASCII**: Simple text representation

## Usage Tips

1. **Start with LangGraph**: Try the new workflow for better automation
2. **Use Thinking Models**: For complex reasoning tasks, enable o1 models
3. **Visualize First**: Check the graph to understand the workflow
4. **Monitor Progress**: The graph shows where you are in the process

## Troubleshooting

### Import Errors
If you see import errors for LangChain packages, install dependencies:
```bash
pip install langchain langchain-openai langchain-community langgraph grandalf
```

### O1 Model Issues
- Ensure `OPENAI_API_KEY` is set
- O1 models don't support temperature or system messages
- They handle reasoning internally

### Graph Visualization
- PNG generation requires internet connection (uses mermaid.ink API)
- For offline use, use ASCII or Mermaid code formats
- Install `pyppeteer` for local PNG generation

## Development

The codebase now includes:
- `providers.py`: LangChain provider wrappers
- `orchestrator.py`: Both original and LangGraph workflows
- `app.py`: UI with workflow selection and graph visualization

## Migration from Original

Existing users can:
1. Keep using the original workflow (select "original" in UI)
2. Gradually migrate to LangGraph workflow
3. Benefit from thinking models support immediately
4. Use graph visualization to understand workflows better