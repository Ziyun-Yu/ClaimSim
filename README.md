# ClaimSim

This repository contains the implementation of **ClaimSim** (Diverse Claims Generation), a method for simulating user responses in surveys using Large Language Models (LLMs).

## Paper

**Title:** An Analysis of Large Language Models for Simulating User Responses in Surveys

**Abstract:**

Using Large Language Models (LLMs) to simulate user opinions has received growing attention. Yet LLMs, especially trained with reinforcement learning from human feedback (RLHF), are known to exhibit biases toward dominant viewpoints, raising concerns about their ability to represent users from diverse demographic and cultural backgrounds.

In this work, we examine the extent to which LLMs can simulate human responses to cross-domain survey questions and propose two LLM-based approaches: chain-of-thought (COT) prompting and Diverse Claims Generation (CLAIMSIM), which elicits viewpoints from LLM parametric knowledge as contextual input. Experiments on the survey question answering task indicate that, while CLAIMSIM produces more diverse responses, both approaches struggle to accurately simulate users. Further analysis reveals two key limitations: (1) LLMs tend to maintain fixed viewpoints across varying demographic features, and generate single-perspective claims; and (2) when presented with conflicting claims, LLMs struggle to reason over nuanced differences among demographic features, limiting their ability to adapt responses to specific user profiles.

## Repository Structure

```
ClaimSim/
├── ClaimSim.py              # Main pipeline for ClaimSim approach
├── cot.py                   # Chain-of-Thought (COT) prompting implementation
├── direct_prompting.py      # Direct prompting baseline
├── utils/
│   ├── __init__.py
│   └── llm.py              # LLM interface utilities (GPT, Qwen, Llama)
├── prompts/                 # Prompt templates (Jinja2)
├── config.json             # LLM hyperparameters for evaluation
generation
└── README.md               # This file
```

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Ziyun-Yu/ClaimSim.git
cd ClaimSim
```

2. Set up your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export TOGETHER_API_KEY="your-together-api-key"  # If using Together AI models
```

## Usage

### Data Preparation

Before running any experiments, ensure you have the following data files in the `data/` directory:
- `question_config.json`: Question definitions and instructions
- `question_binary_config.json`: Binary label mappings for evaluation
- `sampled_100_rows.csv`: Survey response data
- `demographic_info.json`: Demographic metadata (for ClaimSim only)

### ClaimSim Pipeline

Run the full ClaimSim pipeline with claim generation and evaluation:

```bash
python ClaimSim.py \
    --domain <domain_name> \
    --questions <question_ids> \
    [--question-config <path>] \
    [--binary-config <path>] \
    [--data-file <path>] \
    [--demographic-info <path>] \
    [--template-dir <path>] \
    [--output-dir <path>]
```

**Key Arguments:**
- `--domain`: Survey domain (default: `gender`). Options: `gender`, `religion`, `income`, etc.
- `--questions`: Space-separated question IDs to process (default: all questions in domain)
- `--question-config`: Path to question configuration JSON (default: `data/question_config.json`)
- `--binary-config`: Path to binary label configuration (default: `data/question_binary_config.json`)
- `--data-file`: CSV file with survey responses (default: `data/sampled_100_rows.csv`)
- `--demographic-info`: JSON with demographic metadata (default: `data/demographic_info_less.json`)
- `--template-dir`: Directory with Jinja2 templates (default: `prompts/`)
- `--output-dir`: Output directory for results (default: `outputs/`)

**Example:**
```bash
python ClaimSim.py --domain gender --questions Q1 Q2 Q3
```

### Chain-of-Thought (COT) Prompting

Run the COT baseline:

```bash
python cot.py \
    --domain <domain_name> \
    --questions <question_ids> \
    [--question-config <path>] \
    [--binary-config <path>] \
    [--data-file <path>] \
    [--template-dir <path>] \
    [--output-dir <path>] \
    [--max-workers <num>]
```

**Key Arguments:**
- `--domain`: Survey domain (default: `gender`)
- `--questions`: Space-separated question IDs (default: all questions in domain)
- `--template-dir`: Directory with COT templates (default: `baseline/`)
- `--output-dir`: Output directory (default: `outputs/cot/`)
- `--max-workers`: Thread pool size for parallel processing (default: 10)

**Example:**
```bash
python cot.py --domain religion --questions Q1 Q2 --max-workers 4
```

### Direct Prompting

Run the direct prompting baseline:

```bash
python direct_prompting.py \
    --domain <domain_name> \
    --questions <question_ids> \
    [--question-config <path>] \
    [--binary-config <path>] \
    [--data-file <path>] \
    [--template-dir <path>] \
    [--template-name <name>] \
    [--output-dir <path>] \
    [--max-workers <num>]
```

**Key Arguments:**
- `--domain`: Survey domain (default: `gender`)
- `--questions`: Space-separated question IDs (default: all questions in domain)
- `--template-dir`: Directory with prompt templates (default: `prompts/`)
- `--template-name`: Template filename (default: `direct_prompting_template.md`)
- `--output-dir`: Output directory (default: `outputs/direct_prompting/`)
- `--max-workers`: Thread pool size for parallel processing (default: 4)

**Example:**
```bash
python direct_prompting.py --domain income --questions Q1 Q2 Q3 --max-workers 8
```

## Configuration

### LLM Hyperparameters

Edit `config.json` for evaluation parameters:
```json
{
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 1.0,
    ...
}
```

### Prompt Templates

Prompt templates are stored in the `prompts/`. You can customize them for different survey domains or question types.


## Contact

For questions or issues, please open an issue on GitHub or contact [zy2478@nyu.edu].