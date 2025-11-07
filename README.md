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
├── config_generation.json  # LLM hyperparameters for claim generation
└── README.md               # This file
```

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Ziyun-Yu/ClaimSim.git
cd ClaimSim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### ClaimSim Pipeline

Run the full ClaimSim pipeline with claim generation and evaluation:

```bash
python ClaimSim.py \
    --domain <domain_name> \
    --qlist <question_ids> \
    --model <model_name>
```

**Arguments:**
- `--domain`: Survey domain (e.g., `gender`, `religion`, `income`)
- `--qlist`: Comma-separated list of question IDs to process
- `--model`: LLM model to use (e.g., `gpt-4`, `qwen`, `llama`)

**Example:**
```bash
python ClaimSim.py --domain gender --qlist 1,2,3 --model gpt-4o-mini
```

### Chain-of-Thought (COT) Prompting

Run the COT baseline:

```bash
python cot.py \
    --domain <domain_name> \
    --qlist <question_ids> \
    --model <model_name>
```

### Direct Prompting

Run the direct prompting baseline:

```bash
python direct_prompting.py \
    --domain <domain_name> \
    --qlist <question_ids> \
    --model <model_name>
```

## Configuration

### LLM Hyperparameters

Edit `config.json` for evaluation parameters:
```json
{
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
    ...
}
```

### Prompt Templates

Prompt templates are stored in the `prompts/` directory.

## Contact

For questions or issues, please open an issue on GitHub or contact zy2478@nyu.edu.

