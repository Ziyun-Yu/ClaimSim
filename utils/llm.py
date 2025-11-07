import os
import warnings
import json
import pandas as pd
import torch
# from dashscope import Generation  # 核心库
# import dashscope
from together import Together
from openai import OpenAI
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import warnings
from openai import OpenAI
from together import Together

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"

client_openai = OpenAI()
client_together = Together()


def load_hyperparams(config_path: Optional[Path] = None) -> Dict[str, object]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_params(hyperparams: Dict[str, object]) -> Dict[str, object]:
    return {
        "temperature": hyperparams.get("temperature", 0.7),
        "n": hyperparams.get("n", 1),
        "top_p": hyperparams.get("top_p", 1.0),
        "max_tokens": hyperparams.get("max_tokens", 1024),
        "presence_penalty": hyperparams.get("presence_penalty", 0.0),
        "frequency_penalty": hyperparams.get("frequency_penalty", 0.0),
        "logit_bias": hyperparams.get("logit_bias", {}),
        "timeout": hyperparams.get("timeout", 60),
    }


def qwen(
    prompt: str,
    stop: Optional[Sequence[str]] = None,
    hyperparams: Optional[Dict[str, object]] = None,
) -> str:
    params = _extract_params(hyperparams or load_hyperparams())

    messages = [{"role": "user", "content": prompt}]
    response = client_together.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        messages=messages,
        temperature=params["temperature"],
        top_p=params["top_p"],
        n=params["n"],
        max_tokens=params["max_tokens"],
        presence_penalty=params["presence_penalty"],
        frequency_penalty=params["frequency_penalty"],
        logit_bias=params["logit_bias"],
        stop=stop,
        stream=False,
        timeout=params["timeout"],
    )
    return response.choices[0].message.content


def llama(
    prompt: str,
    stop: Optional[Sequence[str]] = None,
    hyperparams: Optional[Dict[str, object]] = None,
) -> str:
    params = _extract_params(hyperparams or load_hyperparams())

    messages = [{"role": "user", "content": prompt}]
    response = client_together.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        messages=messages,
        temperature=params["temperature"],
        top_p=params["top_p"],
        n=params["n"],
        max_tokens=params["max_tokens"],
        presence_penalty=params["presence_penalty"],
        frequency_penalty=params["frequency_penalty"],
        logit_bias=params["logit_bias"],
        stop=stop,
        stream=False,
        timeout=params["timeout"],
    )
    return response.choices[0].message.content


def gpt(
    prompt: str,
    stop: Optional[Sequence[str]] = None,
    hyperparams: Optional[Dict[str, object]] = None,
) -> str:
    params = _extract_params(hyperparams or load_hyperparams())

    messages = [{"role": "user", "content": prompt}]
    response_stream = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=params["temperature"],
        top_p=params["top_p"],
        n=params["n"],
        max_tokens=params["max_tokens"],
        presence_penalty=params["presence_penalty"],
        frequency_penalty=params["frequency_penalty"],
        logit_bias=params["logit_bias"],
        stop=stop,
        stream=True,
        timeout=params["timeout"],
    )

    full_response = ""
    for chunk in response_stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            full_response += delta.content

    return full_response


__all__ = ["load_hyperparams", "qwen", "llama", "gpt"]