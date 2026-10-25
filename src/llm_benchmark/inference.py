import time
import pandas as pd
import ollama
from typing import List, Dict, Any
from llm_benchmark.schemas import parse_and_validate_output

SYSTEM_PROMPT = (
    "You are an automated IT & Customer Support Ticket Triage AI. "
    "Analyze the support ticket and return ONLY a valid JSON object with the exact keys: "
    '"category" (Billing, Technical Bug, Account Access, Feature Request, Security & Compliance, General Inquiry), '
    '"urgency" (Low, Medium, High, Critical), and '
    '"department" (Finance, Tier-1 Support, Engineering, Security).'
)

BASE_MODEL = "llama3.2:3b"
FINETUNED_MODEL = "ticket-classifier-ft"


def load_few_shot_messages(csv_path: str = "data/fewshot_prompt_5.csv") -> List[Dict[str, str]]:
    df_fs = pd.read_csv(csv_path)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _, row in df_fs.iterrows():
        messages.append({"role": "user", "content": row["ticket_text"]})
        messages.append({"role": "assistant", "content": row["target_json"]})
    return messages


def classify_ticket(
    ticket_text: str,
    tier: str,
    few_shot_history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    tier: 'Zero-Shot Base' | 'Few-Shot Base (5-Shot)' | 'QLoRA Fine-Tuned'
    """
    if tier == "Zero-Shot Base":
        model_name = BASE_MODEL
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket_text},
        ]
    elif tier == "Few-Shot Base (5-Shot)":
        model_name = BASE_MODEL
        if few_shot_history is None:
            few_shot_history = load_few_shot_messages()
        messages = few_shot_history + [{"role": "user", "content": ticket_text}]
    elif tier == "QLoRA Fine-Tuned":
        model_name = FINETUNED_MODEL
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket_text},
        ]
    else:
        raise ValueError(f"Unknown tier: {tier}")

    start_ns = time.perf_counter_ns()
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 64 , "num_ctx": 2048, "num_threads": 6},
    )
    latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

    raw_output = response["message"]["content"]
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 1)
    tokens_per_sec = (completion_tokens / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    is_strict_json, is_schema_valid, parsed = parse_and_validate_output(raw_output)

    return {
        "tier": tier,
        "model_name": model_name,
        "raw_output": raw_output,
        "pred_category": parsed["category"],
        "pred_urgency": parsed["urgency"],
        "pred_department": parsed["department"],
        "is_strict_json": is_strict_json,
        "is_schema_valid": is_schema_valid,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }