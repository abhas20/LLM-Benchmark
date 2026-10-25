import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from llm_benchmark.inference import classify_ticket, load_few_shot_messages

TIERS = ["Zero-Shot Base", "Few-Shot Base (5-Shot)", "QLoRA Fine-Tuned"]


def run_benchmark(
    test_csv: str = "data/benchmark_test_200.csv",
    out_csv: str = "results/benchmark_metrics.csv",
):
    os.makedirs("results", exist_ok=True)
    df_test = pd.read_csv(test_csv)
    few_shot_msgs = load_few_shot_messages("data/fewshot_prompt_5.csv")

    # Warm up both Ollama models in memory before timing
    print("Warming up Ollama models in VRAM/RAM...")
    classify_ticket("Warmup test ticket.", "Zero-Shot Base")
    classify_ticket("Warmup test ticket.", "QLoRA Fine-Tuned")

    records = []
    for tier in TIERS:
        print(f"\nEvaluating Tier: {tier} ({len(df_test)} tickets)...")
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            res = classify_ticket(row["ticket_text"], tier=tier, few_shot_history=few_shot_msgs)
            records.append({
                "ticket_id": row["ticket_id"],
                "ticket_text": row["ticket_text"],
                "challenge_type": row["challenge_type"],
                "channel": row["channel"],
                "true_category": row["category"],
                "true_urgency": row["urgency"],
                "true_department": row["department"],
                **res,
                "category_correct": res["pred_category"] == row["category"],
                "urgency_correct": res["pred_urgency"] == row["urgency"],
                "department_correct": res["pred_department"] == row["department"],
                "exact_match_all_3": (
                    res["pred_category"] == row["category"]
                    and res["pred_urgency"] == row["urgency"]
                    and res["pred_department"] == row["department"]
                ),
            })

    df_results = pd.DataFrame(records)
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved full benchmark results to {out_csv}!")

    # Print Terminal Summary Table
    print("\n================ BENCHMARK SUMMARY ================")
    for tier in TIERS:
        sub = df_results[df_results["tier"] == tier]
        cat_f1 = f1_score(sub["true_category"], sub["pred_category"], average="macro", zero_division=0) * 100
        urg_f1 = f1_score(sub["true_urgency"], sub["pred_urgency"], average="macro", zero_division=0) * 100
        exact_acc = sub["exact_match_all_3"].mean() * 100
        json_rate = sub["is_strict_json"].mean() * 100
        schema_rate = sub["is_schema_valid"].mean() * 100
        avg_lat = sub["latency_ms"].mean()
        avg_tok = sub["prompt_tokens"].mean()
        print(
            f"{tier:24s} | Cat Macro-F1: {cat_f1:5.1f}% | Urg F1: {urg_f1:5.1f}% | "
            f"Exact 3/3: {exact_acc:5.1f}% | Strict JSON: {json_rate:5.1f}% | "
            f"Schema Valid: {schema_rate:5.1f}% | Latency: {avg_lat:6.1f} ms | Prompt Toks: {avg_tok:5.0f}"
        )


if __name__ == "__main__":
    run_benchmark()