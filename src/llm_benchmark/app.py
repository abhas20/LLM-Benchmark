import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import f1_score, confusion_matrix
from llm_benchmark.inference import classify_ticket, load_few_shot_messages

st.set_page_config(page_title="LLM Ticket Triage Benchmark", layout="wide")
st.title("🎫 Open-Weight LLM Benchmark: Base vs. QLoRA Fine-Tuned (Llama 3.2 3B)")
st.caption("Evaluating Zero-Shot Base, 5-Shot Base, and Unsloth QLoRA Fine-Tuned models locally via Ollama")

RESULTS_PATH = "results/benchmark_metrics.csv"

if not os.path.exists(RESULTS_PATH):
    st.warning("Run `python -m src.evaluate` first to generate `results/benchmark_metrics.csv`!")
    st.stop()

df = pd.read_csv(RESULTS_PATH)
TIERS = ["Zero-Shot Base", "Few-Shot Base (5-Shot)", "QLoRA Fine-Tuned"]


def compute_tier_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        sub = dataframe[dataframe["tier"] == tier]
        if len(sub) == 0:
            continue
        rows.append({
            "Tier": tier,
            "Category Macro F1 (%)": round(f1_score(sub["true_category"], sub["pred_category"], average="macro", zero_division=0) * 100, 2),
            "Urgency Macro F1 (%)": round(f1_score(sub["true_urgency"], sub["pred_urgency"], average="macro", zero_division=0) * 100, 2),
            "Department Macro F1 (%)": round(f1_score(sub["true_department"], sub["pred_department"], average="macro", zero_division=0) * 100, 2),
            "Exact Match All 3 (%)": round(sub["exact_match_all_3"].mean() * 100, 2),
            "Strict JSON Rate (%)": round(sub["is_strict_json"].mean() * 100, 2),
            "Hallucination Rate (%)": round((1.0 - sub["is_schema_valid"].mean()) * 100, 2),
            "Avg Latency (ms)": round(sub["latency_ms"].mean(), 1),
            "Avg Prompt Tokens": round(sub["prompt_tokens"].mean(), 0),
        })
    return pd.DataFrame(rows)


summary_df = compute_tier_summary(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Executive Scorecard & Diversity Breakdown",
    "🧩 2. Confusion Matrices",
    "🔍 3. Edge-Case Error Inspector",
    "⚡ 4. Live Ollama Arena",
])

with tab1:
    ft_row = summary_df[summary_df["Tier"] == "QLoRA Fine-Tuned"].iloc[0]
    zs_row = summary_df[summary_df["Tier"] == "Zero-Shot Base"].iloc[0]
    fs_row = summary_df[summary_df["Tier"] == "Few-Shot Base (5-Shot)"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fine-Tuned Category F1", f"{ft_row['Category Macro F1 (%)']}%", f"{ft_row['Category Macro F1 (%)'] - zs_row['Category Macro F1 (%)']:+.1f}% vs Zero-Shot")
    c2.metric("Exact 3-Field Match", f"{ft_row['Exact Match All 3 (%)']}%", f"{ft_row['Exact Match All 3 (%)'] - zs_row['Exact Match All 3 (%)']:+.1f}% vs Zero-Shot")
    c3.metric("Strict JSON Compliance", f"{ft_row['Strict JSON Rate (%)']}%", f"{ft_row['Strict JSON Rate (%)'] - zs_row['Strict JSON Rate (%)']:+.1f}% vs Zero-Shot")
    c4.metric("Avg Inference Latency", f"{ft_row['Avg Latency (ms)']} ms", f"{ft_row['Avg Latency (ms)'] - fs_row['Avg Latency (ms)']:+.1f} ms vs Few-Shot", delta_color="inverse")

    st.subheader("Summary Benchmark Table (200 Unseen Test Tickets)")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        melted = summary_df.melt(
            id_vars="Tier",
            value_vars=["Category Macro F1 (%)", "Urgency Macro F1 (%)", "Exact Match All 3 (%)", "Strict JSON Rate (%)"],
            var_name="Metric",
            value_name="Score (%)",
        )
        fig_bar = px.bar(melted, x="Metric", y="Score (%)", color="Tier", barmode="group", title="Classification Quality & Format Compliance (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        fig_scatter = px.scatter(
            summary_df,
            x="Avg Latency (ms)",
            y="Exact Match All 3 (%)",
            size="Avg Prompt Tokens",
            color="Tier",
            text="Tier",
            title="Speed vs. Accuracy Trade-Off (Bubble Size = Prompt Tokens)",
            size_max=55,
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Accuracy Breakdown by Diversity / Challenge Type")
    chal_df = (
        df.groupby(["challenge_type", "tier"])["exact_match_all_3"]
        .mean()
        .reset_index()
    )
    chal_df["Exact Match (%)"] = (chal_df["exact_match_all_3"] * 100).round(1)
    fig_chal = px.bar(
        chal_df,
        x="challenge_type",
        y="Exact Match (%)",
        color="tier",
        barmode="group",
        title="How Each Model Handles Edge Cases (Cross-Category, Hinglish, Urgency Mismatch, Adversarial)",
    )
    st.plotly_chart(fig_chal, use_container_width=True)

with tab2:
    target_field = st.selectbox("Select Target Dimension:", ["category", "urgency", "department"])
    true_col = f"true_{target_field}"
    pred_col = f"pred_{target_field}"
    labels = sorted(df[true_col].unique().tolist())

    cm_cols = st.columns(3)
    for idx, tier in enumerate(TIERS):
        sub = df[df["tier"] == tier]
        cm = confusion_matrix(sub[true_col], sub[pred_col], labels=labels)
        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"{tier} ({target_field.title()})",
            labels={"x": "Predicted", "y": "Ground Truth"},
        )
        cm_cols[idx].plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.subheader("Inspect Tickets Where Base Model Failed & Fine-Tuned Succeeded")
    challenge_filter = st.multiselect(
        "Filter by Challenge Type:",
        options=df["challenge_type"].unique().tolist(),
        default=df["challenge_type"].unique().tolist(),
    )
    piv = df[df["challenge_type"].isin(challenge_filter)].pivot(
        index=["ticket_id", "challenge_type", "ticket_text", "true_category", "true_urgency", "true_department"],
        columns="tier",
        values=["raw_output", "exact_match_all_3", "latency_ms"],
    )
    piv.columns = [f"{col[1]} - {col[0]}" for col in piv.columns]
    piv = piv.reset_index()

    only_wins = st.checkbox("Show ONLY tickets where Zero-Shot Base failed and QLoRA Fine-Tuned succeeded", value=True)
    if only_wins:
        piv = piv[(piv["Zero-Shot Base - exact_match_all_3"] == False) & (piv["QLoRA Fine-Tuned - exact_match_all_3"] == True)]

    st.write(f"Showing **{len(piv)}** tickets:")
    st.dataframe(piv, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Live Side-by-Side Ollama Inference")
    default_ticket = (
        "URGENT 911 EMERGENCY!! 🚨 When I click the 'Download Billing Invoice #INV-2026-8821' button "
        "in Chrome, the tab crashes with TypeError: undefined. My credit card charge is fine, "
        "I need Engineering to fix the broken UI button!"
    )
    user_ticket = st.text_area("Enter a custom support ticket:", value=default_ticket, height=110)
    if st.button("🚀 Run All 3 Models Live"):
        fs_msgs = load_few_shot_messages("data/fewshot_prompt_5.csv")
        live_cols = st.columns(3)
        for idx, tier in enumerate(TIERS):
            with live_cols[idx]:
                with st.spinner(f"Running {tier}..."):
                    res = classify_ticket(user_ticket, tier=tier, few_shot_history=fs_msgs)
                st.markdown(f"### {tier}")
                st.code(res["raw_output"], language="json")
                st.write(f"**Strict JSON:** {'✅ Yes' if res['is_strict_json'] else '❌ No'}")
                st.write(f"**Schema Valid:** {'✅ Yes' if res['is_schema_valid'] else '❌ Hallucinated'}")
                st.write(f"**Latency:** `{res['latency_ms']} ms` | **Prompt Tokens:** `{res['prompt_tokens']}` | **Speed:** `{res['tokens_per_sec']} tok/s`")