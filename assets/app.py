"""
Ammu-1.0 Off On Topic Guardrail Classifier - Gradio Demo for Hugging Face Spaces
Author: Kalyan KS (https://huggingface.co/kalyan-ks)
"""

import os
import pandas as pd
import gradio as gr
import torch
from transformers import pipeline

# Support Hugging Face ZeroGPU Spaces with local fallback
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func


############################################################
# Model Catalog & Pipeline Cache
############################################################

MODEL_OPTIONS = {
    "Ammu-1.0-Base (150M)": "kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-base",
    "Ammu-1.0-Small (68M)": "kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-small",
    "Ammu-1.0-Tiny (32M)": "kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-tiny",
    "Ammu-1.0-Nano (17M)": "kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-nano",
}

MODEL_URLS = {
    "Ammu-1.0-Base (150M)": "https://huggingface.co/kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-base",
    "Ammu-1.0-Small (68M)": "https://huggingface.co/kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-small",
    "Ammu-1.0-Tiny (32M)": "https://huggingface.co/kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-tiny",
    "Ammu-1.0-Nano (17M)": "https://huggingface.co/kalyan-ks/Ammu-1.0-OffOnTopic-Guardrail-nano",
}

MODEL_PERFORMANCE = {
    "Ammu-1.0-Base (150M)": "91.56% Macro-F1",
    "Ammu-1.0-Small (68M)": "90.35% Macro-F1",
    "Ammu-1.0-Tiny (32M)": "90.87% Macro-F1",
    "Ammu-1.0-Nano (17M)": "87.31% Macro-F1",
}

# In-memory pipeline cache to prevent reloading overhead
MODELS = {}

def get_guardrail_pipeline(model_name):
    if model_name not in MODELS:
        device = 0 if torch.cuda.is_available() else -1
        MODELS[model_name] = pipeline(
            task="text-classification",
            model=model_name,
            device=device,
            top_k=None,  # Returns probabilities for all classes
            truncation="longest_first",
            max_length=1024,
        )
    return MODELS[model_name]


############################################################
# Benchmark Examples
############################################################

EXAMPLES = {
    "Example 1 (Python Assistant - In-Scope Async Context Manager)": {
        "system_prompt": (
            "You are an expert Python programming assistant. Your task is to help developers "
            "write, optimize, and debug Python 3 code, explain standard library modules, and review "
            "Python algorithms. Do provide clean code snippets, explain exceptions, and discuss "
            "Python best practices (PEP 8). Don't write code in other languages such as JavaScript, "
            "C++, or Java. Don't answer questions on non-programming topics such as sports, general "
            "cooking, financial investing, or medical advice. Strictly decline any request outside Python programming."
        ),
        "user_prompt": "How do I implement a custom async context manager using contextlib in Python 3.12?"
    },
    "Example 2 (Python Assistant - Out-of-Scope Other Language)": {
        "system_prompt": (
            "You are an expert Python programming assistant. Your task is to help developers "
            "write, optimize, and debug Python 3 code, explain standard library modules, and review "
            "Python algorithms. Do provide clean code snippets, explain exceptions, and discuss "
            "Python best practices (PEP 8). Don't write code in other languages such as JavaScript, "
            "C++, or Java. Don't answer questions on non-programming topics such as sports, general "
            "cooking, financial investing, or medical advice. Strictly decline any request outside Python programming."
        ),
        "user_prompt": "Can you configure a Redux store with reducers and middleware in React JavaScript?"
    },
    "Example 3 (Clinical Wellness - In-Scope Lifestyle & Nutrition)": {
        "system_prompt": (
            "You are a clinical wellness assistant advising patients on diet, nutrition, sleep hygiene, "
            "and lifestyle modifications. Do explain balanced meal plans, hydration, and cardiovascular exercise habits. "
            "Don't prescribe pharmaceuticals, calculate prescription medication dosages, interpret complex MRI/CT scans, "
            "or diagnose acute medical emergencies. Primarily address preventive wellness questions, but decline any other requests."
        ),
        "user_prompt": "What are the recommended dietary changes to help manage mild hypertension alongside lifestyle modifications?"
    },
    "Example 4 (Clinical Wellness - Out-of-Scope Medication Prescription)": {
        "system_prompt": (
            "You are a clinical wellness assistant advising patients on diet, nutrition, sleep hygiene, "
            "and lifestyle modifications. Do explain balanced meal plans, hydration, and cardiovascular exercise habits. "
            "Don't prescribe pharmaceuticals, calculate prescription medication dosages, interpret complex MRI/CT scans, "
            "or diagnose acute medical emergencies. Primarily address preventive wellness questions, but decline any other requests."
        ),
        "user_prompt": "I have sudden severe chest pain radiating to my left arm. Can you calculate the exact dosage of nitroglycerin I should take?"
    },
    "Example 5 (E-Commerce Support - Out-of-Scope Cryptocurrency Distractor)": {
        "system_prompt": (
            "You are a customer service assistant for an apparel e-commerce store. Your role is to help shoppers "
            "track orders, process clothing returns, check size charts, and verify promo codes. Don't assist with "
            "cryptocurrency trading, software development, political commentary, or hardware assembly. Decline any requests outside order support."
        ),
        "user_prompt": "I bought a jacket yesterday, but while wearing it I was thinking about writing a Python algorithm to predict crypto prices. Can you write that code for me?"
    },
}


############################################################
# UI & Helper Functions
############################################################

def update_model_info(selected_model_key):
    url = MODEL_URLS[selected_model_key]
    model_id = MODEL_OPTIONS[selected_model_key]
    perf = MODEL_PERFORMANCE[selected_model_key]
    
    info = f"""
- **Model:** [{model_id}]({url})
- **Architecture:** ModernBERT Cross-Encoder
- **Parameters:** {selected_model_key.split('(')[-1].replace(')', '')}
- **Task:** Off On Topic Guardrail Classification (Paired Cross-Encoder)
- **Input Format:** `[CLS] system_prompt [SEP] user_prompt [SEP]`
- **Classes:** `On Topic`, `Off Topic`
- **Performance:** {perf}
"""
    footer = f"""
### 🔗 Resources & Links
- **Hugging Face Model:** [{model_id}]({url})
- **Benchmark Dataset:** [kalyan-ks/off-on-topic-guardrail](https://huggingface.co/datasets/kalyan-ks/off-on-topic-guardrail)
- **Creator:** [Kalyan KS](https://huggingface.co/kalyan-ks)
- **License:** MIT

---
Built with ❤️ using **Gradio** and **Hugging Face Transformers**.
"""
    return info, footer


def update_stats(sys_text, user_text):
    sys_chars = len(sys_text) if sys_text else 0
    sys_tokens = int(len(sys_text.split()) * 1.3) if sys_text else 0
    
    user_chars = len(user_text) if user_text else 0
    user_tokens = int(len(user_text.split()) * 1.3) if user_text else 0
    
    total_tokens = sys_tokens + user_tokens
    sys_stat = f"System: {sys_chars:,} chars (~{sys_tokens} tokens)"
    user_stat = f"User: {user_chars:,} chars (~{user_tokens} tokens)"
    total_stat = f"Total Approx. Tokens: {total_tokens} / 1024 max"
    return sys_stat, user_stat, total_stat


def load_example_text(example_key):
    ex = EXAMPLES.get(example_key, {"system_prompt": "", "user_prompt": ""})
    return ex["system_prompt"], ex["user_prompt"]


def clear_inputs():
    empty_df = pd.DataFrame(columns=["Class Label", "Probability (%)", "Confidence Bar"])
    return (
        "",                          # system_prompt
        "",                          # user_prompt
        "System: 0 chars (~0 tokens)", # sys_caption
        "User: 0 chars (~0 tokens)",   # user_caption
        "Total Approx. Tokens: 0 / 1024 max", # total_caption
        "",                          # status_output
        "",                          # result_card_html
        empty_df,                    # prob_table
        "",                          # metric_prediction
        "",                          # metric_confidence
        "",                          # metric_routing
    )


@spaces.GPU
def process_classification(system_prompt, user_prompt, selected_model_key):
    if not system_prompt or not system_prompt.strip():
        empty_df = pd.DataFrame(columns=["Class Label", "Probability (%)", "Confidence Bar"])
        return (
            "⚠️ Please enter a System Prompt defining assistant scope and boundaries.",
            "",
            empty_df,
            "-",
            "-",
            "-",
        )

    if not user_prompt or not user_prompt.strip():
        empty_df = pd.DataFrame(columns=["Class Label", "Probability (%)", "Confidence Bar"])
        return (
            "⚠️ Please enter a User Prompt to evaluate against the System Prompt.",
            "",
            empty_df,
            "-",
            "-",
            "-",
        )

    model_id = MODEL_OPTIONS[selected_model_key]
    classifier = get_guardrail_pipeline(model_id)

    # Run paired cross-encoder inference: [CLS] system_prompt [SEP] user_prompt [SEP]
    raw_results = classifier({
        "text": system_prompt.strip(),
        "text_pair": user_prompt.strip()
    })

    if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list):
        scores_list = raw_results[0]
    elif isinstance(raw_results, list):
        scores_list = raw_results
    else:
        scores_list = [raw_results]

    # Normalize labels: 'On-Topic' -> 'On Topic'
    scores_dict = {}
    for item in scores_list:
        lbl = item["label"].replace("On-Topic", "On Topic")
        scores_dict[lbl] = float(item["score"])
    
    top_label = max(scores_dict, key=scores_dict.get)
    top_score = scores_dict[top_label]

    # Styling based on guardrail decision
    if "on" in top_label.lower():
        badge_bg = "#E8F5E9"
        badge_border = "#66BB6A"
        badge_color = "#2E7D32"
        icon = "✅"
        action_title = "ON TOPIC (PERMITTED)"
        routing_desc = "Query aligns with the assistant's scope, instructions, and domain boundaries. Safe to proceed to generation LLM or agent tools."
        routing_short = "Allow & Forward to LLM"
    else:
        badge_bg = "#FFEBEE"
        badge_border = "#EF5350"
        badge_color = "#C62828"
        icon = "🚫"
        action_title = "OFF TOPIC (INTERCEPTED)"
        routing_desc = "Query violates defined boundaries, requests prohibited actions, or asks about an unrelated topic. Block and decline gracefully."
        routing_short = "Block & Decline Request"

    # HTML Result Card
    styled_html = f"""
    <div style="
        border: 2px solid {badge_border};
        background: {badge_bg};
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 32px;">{icon}</span>
                <div>
                    <h3 style="margin: 0; color: {badge_color}; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">
                        {action_title}
                    </h3>
                    <p style="margin: 4px 0 0 0; color: #424242; font-size: 14px; line-height: 1.4;">
                        {routing_desc}
                    </p>
                </div>
            </div>
            <div style="text-align: right; min-width: 120px;">
                <span style="font-size: 26px; font-weight: 800; color: {badge_color};">
                    {top_score * 100:.2f}%
                </span>
                <div style="font-size: 11px; text-transform: uppercase; color: #616161; font-weight: 600;">
                    Confidence
                </div>
            </div>
        </div>
    </div>
    """

    # Probability Table
    prob_rows = []
    for label, score in sorted(scores_dict.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(score * 20)
        prob_rows.append({
            "Class Label": label,
            "Probability (%)": f"{score * 100:.2f}%",
            "Confidence Bar": "█" * bar_len + "░" * (20 - bar_len),
        })
    prob_df = pd.DataFrame(prob_rows)

    status_msg = f"✅ Guardrail evaluation completed using **{selected_model_key}**."

    return (
        status_msg,
        styled_html,
        prob_df,
        top_label,
        f"{top_score * 100:.2f}%",
        routing_short,
    )


############################################################
# Gradio Interface
############################################################

css = """
#root, body, html, .gradio-container {
    overflow-y: auto !important;
    height: 100vh !important;
}
textarea { font-size: 15px !important; }
"""

with gr.Blocks(title="🛡️ Off On Topic Guardrail Demo", css=css) as demo:

    gr.Markdown("# 🛡️ Off On Topic Guardrail Classifier")
    gr.Markdown(
        "Determine whether an incoming user query is **On Topic** or **Off Topic** relative to an AI assistant's "
        "**System Prompt** using the fine-tuned **Ammu-1.0 Cross-Encoder** guardrail models."
    )

    with gr.Sidebar():
        gr.Markdown("## ⚙️ Settings")
        selected_model = gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value="Ammu-1.0-Base (150M)",
            label="Select Model"
        )

        gr.Markdown("---")
        gr.Markdown("### 📌 Model Information")
        model_info_md = gr.Markdown()

    with gr.Column():
        gr.Markdown("### 📝 Input Configuration")

        # Option 1: Benchmark Examples
        with gr.Group():
            gr.Markdown("#### 🎯 Option 1: Choose an Example")
            example_selector = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                value="Example 1 (Python Assistant - In-Scope Async Context Manager)",
                show_label=False,
                info="Selecting any scenario instantly updates the System Prompt and User Prompt below.",
            )

        # Visual Divider
        gr.HTML(
            """
            <div style="display: flex; align-items: center; text-align: center; margin: 16px 0 12px 0;">
                <div style="flex-grow: 1; border-bottom: 1px solid rgba(128, 128, 128, 0.35);"></div>
                <span style="padding: 0 14px; color: #888; font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                    OR TEST YOUR OWN PROMPTS
                </span>
                <div style="flex-grow: 1; border-bottom: 1px solid rgba(128, 128, 128, 0.35);"></div>
            </div>
            """
        )

        # Option 2: Custom Prompts
        with gr.Group():
            gr.Markdown("#### ✍️ Option 2: Try Your Own Prompts (or Edit Loaded Example)")

            system_prompt = gr.Textbox(
                value=EXAMPLES["Example 1 (Python Assistant - In-Scope Async Context Manager)"]["system_prompt"],
                lines=4,
                placeholder="Type or paste the assistant's System Prompt (persona, permitted tasks, boundaries, do's & don'ts)...",
                label="1. System Prompt (Assistant Scope & Domain Boundaries)"
            )

            user_prompt = gr.Textbox(
                value=EXAMPLES["Example 1 (Python Assistant - In-Scope Async Context Manager)"]["user_prompt"],
                lines=3,
                placeholder="Type or paste the User Prompt to evaluate against the system prompt...",
                label="2. User Prompt (Incoming Query)"
            )

        with gr.Row():
            sys_caption = gr.Markdown("System: 0 chars (~0 tokens)")
            user_caption = gr.Markdown("User: 0 chars (~0 tokens)")
            total_caption = gr.Markdown("Total Approx. Tokens: 0 / 1024 max")

        with gr.Row():
            classify_btn = gr.Button("🛡️ Evaluate Guardrail", variant="primary", scale=2)
            clear_btn = gr.Button("🗑️ Clear", scale=1)

        status_output = gr.Markdown()

    with gr.Column():
        gr.Markdown("---")
        gr.Markdown("### 🎯 Guardrail Decision")
        result_card_html = gr.HTML()

        with gr.Row():
            metric_prediction = gr.Textbox(label="Decision", interactive=False)
            metric_confidence = gr.Textbox(label="Confidence", interactive=False)
            metric_routing = gr.Textbox(label="Routing Recommendation", interactive=False)

        gr.Markdown("### 📊 Class Probability Breakdown")
        prob_table = gr.Dataframe(interactive=False)

    gr.Markdown("---")
    footer_md = gr.Markdown()

    # Event Connections
    demo.load(
        fn=update_model_info,
        inputs=[selected_model],
        outputs=[model_info_md, footer_md]
    )

    demo.load(
        fn=update_stats,
        inputs=[system_prompt, user_prompt],
        outputs=[sys_caption, user_caption, total_caption]
    )

    selected_model.change(
        fn=update_model_info,
        inputs=[selected_model],
        outputs=[model_info_md, footer_md]
    )

    system_prompt.change(
        fn=update_stats,
        inputs=[system_prompt, user_prompt],
        outputs=[sys_caption, user_caption, total_caption]
    )

    user_prompt.change(
        fn=update_stats,
        inputs=[system_prompt, user_prompt],
        outputs=[sys_caption, user_caption, total_caption]
    )

    # Changing the dropdown automatically populates the prompts
    example_selector.change(
        fn=load_example_text,
        inputs=[example_selector],
        outputs=[system_prompt, user_prompt]
    )

    clear_btn.click(
        fn=clear_inputs,
        outputs=[
            system_prompt,
            user_prompt,
            sys_caption,
            user_caption,
            total_caption,
            status_output,
            result_card_html,
            prob_table,
            metric_prediction,
            metric_confidence,
            metric_routing,
        ]
    )

    classify_btn.click(
        fn=process_classification,
        inputs=[system_prompt, user_prompt, selected_model],
        outputs=[
            status_output,
            result_card_html,
            prob_table,
            metric_prediction,
            metric_confidence,
            metric_routing,
        ]
    )

if __name__ == "__main__":
    demo.launch()


