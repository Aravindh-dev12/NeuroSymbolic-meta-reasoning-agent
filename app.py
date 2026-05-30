"""
app.py — Premium Gradio Dashboard for the NeuroSymbolic Meta-Reasoning Agent.
Provides an interactive chatbot, visual reasoning trace explorer, knowledge base inspector,
real-time plotly telemetry analytics, and active configurations manager.
"""
import os
import sys
import json
import time
import yaml
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr

# Add project directory to python path
sys.path.insert(0, str(Path(__file__).parent / "neurosymbolic_agent"))

from main import NeuroSymbolicAgent
from utils.config import load_config

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# CSS for a premium, glassmorphic dark-mode look
CUSTOM_CSS = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Outfit', 'Inter', sans-serif;
}
.gradio-container {
    background-color: #0b0f19 !important;
    border: none !important;
}
.sidebar {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
.glass-panel {
    background: rgba(17, 24, 39, 0.5) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.btn-primary {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
.tab-nav {
    border-bottom: 2px solid rgba(255, 255, 255, 0.08) !important;
}
.tab-nav button {
    font-weight: 600 !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
}
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(to right, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px !important;
}
h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}
.chatbot-container {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
}
.trace-step {
    border-left: 3px solid #6366f1;
    padding-left: 15px;
    margin-bottom: 15px;
}
"""

class GradioAgentWrapper:
    """Helper class to load and interface with the NeuroSymbolicAgent."""
    def __init__(self):
        self.agent = None
        self.config_path = "neurosymbolic_agent/configs/agent_config.yaml"
        self.initialize_agent()

    def initialize_agent(self):
        try:
            # Overwrite LLM keys if not set to prevent initialisation failures
            if "LLM_BACKEND" not in os.environ:
                os.environ["LLM_BACKEND"] = "anthropic"  # default
            
            self.agent = NeuroSymbolicAgent(config_path=self.config_path)
            return "🧬 Agent Initialised successfully."
        except Exception as e:
            return f"Error initialising agent: {e}"

    def update_config(self, backend: str, threshold: float, max_rounds: int, verbose: bool):
        try:
            # Read current yaml
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            data['agent']['llm_backend'] = backend.lower()
            data['agent']['confidence_threshold'] = threshold
            data['agent']['max_self_improvement_rounds'] = max_rounds
            data['agent']['verbose'] = verbose

            with open(self.config_path, 'w') as f:
                yaml.safe_dump(data, f)
            
            # Re-init agent
            self.initialize_agent()
            return "⚙️ Configuration updated & Agent re-initialised."
        except Exception as e:
            return f"Failed to update configuration: {e}"

    def run_query(self, query: str):
        if not self.agent:
            return "Agent not initialised.", [], "No trace available.", 0.0, "unknown"
        
        start_time = time.time()
        answer = self.agent.run(query)
        elapsed = time.time() - start_time
        
        # Extract metadata from last recorded trace
        trace_path = Path("logs/reasoning_traces.jsonl")
        trace_steps = []
        path_used = "hybrid"
        confidence = 0.8
        
        if trace_path.exists():
            try:
                with open(trace_path, 'r') as f:
                    lines = f.readlines()
                if lines:
                    last_trace = json.loads(lines[-1])
                    steps_raw = last_trace.get("steps", [])
                    path_used = last_trace.get("path_used", "hybrid")
                    confidence = last_trace.get("final_confidence", 0.8)
                    
                    for step in steps_raw:
                        s_name = step.get("step_name", "").upper()
                        s_in = step.get("input_summary", "")
                        s_out = step.get("output_summary", "")
                        details = step.get("metadata", {})
                        
                        trace_steps.append(
                            f"### 📍 {s_name}\n"
                            f"**Input:** {s_in}\n\n"
                            f"**Output:** {s_out}\n\n"
                            f"**Details:** {json.dumps(details) if details else 'N/A'}\n"
                            "---"
                        )
            except Exception as e:
                trace_steps = [f"Error parsing trace steps: {e}"]
        
        if not trace_steps:
            trace_steps = ["No trace details recorded for this run."]
        
        trace_html = "\n\n".join(trace_steps)
        metrics_summary = f"⏱️ Time taken: {elapsed:.2f}s | 🧠 Path: {path_used.upper()} | 🎯 Confidence: {confidence:.2%}"
        
        return answer, metrics_summary, trace_html, confidence, path_used

wrapper = GradioAgentWrapper()

# ─── Telemetry Plotting Helpers ──────────────────────────────────────────────
def get_telemetry_charts():
    telemetry_path = Path("logs/telemetry.jsonl")
    
    # Initialize dummy metrics if telemetry file doesn't exist
    if not telemetry_path.exists() or os.path.getsize(telemetry_path) == 0:
        # Create some beautiful dummy analytics for demonstration
        dummy_data = [
            {"session_id": "1", "task": "Math solving", "success": True, "duration_ms": 1200, "routing_path": "symbolic", "self_improvement_rounds": 0, "violations_count": 0, "timestamp": time.time() - 3600},
            {"session_id": "2", "task": "Sentiment analysis", "success": True, "duration_ms": 950, "routing_path": "neural", "self_improvement_rounds": 1, "violations_count": 0, "timestamp": time.time() - 3000},
            {"session_id": "3", "task": "Logic constraint puzzle", "success": True, "duration_ms": 2500, "routing_path": "hybrid", "self_improvement_rounds": 2, "violations_count": 1, "timestamp": time.time() - 2000},
            {"session_id": "4", "task": "Math equation", "success": True, "duration_ms": 1800, "routing_path": "symbolic", "self_improvement_rounds": 1, "violations_count": 0, "timestamp": time.time() - 1000},
            {"session_id": "5", "task": "NLU summarisation", "success": True, "duration_ms": 1100, "routing_path": "neural", "self_improvement_rounds": 0, "violations_count": 0, "timestamp": time.time()}
        ]
        with open(telemetry_path, "w") as f:
            for entry in dummy_data:
                f.write(json.dumps(entry) + "\n")
                
    try:
        df = pd.read_json(telemetry_path, lines=True)
    except Exception as e:
        return go.Figure(), go.Figure(), go.Figure(), f"Error loading telemetry: {e}"
        
    # Latency / Duration Chart
    fig_latency = px.line(
        df, x=df.index, y="duration_ms",
        title="⚡ Task Execution Latency (ms)",
        labels={"index": "Task Run Sequence", "duration_ms": "Latency (ms)"},
        markers=True,
        template="plotly_dark"
    )
    fig_latency.update_traces(line_color="#818cf8", marker=dict(size=8, color="#cbd5e1"))
    
    # Path Routing Distribution
    routing_counts = df["routing_path"].value_counts().reset_index()
    routing_counts.columns = ["Routing Path", "Count"]
    fig_routing = px.pie(
        routing_counts, names="Routing Path", values="Count",
        title="🔀 Meta-Controller Path Decisions",
        color="Routing Path",
        color_discrete_map={"symbolic": "#34d399", "neural": "#60a5fa", "hybrid": "#c084fc", "symbolic+llm": "#fb7185"},
        hole=0.4,
        template="plotly_dark"
    )
    
    # Self Improvement Distribution
    improvement_rounds = df["self_improvement_rounds"].value_counts().reset_index()
    improvement_rounds.columns = ["Critique Rounds", "Count"]
    improvement_rounds["Critique Rounds"] = improvement_rounds["Critique Rounds"].astype(str)
    fig_improvement = px.bar(
        improvement_rounds, x="Critique Rounds", y="Count",
        title="🔄 Self-Improvement Critique Rounds Frequency",
        color="Critique Rounds",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_dark"
    )
    
    summary_text = (
        f"📋 Total Tasks Processed: **{len(df)}**\n"
        f"⚡ Average Latency: **{df['duration_ms'].mean():.2f} ms**\n"
        f"🔄 Average Self-Improvement Rounds: **{df['self_improvement_rounds'].mean():.1f} rounds**\n"
        f"⚠️ Safety/Constitutional Violations Triggered: **{df.get('violations_count', pd.Series([0])).sum()}**\n"
    )
    
    return fig_latency, fig_routing, fig_improvement, summary_text

# ─── Knowledge Base Helpers ──────────────────────────────────────────────────
def get_kb_summary():
    if not wrapper.agent or not wrapper.agent.kb:
        return "Knowledge base not loaded."
    summary = wrapper.agent.kb.summary()
    facts_str = "\n".join(str(f) for f in wrapper.agent.kb.facts)
    rules_str = "\n".join(str(r) for r in wrapper.agent.kb.rules)
    
    details = (
        f"### 📊 Knowledge Base Stats\n"
        f"- Asserted Ground Facts: **{summary['fact_count']}**\n"
        f"- Implemented Forward Logic Rules: **{summary['rule_count']}**\n"
        f"- Active Logic Predicates: **{', '.join(summary['predicates']) if summary['predicates'] else 'None'}**\n\n"
        f"### 📍 Ground Facts Store\n"
        f"```prolog\n"
        f"{facts_str if facts_str else '/* No facts asserted */'}\n"
        f"```\n\n"
        f"### 📐 Logical Rules (Prolog-style)\n"
        f"```prolog\n"
        f"{rules_str if rules_str else '/* No rules asserted */'}\n"
        f"```"
    )
    return details

def assert_kb_fact(predicate: str, args_csv: str):
    if not wrapper.agent or not wrapper.agent.kb:
        return "Knowledge base not initialised."
    
    args = [a.strip() for a in args_csv.split(",") if a.strip()]
    if not predicate or not args:
        return "Please specify a valid predicate and arguments."
    
    try:
        wrapper.agent.kb.assert_fact(predicate, *args)
        return get_kb_summary()
    except Exception as e:
        return f"Error asserting fact: {e}"

# ─── Gradio App Interface Builder ─────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="indigo", secondary_hue="slate"), css=CUSTOM_CSS, title="NeuroSymbolic Meta-Reasoner Dashboard") as demo:
    
    gr.HTML(
        """
        <div style="text-align: center; margin-top: 15px; margin-bottom: 25px;">
            <h1>🧬 NeuroSymbolic Meta-Reasoning Agent</h1>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 5px;">
                AGI/ASI-level multi-model cognitive orchestrator running dynamic neural reasoning and formal SMT/symbolic logic.
            </p>
        </div>
        """
    )
    
    with gr.Tabs() as tabs:
        
        # ── TAB 1: Chat interface ──
        with gr.TabItem("🧬 AGI Meta-Reasoner", id="chat_tab"):
            with gr.Row():
                with gr.Column(scale=4):
                    chat_input = gr.Textbox(
                        label="💡 Input your task / math problem / logic puzzle",
                        placeholder="All mammals breathe. Whales are mammals. Do whales breathe?",
                        lines=3
                    )
                    
                    with gr.Row():
                        btn_clear = gr.Button("🗑️ Clear", variant="secondary")
                        btn_submit = gr.Button("🚀 Execute Task", variant="primary", elem_classes=["btn-primary"])
                    
                    gr.HTML("<h3 style='margin-top: 20px;'>💡 Cognitive Task Templates</h3>")
                    gr.Examples(
                        examples=[
                            ["All mammals breathe. Whales are mammals. Do whales breathe?"],
                            ["Solve the system of equations: 2*x + 3*y = 7 and 3*x - y = 5. Find x and y."],
                            ["Find the derivative of x^3 + 5*x^2 - 7*x + 12 with respect to x."],
                            ["Solve this puzzle: Alice is older than Bob. Bob is older than Charlie. Charlie is older than David. Who is the oldest?"],
                            ["Determine the sentiment of this text: The application is incredibly fast and intuitive, but the pricing is slightly high."],
                            ["Knights and Knaves logic: A says 'We are both knaves'. B says nothing. Knights always tell the truth, knaves always lie. What are A and B?"]
                        ],
                        inputs=chat_input,
                        label="Pre-loaded Reasoning Benchmarks"
                    )
                    
                with gr.Column(scale=5, elem_classes=["glass-panel"]):
                    gr.HTML("<h2>🔮 Execution Response</h2>")
                    chat_output = gr.Markdown(label="Final Synthesis Output", value="*Submit a task to see results...*")
                    metrics_box = gr.HTML(value="<div style='color: #cbd5e1;'>⏱️ Latency: 0.00s | 🧠 Routing: N/A</div>")
                    
                    with gr.Accordion("🔍 Active Reasoning Trace Explorer", open=True):
                        trace_output = gr.Markdown(value="*Step-by-step cognitive steps will appear here after execution...*")
                        
            # Event triggers
            def on_submit(query):
                answer, metrics, trace, conf, path = wrapper.run_query(query)
                return answer, metrics, trace
            
            btn_submit.click(on_submit, inputs=chat_input, outputs=[chat_output, metrics_box, trace_output])
            btn_clear.click(lambda: ("", "*Submit a task to see results...*", "⏱️ Latency: 0.00s", "*Step-by-step trace will appear here...*"), outputs=[chat_input, chat_output, metrics_box, trace_output])
            
        # ── TAB 2: Telemetry dashboard ──
        with gr.TabItem("📊 Telemetry Analytics", id="telemetry_tab"):
            gr.HTML("<h2>📊 Live System Telemetry & Performance Analytics</h2>")
            gr.HTML("<p style='color: #94a3b8;'>Monitor cognitive latencies, meta-controller path distributions, and recursive self-improvement rates.</p>")
            
            with gr.Row():
                summary_metrics = gr.Markdown(value="*Loading metrics...*")
                btn_refresh = gr.Button("🔄 Refresh Charts", variant="secondary")
            
            with gr.Row():
                plot_latency = gr.Plot(label="⚡ Durations Trend")
                plot_routing = gr.Plot(label="🔀 Path Routing Splits")
                
            with gr.Row():
                plot_improvement = gr.Plot(label="🔄 critique Rounds Frequency")
                
            # Telemetry Refresh Event
            def refresh_telemetry():
                l_fig, r_fig, i_fig, summary = get_telemetry_charts()
                return l_fig, r_fig, i_fig, summary
            
            btn_refresh.click(refresh_telemetry, outputs=[plot_latency, plot_routing, plot_improvement, summary_metrics])
            demo.load(refresh_telemetry, outputs=[plot_latency, plot_routing, plot_improvement, summary_metrics])
            
        # ── TAB 3: Knowledge Base ──
        with gr.TabItem("📚 Knowledge Base", id="kb_tab"):
            gr.HTML("<h2>📚 Dynamic Knowledge Store Inspector (Prolog Fragment)</h2>")
            gr.HTML("<p style='color: #94a3b8;'>Inspect formal asserted facts, conditional logic rules, and add new assertions in real time.</p>")
            
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["glass-panel"]):
                    gr.HTML("<h3>📥 Assert Ground Fact</h3>")
                    fact_predicate = gr.Textbox(label="Predicate Name", placeholder="is_a")
                    fact_args = gr.Textbox(label="Arguments (comma separated)", placeholder="whale, mammal")
                    btn_assert = gr.Button("Assert Fact", variant="primary")
                    
                with gr.Column(scale=7):
                    kb_view = gr.Markdown(value="*Loading knowledge base...*")
                    btn_kb_refresh = gr.Button("🔄 Refresh KB Store", variant="secondary")
                    
            btn_kb_refresh.click(get_kb_summary, outputs=kb_view)
            btn_assert.click(assert_kb_fact, inputs=[fact_predicate, fact_args], outputs=kb_view)
            demo.load(get_kb_summary, outputs=kb_view)
            
        # ── TAB 4: Configurations ──
        with gr.TabItem("⚙️ Configs & Principles", id="config_tab"):
            gr.HTML("<h2>⚙️ Cognitive Configurations & Constitutional Guardrails</h2>")
            gr.HTML("<p style='color: #94a3b8;'>Fine-tune meta-controller parameters and AI safety principles dynamically.</p>")
            
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["glass-panel"]):
                    gr.HTML("<h3>⚙️ Agent Settings</h3>")
                    backend_opt = gr.Dropdown(
                        choices=["local", "anthropic", "openai"],
                        label="LLM Routing Backend",
                        value="local"
                    )
                    threshold_opt = gr.Slider(
                        minimum=0.1, maximum=1.0, step=0.05,
                        label="Confidence Threshold (below this triggers Critique-Improvement)",
                        value=0.75
                    )
                    rounds_opt = gr.Slider(
                        minimum=1, maximum=10, step=1,
                        label="Max Self-Improvement Rounds",
                        value=3
                    )
                    verbose_opt = gr.Checkbox(label="Console Verbose Output", value=True)
                    
                    btn_save_config = gr.Button("💾 Save settings & Re-initialise", variant="primary")
                    config_status = gr.Markdown(value="*Settings unchanged.*")
                    
                with gr.Column(scale=5):
                    gr.HTML("<h3>🛡️ Constitutional AI Safety Principles Registry</h3>")
                    
                    try:
                        principles_path = "neurosymbolic_agent/configs/constitutional_principles.yaml"
                        with open(principles_path, 'r') as f:
                            p_data = yaml.safe_load(f)
                        principles_txt = yaml.safe_dump(p_data)
                    except Exception:
                        principles_txt = "Failed to load principles file."
                        
                    gr.Code(value=principles_txt, language="yaml", label="configs/constitutional_principles.yaml (Read Only)")
                    
            btn_save_config.click(
                wrapper.update_config,
                inputs=[backend_opt, threshold_opt, rounds_opt, verbose_opt],
                outputs=config_status
            )
            
            # Load initial config dropdowns
            def load_initial_config():
                try:
                    with open(wrapper.config_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                    return (
                        cfg['agent']['llm_backend'],
                        cfg['agent']['confidence_threshold'],
                        cfg['agent']['max_self_improvement_rounds'],
                        cfg['agent']['verbose']
                    )
                except Exception:
                    return "local", 0.75, 3, True
                
            demo.load(load_initial_config, outputs=[backend_opt, threshold_opt, rounds_opt, verbose_opt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
