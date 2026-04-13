"""Gradio web UI for the FPL Transfer Strategist."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the import path (needed when running as `python app.py`
# without an editable install, e.g. on HuggingFace Spaces).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from gradio import ChatMessage

from fpl_strategist.data.fpl_client import FPLClient
from fpl_strategist.graph import graph


# -- Node-to-card formatting helpers ------------------------------------------

NODE_TITLES = {
    "fetch_context": "Fetching Context",
    "analyze_and_propose": "Analyzing & Proposing Transfer",
    "validate_constraints": "Constraint Validation",
    "replan_transfer": "Replanning Transfer",
    "select_captain": "Selecting Captain",
    "explain_recommendation": "Generating Recommendation",
}


def _format_fetch_context(update: dict) -> str:
    squad = update.get("current_squad", [])
    bank = update.get("bank", 0)
    candidates = update.get("candidates", [])
    ft = update.get("free_transfers", 1)
    return (
        f"Pulled live data from the FPL API.\n\n"
        f"- **Squad:** {len(squad)} players loaded\n"
        f"- **Bank:** £{bank / 10:.1f}m\n"
        f"- **Free transfers:** {ft}\n"
        f"- **Transfer candidates:** {len(candidates)} ranked by form + fixtures"
    )


def _format_analyze(update: dict) -> str:
    transfer = update.get("proposed_transfer")
    reasoning = update.get("transfer_reasoning", "")
    if transfer is None:
        return f"**Decision:** Hold — no transfer this gameweek.\n\n{reasoning}"
    out_p = transfer["out"]
    in_p = transfer["in"]
    return (
        f"**Proposed:** Sell **{out_p['web_name']}** "
        f"(£{out_p['now_cost'] / 10:.1f}m, form {out_p['form']}) → "
        f"Buy **{in_p['web_name']}** "
        f"(£{in_p['now_cost'] / 10:.1f}m, form {in_p['form']})\n\n"
        f"{reasoning}"
    )


def _format_validate(update: dict) -> tuple[str, str]:
    """Returns (title, content) since title depends on pass/fail."""
    is_valid = update.get("is_valid", False)
    violations = update.get("violations", [])
    if is_valid:
        title = "✅ Constraint Validation — PASSED"
        content = "All constraints satisfied."
        return title, content
    title = "⚠️ Constraint Validation — FAILED"
    v_lines = "\n".join(f"❌ {v}" for v in violations)
    content = f"```\n{v_lines}\n```\n\nSending back for replanning..."
    return title, content


def _format_replan(update: dict) -> str:
    transfer = update.get("proposed_transfer")
    reasoning = update.get("transfer_reasoning", "")
    count = update.get("replan_count", 1)
    if transfer is None:
        return f"**Attempt {count} — Decision:** Hold transfer.\n\n{reasoning}"
    out_p = transfer["out"]
    in_p = transfer["in"]
    return (
        f"**Revised proposal (attempt {count}):** Sell **{out_p['web_name']}** "
        f"(£{out_p['now_cost'] / 10:.1f}m) → "
        f"Buy **{in_p['web_name']}** "
        f"(£{in_p['now_cost'] / 10:.1f}m, form {in_p['form']})\n\n"
        f"{reasoning}"
    )


def _format_captain(update: dict) -> str:
    cap = update.get("captain_pick", {})
    vc = update.get("vice_captain_pick", {})
    reasoning = update.get("captain_reasoning", "")
    return (
        f"**Captain:** {cap.get('name', 'Unknown')} (C)\n"
        f"**Vice-captain:** {vc.get('name', 'Unknown')}\n\n"
        f"{reasoning}"
    )


def _format_explain(update: dict) -> str:
    return update.get("recommendation", "")


# -- Async generator that streams the graph -----------------------------------

async def run_agent(team_id: int, force_replan: bool):
    """Async generator that streams the graph execution as ChatMessage cards.

    Yields (messages, recommendation_str) tuples.
    """
    # Auto-detect next gameweek
    async with FPLClient() as client:
        gw_info = await client.get_next_gameweek()
    if gw_info is None:
        yield [ChatMessage(
            role="assistant",
            content="No upcoming gameweek found — the season may be over.",
            metadata={"title": "Error", "status": "done"},
        )], ""
        return

    target_gw = gw_info.id

    initial_state = {
        "team_id": int(team_id),
        "target_gw": target_gw,
        "provider": "openai",
        "force_replan": force_replan,
    }

    messages: list[ChatMessage] = []
    recommendation = ""

    async for event in graph.astream(initial_state):
        # Each event is {node_name: state_update_dict}
        for node_name, update in event.items():
            if node_name.startswith("__"):
                continue

            # Mark previous message as done
            if messages:
                messages[-1].metadata["status"] = "done"

            # Build card content based on node type
            if node_name == "fetch_context":
                title = NODE_TITLES[node_name]
                content = _format_fetch_context(update)
            elif node_name == "analyze_and_propose":
                title = NODE_TITLES[node_name]
                content = _format_analyze(update)
            elif node_name == "validate_constraints":
                title, content = _format_validate(update)
            elif node_name == "replan_transfer":
                count = update.get("replan_count", 1)
                title = f"{NODE_TITLES[node_name]} (attempt {count})"
                content = _format_replan(update)
            elif node_name == "select_captain":
                title = NODE_TITLES[node_name]
                content = _format_captain(update)
            elif node_name == "explain_recommendation":
                title = NODE_TITLES[node_name]
                content = _format_explain(update)
                recommendation = update.get("recommendation", "")
            else:
                title = node_name
                content = str(update)

            messages.append(ChatMessage(
                role="assistant",
                content=content,
                metadata={"title": title, "status": "pending"},
            ))

            yield messages, recommendation

    # Mark final message as done
    if messages:
        messages[-1].metadata["status"] = "done"
        yield messages, recommendation


# -- Gradio layout ------------------------------------------------------------

with gr.Blocks(title="FPL Transfer Strategist") as demo:
    gr.Markdown(
        "## FPL Transfer Strategist\n\n"
        "Watch the agent reason about your team in real time. Each step shows "
        "what it's doing, what it found, and how it self-corrects when it makes "
        "a mistake."
    )

    with gr.Row():
        team_id = gr.Number(label="Team ID", value=44, precision=0)
        force_replan = gr.Checkbox(label="Force replan (demo mode)", value=False)
        run_btn = gr.Button("Get Recommendation", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Agent Reasoning Trace", height=600)
            recommendation_md = gr.Markdown(label="Recommendation")
        with gr.Column(scale=1):
            gr.Markdown("_Squad and recommendation details will appear here in Step 5.3_")

    run_btn.click(
        fn=run_agent,
        inputs=[team_id, force_replan],
        outputs=[chatbot, recommendation_md],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
