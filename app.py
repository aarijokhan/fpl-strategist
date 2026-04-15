"""Gradio web UI for the FPL Transfer Strategist."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, timezone, datetime
from pathlib import Path

# Ensure src/ is on the import path (needed when running as `python app.py`
# without an editable install, e.g. on HuggingFace Spaces).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
import httpx
import openai
from gradio import ChatMessage

from app_helpers import (
    build_captain_card,
    build_meta_footer,
    build_squad_html,
    build_summary_headline,
    build_transfer_card,
)
from fpl_strategist.data.fpl_client import FPLClient
from fpl_strategist.graph import graph
from fpl_strategist.llm import set_api_key_override

# -- Cached demo fallback -----------------------------------------------------

_DEMO_CACHE_PATH = Path(__file__).resolve().parent / "src" / "fpl_strategist" / "data" / "demo_cache.json"

# -- Daily rate-limit (in-memory, resets on restart) --------------------------

_daily_runs = 0
_day_start: date | None = None
_DAILY_CAP = 50


def _check_daily_cap() -> str | None:
    """Return an error message if the daily cap is exceeded, else None."""
    global _daily_runs, _day_start
    today = datetime.now(timezone.utc).date()
    if _day_start != today:
        _daily_runs = 0
        _day_start = today
    if _daily_runs >= _DAILY_CAP:
        return (
            "Daily limit of 50 runs reached. Please try again tomorrow, "
            "or use your own OpenAI API key below for unlimited runs."
        )
    return None


def _increment_daily_runs() -> None:
    global _daily_runs
    _daily_runs += 1


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


# Skip token — tells Gradio "don't touch this component on this yield"
_S = gr.skip()

# -- Cached demo fallback generator -------------------------------------------

async def _replay_cached_demo():
    """Replay demo_cache.json as a generator with the same yield shape as run_agent."""
    with open(_DEMO_CACHE_PATH) as f:
        cache = json.load(f)

    gw = cache["gameweek"]
    messages: list[ChatMessage] = []

    # Warning banner
    messages.append(ChatMessage(
        role="assistant",
        content=f"Live FPL API unavailable — showing a cached demo from the 2023/24 season (GW {gw}).",
        metadata={"title": "Notice", "status": "done"},
    ))
    yield messages, _S, _S, _S, _S, _S, _S

    for event in cache["trace_events"]:
        # Mark previous as done (the warning banner is already done)
        if len(messages) > 1:
            messages[-1].metadata["status"] = "done"

        messages.append(ChatMessage(
            role="assistant",
            content=event["content"],
            metadata={"title": event["title"], "status": "pending"},
        ))
        yield messages, _S, _S, _S, _S, _S, _S
        await asyncio.sleep(1.0)

    # Final yield: mark last message done and populate right column
    if messages:
        messages[-1].metadata["status"] = "done"

    fs = cache["final_state"]
    recommendation = fs.get("recommendation", "")
    yield (
        messages,
        build_summary_headline(fs),
        recommendation,
        build_transfer_card(fs),
        build_captain_card(fs),
        build_meta_footer(fs),
        build_squad_html(fs),
    )


# -- Async generator that streams the graph -----------------------------------

async def run_agent(team_id: int, force_replan: bool, byok_key: str = ""):
    """Async generator that streams the graph execution as ChatMessage cards.

    Yields 6-tuples:
      (trace_messages, recommendation_text, squad_html, transfer_card_md,
       captain_card_md, meta_md)

    During streaming, only trace_messages and recommendation_text update.
    On the final yield, all six populate from the accumulated state.
    """

    using_byok = bool(byok_key and byok_key.strip())

    # --- Daily cap (hosted key only) ---
    if not using_byok:
        cap_error = _check_daily_cap()
        if cap_error is not None:
            yield [ChatMessage(
                role="assistant",
                content=cap_error,
                metadata={"title": "Daily limit reached", "status": "done"},
            )], "", "", "", "", "", ""
            return

    # Set BYOK override (None clears it for hosted-key runs)
    set_api_key_override(byok_key.strip() if using_byok else None)

    def _error_yield(title: str, content: str):
        return [ChatMessage(
            role="assistant",
            content=content,
            metadata={"title": title, "status": "done"},
        )], "", "", "", "", "", ""

    try:
        # Auto-detect next gameweek
        async with FPLClient() as client:
            gw_info = await client.get_next_gameweek()
        if gw_info is None:
            async for result in _replay_cached_demo():
                yield result
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
        cumulative_state: dict = dict(initial_state)

        async for event in graph.astream(initial_state):
            # Each event is {node_name: state_update_dict}
            for node_name, update in event.items():
                if node_name.startswith("__"):
                    continue

                # Accumulate state for final right-column build
                cumulative_state.update(update)

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

                yield messages, _S, recommendation, _S, _S, _S, _S

        # Mark final message as done and populate right column
        if messages:
            messages[-1].metadata["status"] = "done"
            yield (
                messages,
                build_summary_headline(cumulative_state),
                recommendation,
                build_transfer_card(cumulative_state),
                build_captain_card(cumulative_state),
                build_meta_footer(cumulative_state),
                build_squad_html(cumulative_state),
            )

        # Only count successful runs against the daily cap
        if not using_byok:
            _increment_daily_runs()

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            yield _error_yield(
                "Team not found",
                f"No FPL team found for ID **{int(team_id)}**. "
                "Check yours at fantasy.premierleague.com \u2192 My Team "
                "\u2192 number in the URL.",
            )
        else:
            async for result in _replay_cached_demo():
                yield result

    except httpx.ConnectError:
        async for result in _replay_cached_demo():
            yield result

    except openai.AuthenticationError:
        yield _error_yield(
            "\u26a0\ufe0f API key rejected",
            "The OpenAI API rejected the provided key. If you pasted a custom "
            "key in the Advanced section, please verify it starts with `sk-` "
            "and is still active. Otherwise, remove the custom key to use the "
            "hosted key.",
        )

    except openai.RateLimitError:
        yield _error_yield(
            "\u26a0\ufe0f Rate limit",
            "Rate limit reached on the OpenAI API. Please wait a moment and "
            "try again, or use your own API key in Advanced.",
        )

    except Exception:
        yield _error_yield(
            "\u26a0\ufe0f Something went wrong",
            "An unexpected error occurred while running the agent. Please try "
            "again. If the problem persists, check back later or use your own "
            "API key in Advanced.",
        )


# -- Gradio layout ------------------------------------------------------------

with gr.Blocks(title="FPL Transfer Strategist") as demo:
    gr.Markdown(
        "## FPL Transfer Strategist\n\n"
        "Watch the agent reason about your team in real time. Each step shows "
        "what it's doing, what it found, and how it self-corrects when it makes "
        "a mistake."
    )

    with gr.Row():
        team_id = gr.Number(
            label="Team ID", value=44, precision=0,
            info="Find yours at fantasy.premierleague.com → My Team → number in the URL",
        )
        run_btn = gr.Button("Get Recommendation", variant="primary")

    with gr.Accordion("Advanced", open=False):
        force_replan = gr.Checkbox(
            label="Simulate constraint violation",
            value=False,
            info="Forces an over-budget proposal to demonstrate the replan loop.",
        )
        byok_key = gr.Textbox(
            label="OpenAI API Key (optional)",
            type="password",
            placeholder="sk-...",
        )
        gr.Markdown(
            "Your key is used only for this session — never stored or logged. "
            "[Source code on GitHub](https://github.com/aarijokhan/fpl-strategist) "
            "for verification."
        )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Agent Reasoning Trace", height=600)
            summary_md = gr.Markdown()
            recommendation_md = gr.Markdown(label="Recommendation")
        with gr.Column(scale=1):
            transfer_card = gr.Markdown()
            captain_card = gr.Markdown()
            meta_footer = gr.Markdown()
            squad_table = gr.HTML(label="Post-Transfer Squad")

    run_btn.click(
        fn=run_agent,
        inputs=[team_id, force_replan, byok_key],
        outputs=[chatbot, summary_md, recommendation_md, transfer_card, captain_card, meta_footer, squad_table],
        concurrency_limit=1,
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
