"""
Streamlit UI for the multi-agent research system.

Drop this file into the same folder as agents.py, pipeline.py, tools.py,
and your .env file, then run:

    streamlit run app.py

Requires streamlit >= 1.32 (uses `key=` on st.container to scope CSS to
specific panels).

NAMING NOTE: every function/variable name that mirrors pipeline.py
(run_research_pipeline, search_agent, search_result, reader_agent,
reader_result, research_combined, writer_report, critic_result, state,
route_decision, route) is kept identical to what's in pipeline.py on
purpose, so the two files stay easy to compare side by side. The only
new names below (render_*, run_research_pipeline_ui, STAGE_COLORS, etc.)
are UI-only helpers that don't exist in pipeline.py.

ROUTER ASSUMPTION: this mirrors pipeline.py's own routing code exactly —
`route_decision = router_chain.invoke(...)` then `route_decision.category`
— which only works if router_chain returns a structured object with a
`.category` field (e.g. via `.with_structured_output(...)`). A defensive
fallback below also accepts a plain string in case router_chain still ends
in StrOutputParser(); see `resolve_route()`.
"""

import datetime
import uuid

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents import (
    build_fetch_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    router_chain,
    llm,
)


# --------------------------------------------------------------------------
# Config — edit this for your resume / portfolio write-up
# --------------------------------------------------------------------------
PROJECT_TITLE = "Multi-Agent Research System"
PROJECT_TAGLINE = (
    "A router chain classifies each query, then either answers directly "
    "or hands it to a four-stage agent pipeline that searches the web, "
    "extracts the best source, drafts a report, and critiques its own output."
)
TECH_STACK = [
    "Python",
    "LangChain",
    "Groq",
    "Tavily",
    "BeautifulSoup",
    "Streamlit"
]

# id -> (role text shown in sidebar, accent color used everywhere for that stage)
STEPS = [
    {"id": "SEARCH_AGENT", "role": "Finds current, credible sources for the topic", "color": "#22D3EE", "icon": "🔎"},
    {"id": "FETCH_AGENT", "role": "Scrapes the most relevant result for full context", "color": "#F5B942", "icon": "📄"},
    {"id": "WRITER_CHAIN", "role": "Synthesizes search + scraped data into a report", "color": "#F5679C", "icon": "✍️"},
    {"id": "CRITIC_CHAIN", "role": "Reviews the report for gaps and accuracy", "color": "#34D399", "icon": "🧪"},
]
ROUTER_COLOR = "#8B7CF6"

st.set_page_config(page_title=PROJECT_TITLE, page_icon="◆", layout="wide")


# --------------------------------------------------------------------------
# Theme — dark technical dashboard, recolored to be vivid per pipeline stage
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0A0D14;
  --surface: #12161F;
  --surface-alt: #171C27;
  --border: #242A3B;
  --text: #E8EAF0;
  --text-muted: #8891A5;
  --violet: #8B7CF6;
  --cyan: #22D3EE;
  --amber: #F5B942;
  --pink: #F5679C;
  --green: #34D399;
  --danger: #F16063;
}

/* Content was rendering oversized because Streamlit's default block
   container is very wide/tall with generous padding. Scaling the root
   font-size down tightens every rem-based measurement in this stylesheet
   at once (paddings, gaps, font sizes) without touching any colors,
   layout structure, or component design. */
html { font-size: 15px; }

[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1180px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
}
[data-testid="stSidebar"] .block-container {
  padding-top: 1.4rem;
}

.stApp {
  background:
    radial-gradient(circle at 15% -10%, rgba(139,124,246,.14), transparent 45%),
    radial-gradient(circle at 85% 0%, rgba(34,211,238,.10), transparent 40%),
    var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: .95rem;
}
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

/* keep long unbroken strings (urls, ids) from stretching cards wider
   than the viewport and forcing horizontal scrolling / zoomed-out look */
[data-testid="stChatMessage"], [class*="st-key-report"], [class*="st-key-critic"],
[data-testid="stExpander"], .log-panel {
  overflow-wrap: anywhere;
  word-break: break-word;
}
[data-testid="stMarkdownContainer"] pre { overflow-x: auto; }
[data-testid="stMarkdownContainer"] img { max-width: 100%; height: auto; }

[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }

.sb-logo {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.05rem; font-weight: 700; letter-spacing: .02em;
  margin-top: .25rem; display: flex; align-items: center; gap: .4rem;
  background: linear-gradient(90deg, var(--violet), var(--cyan));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-logo::before { content: "◆"; color: var(--violet); -webkit-text-fill-color: var(--violet); }
.sb-sub { font-size: .76rem; color: var(--text-muted); margin-bottom: 1rem; }
.sb-divider { height: 1px; background: var(--border); margin: 1rem 0; }
.sb-heading {
  font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .12em;
  color: var(--text-muted); text-transform: uppercase; margin-bottom: .55rem;
}
.sb-about { font-size: .8rem; color: var(--text-muted); line-height: 1.55; margin-bottom: .25rem; }
.sb-stage { display: flex; gap: .5rem; align-items: baseline; font-size: .74rem; margin-bottom: .5rem; line-height: 1.35; }
.sb-stage-id { font-family: 'JetBrains Mono', monospace; font-weight: 600; white-space: nowrap; }
.sb-stage-role { color: var(--text-muted); }
.sb-pill-row { display: flex; flex-wrap: wrap; gap: .4rem; }
.sb-pill {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: var(--text-muted);
  border: 1px solid var(--border); border-radius: 3px; padding: .15rem .5rem;
}
.sb-history { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--text-muted);
  padding: .3rem .5rem; border-left: 2px solid var(--border); margin-bottom: .35rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-history-empty { font-size: .74rem; color: var(--text-muted); font-style: italic; }

.hdr-title {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700;
  letter-spacing: -.01em; animation: fadein .5s ease-out;
  background: linear-gradient(90deg, var(--violet), var(--cyan) 55%, var(--green));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  display: inline-block;
}
.hdr-tagline {
  color: var(--text-muted); font-size: .85rem; line-height: 1.5; max-width: 720px;
  margin-top: .3rem; margin-bottom: .8rem; animation: fadein .7s ease-out;
}
.hdr-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1rem; }
.hdr-badge {
  font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: var(--violet);
  border: 1px solid rgba(139,124,246,.35); background: rgba(139,124,246,.12);
  border-radius: 3px; padding: .2rem .55rem;
}
@keyframes fadein { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }

.empty-state {
  color: var(--text-muted); font-style: italic; font-size: .9rem;
  padding: 1rem 0; border-top: 1px dashed var(--border); margin-top: .25rem;
}

/* architecture diagram */
[class*="st-key-arch"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: .9rem 1.1rem 1.1rem 1.1rem; margin-bottom: .9rem;
}
.arch-heading {
  font-family: 'JetBrains Mono', monospace; font-size: .64rem; letter-spacing: .1em;
  color: var(--text-muted); text-transform: uppercase; margin-bottom: .7rem;
}
.arch-wrap { display: flex; align-items: center; gap: .7rem; flex-wrap: wrap; }
.arch-router {
  border: 1px solid var(--violet); background: rgba(139,124,246,.12); border-radius: 6px;
  padding: .45rem .7rem; font-family: 'JetBrains Mono', monospace; font-size: .68rem;
  color: var(--violet); font-weight: 600; white-space: nowrap; flex-shrink: 0;
}
.arch-arrow { color: var(--text-muted); font-size: 1rem; flex-shrink: 0; }
.arch-branches { display: flex; flex-direction: column; gap: .8rem; flex: 1; min-width: 260px; }
.arch-branch { display: flex; align-items: center; gap: .65rem; flex-wrap: wrap; }
.arch-branch-label {
  font-family: 'JetBrains Mono', monospace; font-size: .64rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: .08em; width: 68px; flex-shrink: 0;
}
.arch-chain { display: flex; align-items: center; flex-wrap: wrap; gap: 0; }
.arch-node {
  border: 1px solid var(--border); background: var(--surface-alt); border-radius: 6px;
  padding: .32rem .6rem; font-family: 'JetBrains Mono', monospace; font-size: .64rem;
  color: var(--text-muted); white-space: nowrap; display: flex; align-items: center; gap: .35rem;
  transition: all .25s ease;
}
.arch-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.arch-node.active { color: var(--text); box-shadow: 0 0 14px 0 var(--node-color, var(--violet)); border-color: var(--node-color, var(--violet)); }
.arch-node.active .arch-dot { background: var(--node-color, var(--violet)); animation: dotpulse 1.4s ease-out infinite; }
.arch-node.done { color: var(--node-color, var(--green)); border-color: var(--node-color, var(--green)); background: color-mix(in srgb, var(--node-color, var(--green)) 12%, var(--surface-alt)); }
.arch-node.done .arch-dot { background: var(--node-color, var(--green)); }
@keyframes dotpulse {
  0% { box-shadow: 0 0 0 0 rgba(245,185,66,.5); }
  70% { box-shadow: 0 0 0 6px rgba(245,185,66,0); }
  100% { box-shadow: 0 0 0 0 rgba(245,185,66,0); }
}
.arch-connector { width: 18px; height: 1px; background: var(--border); transition: background .3s ease; }
.arch-connector.done { background: var(--green); }

/* chat */
[data-testid="stChatMessage"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: .65rem .95rem; margin-bottom: .5rem;
}
[data-testid="stChatInput"] textarea {
  font-family: 'Inter', sans-serif; background: var(--surface) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text) !important;
}
[data-testid="stChatInput"] { border-top: 1px solid var(--border); }

.route-badge {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: .68rem;
  letter-spacing: .06em; padding: .22rem .6rem; border-radius: 4px; margin-bottom: .7rem;
}
.route-research { border: 1px solid var(--violet); color: var(--violet); background: rgba(139,124,246,.12); }
.route-chat { border: 1px solid var(--green); color: var(--green); background: rgba(52,211,153,.1); }
.route-unknown { border: 1px solid var(--danger); color: var(--danger); background: rgba(241,96,99,.1); }

/* execution log */
.log-panel {
  background: var(--surface-alt); border: 1px solid var(--border); border-radius: 6px;
  padding: .8rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: .74rem;
  line-height: 1.65; margin: .5rem 0 1rem 0;
}
.log-line { color: var(--text-muted); }
.log-line.running { color: var(--amber); }
.log-line.done { color: var(--green); }

/* report / critic cards */
[class*="st-key-report"] {
  background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--pink);
  border-radius: 6px; padding: .9rem 1.1rem; margin-bottom: .55rem;
  max-height: 560px; overflow-y: auto;
}
[class*="st-key-critic"] {
  background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--green);
  border-radius: 6px; padding: .85rem 1.1rem; margin-bottom: .55rem;
  max-height: 420px; overflow-y: auto;
}
.card-tag {
  font-family: 'JetBrains Mono', monospace; font-size: .62rem; letter-spacing: .1em;
  color: var(--text-muted); text-transform: uppercase; border-bottom: 1px solid var(--border);
  padding-bottom: .45rem; margin-bottom: .65rem; display: flex; justify-content: space-between; align-items: center;
  position: sticky; top: 0; background: var(--surface);
}
.score-pill {
  font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: .72rem;
  padding: .18rem .55rem; border-radius: 999px; letter-spacing: .04em;
  background: linear-gradient(90deg, var(--violet), var(--cyan)); color: #0A0D14;
}

[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 6px !important; background: var(--surface) !important; }
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: var(--text-muted); }

.stButton > button {
  background: linear-gradient(90deg, var(--violet), var(--cyan));
  color: #0A0D14; border: none; border-radius: 6px;
  font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: .01em;
  transition: opacity .2s ease;
}
.stButton > button:hover { opacity: .85; }
.stButton > button:focus-visible { outline: 2px solid var(--violet); outline-offset: 2px; }

.stDownloadButton > button {
  background: var(--surface-alt); color: var(--text); border: 1px solid var(--green);
  border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: .78rem;
}
.stDownloadButton > button:hover { background: rgba(52,211,153,.12); }

.error-note {
  font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: var(--danger);
  border-left: 3px solid var(--danger); padding: .5rem .8rem; background: var(--surface);
  margin-bottom: .6rem; border-radius: 4px;
}
[data-testid="stAlert"] { border-left: 3px solid var(--danger) !important; }

@media (max-width: 640px) {
  .hdr-title { font-size: 1.3rem; }
  .arch-branch-label { width: auto; }
  [class*="st-key-report"], [class*="st-key-critic"] { max-height: 400px; }
}
@media (prefers-reduced-motion: reduce) {
  .arch-node.active .arch-dot, .hdr-title, .hdr-tagline { animation: none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------------------------------
# Render helpers
# --------------------------------------------------------------------------
def _node(label: str, color: str, state: str = "idle") -> str:
    cls = f"arch-node {state}" if state != "idle" else "arch-node"
    return f'<div class="{cls}" style="--node-color:{color}"><span class="arch-dot"></span>{label}</div>'


def render_architecture_html(active_idx: int = None, done: set = frozenset(), route: str = None) -> str:
    """Static explainer diagram; also doubles as the live status view when
    route/active_idx/done are supplied for the current turn."""
    router_state = "done" if route else "idle"
    chat_state = "done" if route == "CASUAL" else "idle"

    chain_nodes = []
    for i, step in enumerate(STEPS):
        if route == "RESEARCH" and i in done:
            state = "done"
        elif route == "RESEARCH" and i == active_idx:
            state = "active"
        else:
            state = "idle"
        chain_nodes.append(_node(f'{step["icon"]} {step["id"]}', step["color"], state))
        if i < len(STEPS) - 1:
            conn_cls = "arch-connector done" if (route == "RESEARCH" and i in done) else "arch-connector"
            chain_nodes.append(f'<div class="{conn_cls}"></div>')

    router_style = f' style="border-color:{ROUTER_COLOR};color:{ROUTER_COLOR}"' if router_state == "done" else ""

    return (
        '<div class="arch-heading">System architecture</div>'
        '<div class="arch-wrap">'
        f'<div class="arch-router"{router_style}>🧭 ROUTER_CHAIN</div>'
        '<div class="arch-arrow">&#8594;</div>'
        '<div class="arch-branches">'
        '<div class="arch-branch">'
        '<div class="arch-branch-label">research</div>'
        f'<div class="arch-chain">{"".join(chain_nodes)}</div>'
        '</div>'
        '<div class="arch-branch">'
        '<div class="arch-branch-label">casual</div>'
        f'<div class="arch-chain">{_node("💬 LLM (direct)", "#34D399", chat_state)}</div>'
        '</div>'
        '</div>'
        '</div>'
    )


def render_badge_html(route: str) -> str:
    if route == "RESEARCH":
        return '<div class="route-badge route-research">ROUTED &rarr; RESEARCH PIPELINE</div>'
    if route == "CASUAL":
        return '<div class="route-badge route-chat">ROUTED &rarr; DIRECT RESPONSE</div>'
    return '<div class="route-badge route-unknown">ROUTE UNKNOWN</div>'


def render_log_html(lines: list) -> str:
    rows = []
    for line in lines:
        cls = "log-line"
        if line.endswith("running…"):
            cls += " running"
        elif "completed" in line:
            cls += " done"
        rows.append(f'<div class="{cls}">&#10095; {line}</div>')
    return f'<div class="log-panel">{"".join(rows)}</div>'


def render_status_block(active_idx: int, done: set, log_lines: list) -> str:
    return render_architecture_html(active_idx, done, "RESEARCH") + render_log_html(log_lines)


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def extract_score(critic_report: str) -> str:
    """Pulls the 'Score: X/10' line out of critic_chain's output for the score pill."""
    import re
    match = re.search(r"Score:\s*([\d.]+)\s*/\s*10", critic_report or "")
    return match.group(1) if match else "?"


def resolve_route(route_decision) -> str:
    """
    Mirrors pipeline.py: `route = route_decision.category`.
    Falls back to treating route_decision as a plain string if router_chain
    hasn't been switched to structured output yet, so this UI doesn't hard
    -crash either way.
    """
    category = getattr(route_decision, "category", route_decision)
    if isinstance(category, str):
        category = category.strip().upper()
    return category


def run_research_pipeline_ui(topic: str, placeholder) -> dict:
    """
    Same four calls, same variable names, and the same try/except-per-step
    shape as pipeline.run_research_pipeline — just instrumented to update
    the live architecture diagram + execution log as each stage finishes.
    """
    done = set()
    log_lines = [f"{now()}  router_chain.invoke(query) &rarr; RESEARCH"]
    state = {}

    # STEP 1 - SEARCH AGENT
    log_lines.append(f"{now()}  search_agent.invoke(...) &rarr; running…")
    placeholder.markdown(render_status_block(0, done, log_lines), unsafe_allow_html=True)
    try:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable, and detailed information about:\n\n{topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content
    except Exception as e:
        state["search_results"] = f"Search failed: {str(e)}"
    done.add(0)
    log_lines[-1] = f"{now()}  search_agent.invoke(...) &rarr; completed ({len(state['search_results'])} chars)"
    placeholder.markdown(render_status_block(1, done, log_lines), unsafe_allow_html=True)

    # STEP 2 - READER / FETCH AGENT
    log_lines.append(f"{now()}  reader_agent.invoke(...) &rarr; running…")
    placeholder.markdown(render_status_block(1, done, log_lines), unsafe_allow_html=True)
    try:
        reader_agent = build_fetch_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on the following search results about:\n\n{topic}\n\n"
                f"Identify the most relevant URL and scrape it using the scrape_url tool.\n\n"
                f"Search Results:\n{state['search_results'][:3000]}\n\n"
                f"Return the useful factual information obtained from the source."
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content
    except Exception as e:
        state["scraped_content"] = f"Source reading failed: {str(e)}"
    done.add(1)
    log_lines[-1] = f"{now()}  reader_agent.invoke(...) &rarr; completed ({len(state['scraped_content'])} chars)"
    placeholder.markdown(render_status_block(2, done, log_lines), unsafe_allow_html=True)

    # STEP 3 - WRITER
    log_lines.append(f"{now()}  writer_chain.invoke(...) &rarr; running…")
    placeholder.markdown(render_status_block(2, done, log_lines), unsafe_allow_html=True)
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    try:
        writer_report = writer_chain.invoke({"topic": topic, "research": research_combined})
        state["report"] = writer_report
    except Exception as e:
        state["report"] = f"Report generation failed: {str(e)}"
    done.add(2)
    log_lines[-1] = f"{now()}  writer_chain.invoke(...) &rarr; completed ({len(state['report'])} chars)"
    placeholder.markdown(render_status_block(3, done, log_lines), unsafe_allow_html=True)

    # STEP 4 - CRITIC
    log_lines.append(f"{now()}  critic_chain.invoke(...) &rarr; running…")
    placeholder.markdown(render_status_block(3, done, log_lines), unsafe_allow_html=True)
    try:
        critic_result = critic_chain.invoke({"report": state["report"]})
        state["critic_report"] = critic_result
    except Exception as e:
        state["critic_report"] = f"Critic review failed: {str(e)}"
    done.add(3)
    log_lines[-1] = f"{now()}  critic_chain.invoke(...) &rarr; completed ({len(state['critic_report'])} chars)"
    placeholder.markdown(render_status_block(-1, done, log_lines), unsafe_allow_html=True)

    return state


def render_research_result(state: dict, topic: str, key_suffix: str):
    score = extract_score(state.get("critic_report", ""))

    with st.container(key=f"report_{key_suffix}"):
        st.markdown(
            f'<div class="card-tag"><span>📝 Research report &mdash; {topic}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(state["report"])

    st.download_button(
        "⬇ Download report (.md)",
        data=state["report"],
        file_name=f"{topic[:40].strip().replace(' ', '_') or 'report'}.md",
        mime="text/markdown",
        key=f"dl_{key_suffix}",
    )

    with st.container(key=f"critic_{key_suffix}"):
        st.markdown(
            f'<div class="card-tag"><span>🧪 Critic review</span><span class="score-pill">SCORE {score}/10</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(state["critic_report"])

    with st.expander("Raw search results"):
        st.text(state["search_results"])
    with st.expander("Raw scraped content"):
        st.text(state["scraped_content"])


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-logo">RESEARCH SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Multi-agent pipeline console</div>', unsafe_allow_html=True)

    if st.button("New session", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-heading">About</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-about">{PROJECT_TAGLINE}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-heading">Pipeline stages</div>', unsafe_allow_html=True)
    for step in STEPS:
        st.markdown(
            f'<div class="sb-stage"><span class="sb-stage-id" style="color:{step["color"]}">{step["icon"]} {step["id"]}</span>'
            f'<span class="sb-stage-role">{step["role"]}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-heading">Tech stack</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-pill-row">' + "".join(f'<span class="sb-pill">{t}</span>' for t in TECH_STACK) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-heading">Session history</div>', unsafe_allow_html=True)
    queries = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
    if queries:
        for i, q in enumerate(queries, 1):
            label = q if len(q) <= 34 else q[:31] + "…"
            st.markdown(f'<div class="sb-history">{i:02d}  {label}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-history-empty">No queries yet.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header + architecture explainer
# --------------------------------------------------------------------------
st.markdown(f'<div class="hdr-title">{PROJECT_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hdr-tagline">{PROJECT_TAGLINE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hdr-badges">' + "".join(f'<span class="hdr-badge">{t}</span>' for t in TECH_STACK) + "</div>",
    unsafe_allow_html=True,
)

with st.container(key="arch_static"):
    st.markdown(render_architecture_html(), unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown('<div class="empty-state">No queries yet — ask something below to see the pipeline run.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Replay chat history
# --------------------------------------------------------------------------
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(render_badge_html(msg["meta"]["route"]), unsafe_allow_html=True)
            if msg["meta"]["type"] == "research":
                st.markdown(
                    render_architecture_html(-1, {0, 1, 2, 3}, "RESEARCH") + render_log_html(msg["meta"]["log"]),
                    unsafe_allow_html=True,
                )
                render_research_result(msg["meta"]["state"], msg["meta"]["topic"], key_suffix=f"h{idx}")
            else:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])


# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------
user_input = st.chat_input("Ask something, or ask me to research a topic…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # --------------------------------------------------
        # ROUTER — mirrors pipeline.py's route_decision/route
        # --------------------------------------------------
        try:
            route_decision = router_chain.invoke({"query": user_input})
            route = resolve_route(route_decision)
        except Exception as e:
            route = None
            st.markdown(f'<div class="error-note">Router error: {e}</div>', unsafe_allow_html=True)

        if route is not None:
            st.markdown(render_badge_html(route), unsafe_allow_html=True)

        # --------------------------------------------------
        # RESEARCH
        # --------------------------------------------------
        if route == "RESEARCH":
            placeholder = st.empty()
            state = run_research_pipeline_ui(user_input, placeholder)
            log_snapshot = [
                "router_chain.invoke(query) → RESEARCH",
                f"search_agent.invoke(...) → completed ({len(state['search_results'])} chars)",
                f"reader_agent.invoke(...) → completed ({len(state['scraped_content'])} chars)",
                f"writer_chain.invoke(...) → completed ({len(state['report'])} chars)",
                f"critic_chain.invoke(...) → completed ({len(state['critic_report'])} chars)",
            ]
            render_research_result(state, user_input, key_suffix=f"live{uuid.uuid4().hex[:8]}")
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": state["report"],
                    "meta": {
                        "type": "research",
                        "state": state,
                        "topic": user_input,
                        "route": route,
                        "log": log_snapshot,
                    },
                }
            )

        # --------------------------------------------------
        # CASUAL
        # --------------------------------------------------
        elif route == "CASUAL":
            try:
                response = llm.invoke(user_input)
                answer = response.content
            except Exception as e:
                answer = f"Assistant error: {e}"
            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "meta": {"type": "chat", "route": route}}
            )

        # --------------------------------------------------
        # UNKNOWN ROUTE
        # --------------------------------------------------
        elif route is not None:
            msg = "Unable to determine the request type."
            st.markdown(msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": msg, "meta": {"type": "chat", "route": route}}
            )

    st.rerun()