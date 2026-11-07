"""Streamlit front-end ("flight log" console).   streamlit run app.py
All research logic lives in src/ ; this file only renders."""
import json
import os
import streamlit as st
import yaml

from src.benchmark.generator import load_scenarios
from src.models.schemas import Mission
from src.planner.llm import LLMError, PRESETS
from src.repair.loop import LoopConfig
from src.app_logic import run_session, run_dynamic_session


def _load_dotenv(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.split(" #", 1)[0].strip()   # ignore trailing "  # comment" as in .env.example
                    if v:
                        os.environ.setdefault(k.strip(), v)


for _p in (".env", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    _load_dotenv(_p)

st.set_page_config(page_title="Mission Verification Log", layout="wide", initial_sidebar_state="expanded")

PAPER, INK, RULE, RED, GREEN, AMBER = "#f3efe6", "#1d1c1a", "#cfc8b8", "#a3321f", "#2f5d3a", "#9a6a12"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:wght@400;600&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background:{PAPER}; color:{INK}; }}
[data-testid="stSidebar"] {{ background:#e9e3d3; border-right:1px solid {RULE}; }}
h1,h2,h3 {{ font-family:'IBM Plex Serif',Georgia,serif !important; font-weight:600; letter-spacing:-0.01em; }}
html, body, p, label, span, div {{ font-family:'IBM Plex Serif',Georgia,serif; }}
code, pre, .mono {{ font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace !important; }}
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ visibility:hidden; }}
/* v12.2: keep the sidebar and its open/close arrows readable on the light page in every Streamlit theme (dark mode made them invisible) */
[data-testid="stSidebar"], [data-testid="stSidebar"] * {{ color:{INK}; }}
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {{ visibility:visible !important; opacity:1 !important; }}
[data-testid="collapsedControl"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapseButton"] *,
[data-testid="stHeader"] button, [data-testid="stHeader"] button * {{ color:{INK} !important; fill:{INK} !important; }}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined {{ font-family:'Material Symbols Rounded','Material Icons' !important; }}
.block-container {{ padding-top:1.6rem; max-width:1250px; }}
.masthead {{ border-top:3px solid {INK}; border-bottom:1px solid {INK}; padding:.5rem 0 .6rem; margin-bottom:1rem; }}
.masthead .t {{ font-size:1.55rem; font-weight:600; }}
.masthead .s {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:#6b665b; }}
.lbl {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.14em; color:#6b665b; margin-bottom:.15rem; }}
.strip {{ display:grid; grid-template-columns:70px 150px 1fr 120px; gap:0; border:1px solid {INK}; border-left:6px solid var(--c); background:#faf8f2; margin:0 0 6px; }}
.strip > div {{ padding:.45rem .7rem; border-right:1px solid {RULE}; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }}
.strip > div:last-child {{ border-right:none; text-align:right; }}
.stamp {{ display:inline-block; border:1.5px solid var(--c); color:var(--c); padding:0 .4rem; font-weight:500; letter-spacing:.06em; transform:rotate(-1.5deg); }}
.vio {{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; border-left:3px solid {RED}; background:#f6e9e4; padding:.35rem .6rem; margin:.25rem 0; }}
.ok {{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; border-left:3px solid {GREEN}; background:#e6eee6; padding:.35rem .6rem; }}
.mockbar {{ background:{RED}; color:#fff; font-family:'IBM Plex Mono',monospace; font-size:.74rem; letter-spacing:.08em; padding:.4rem .7rem; margin-bottom:.8rem; }}
.stButton>button {{ border-radius:0; border:1.5px solid {INK}; background:{INK}; color:{PAPER}; font-family:'IBM Plex Mono',monospace; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem; }}
.stButton>button:hover {{ background:{PAPER}; color:{INK}; border-color:{INK}; }}
[data-baseweb="select"] > div, textarea, input {{ border-radius:0 !important; background:#faf8f2 !important; color:{INK} !important; }}
[data-baseweb="select"] span, [data-baseweb="select"] [data-testid="stSelectboxNativeSelect"], [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{ color:{INK} !important; }}
[data-baseweb="select"] svg {{ color:{INK} !important; fill:{INK} !important; }}
input::placeholder, textarea::placeholder {{ color:#6b665b !important; opacity:1; }}
label, summary, [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p, h1,h2,h3,h4,h5,h6 {{ color:{INK} !important; }}
[data-testid="stMetric"] {{ border-top:2px solid {INK}; padding-top:.2rem; }}
[data-testid="stMetricLabel"] p {{ font-family:'IBM Plex Mono',monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; }}
.stTabs [data-baseweb="tab"] {{ font-family:'IBM Plex Mono',monospace; font-size:.76rem; text-transform:uppercase; letter-spacing:.1em; color:{INK} !important; }}
[data-testid="stJson"], [data-testid="stJson"] * {{ background:#faf8f2 !important; color:{INK} !important; }}
pre, code {{ background:#faf8f2 !important; color:{INK} !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{ color:{INK} !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {{ color:#6b665b !important; }}
[data-testid="stProgress"] * {{ color:{INK} !important; }}
</style>
""", unsafe_allow_html=True)

with open("config/default.yaml", encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)
COLORS = {"SUCCESS": GREEN, "PARTIAL_REPAIR": AMBER, "NO_CHANGE": "#6b665b", "LATERAL": "#6b665b", "REGRESSION": RED, "PARSE_FAILURE": RED, "INITIAL": INK,
          "FAILURE": RED, "INFEASIBLE": "#5b3b8c"}

st.markdown('<div class="masthead"><div class="s">Verification log / UAV mission planning</div>'
            '<div class="t">Plan, verify, repair: every revision on the record</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="lbl">Planner</div>', unsafe_allow_html=True)
    provider = st.selectbox("Provider", ["mock", "ollama", "gemini", "groq", "openrouter", "requesty", "grok", "cohere", "anthropic", "openai"], help="mock = heuristic stand-in, not a language model. ollama/gemini/groq/openrouter/requesty/grok/cohere have free tiers or free credits.")
    DEFAULT_IDS = {"mock": cfg["model"], "gemini": "gemini-3.5-flash-lite", "ollama": "qwen2.5:7b"}   # gemini: the model of the paper
    model = st.text_input("Model id", DEFAULT_IDS.get(provider, ""), help="Exact id from the provider console (for Gemini: Google AI Studio).")
    rpm = st.number_input("Requests per minute cap", 0, 120, int(PRESETS.get(provider, {}).get("rpm", 0) or 0), help="Keeps free-tier providers from answering HTTP 429.") if provider != "mock" else 0
    st.markdown('<div class="lbl" style="margin-top:1rem">Repair policy</div>', unsafe_allow_html=True)
    level = st.selectbox("Feedback detail", ["evidence", "localized", "binary", "prescriptive"], help="prescriptive = the verifier also computes the detour (condition C_prescriptive)")
    mode = st.selectbox("Repair mode", ["targeted", "regenerate"])
    guard = st.checkbox("Regression guard", True)
    iters = st.slider("Max repair rounds", 1, 5, 3)
    seed = st.number_input("Seed", 0, 10_000, 0)

if provider == "mock":
    st.markdown('<div class="mockbar">MOCK PLANNER: scripted heuristics, not a language model. Output demonstrates the pipeline only.</div>', unsafe_allow_html=True)

sets = {"synthetic benchmark": "benchmark/generated", "airport-anchored (OurAirports, modeled zones)": "benchmark/real",
        "Bengaluru 3D (OSM buildings + SRTM terrain)": "benchmark/bengaluru"}
avail = {k: load_scenarios(v) for k, v in sets.items()}
avail = {k: v for k, v in avail.items() if v}
if not avail:
    st.warning("No scenarios found. Run `python run.py generate` (synthetic) and/or `python run.py real` (real airports).")
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown('<div class="lbl">Scenario</div>', unsafe_allow_html=True)
    src = st.selectbox("Source", list(avail))
    sid = st.selectbox("Scenario id", [s["scenario_id"] for s in avail[src]])
    sc = next(x for x in avail[src] if x["scenario_id"] == sid)
    meta = sc["metadata"]
    st.markdown(f'<div class="mono" style="font-size:.78rem">difficulty: {sc["difficulty"]}<br>zones: {meta["n_zones"]} &nbsp; required: {meta["n_required"]}<br>'
                f'feasibility: {meta["feasibility"]}<br>source: {meta.get("source","synthetic")}</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="lbl">Mission as given to the planner</div>', unsafe_allow_html=True)
    m = Mission.from_dict(sc["mission"])
    m.description = st.text_area("Mission text", m.description, height=150, label_visibility="collapsed")

go = st.button("Run mission")
if go:
    log_box = st.empty()
    bar = st.progress(0, text="starting the run…")
    lines = []

    def on_event(e):
        t = e["type"]
        if t == "planning":
            lines.append(e["msg"])
        elif t == "verified":
            lines.append(f"verified round {e['iteration']}: {e['score']} violation(s)" if e.get("parsed", True) else "plan could not be parsed")
        elif t == "repair_start":
            lines.append(f"repair round {e['iteration']}/{e['of']}: asking the planner to fix it")
        elif t == "rate_limit":
            lines.append(f"provider is rate limiting ({e['why']}); waiting {e['wait']:.0f}s, retry {e['retry']}")
        elif t == "feasibility":
            lines.append(f"feasibility check: {e['status']}")
        log_box.markdown("<div class='mono' style='font-size:.78rem;border:1px solid #cfc8b8;padding:.5rem;background:#faf8f2'>" + "<br>".join(lines[-8:]) + "</div>", unsafe_allow_html=True)
        pct = {"planning": 10, "feasibility": 20, "repair_start": None, "verified": None, "rate_limit": None}.get(t)
        if t == "verified":
            k = int(e.get("iteration") or 0)
            pct = 30 if k == 0 else 40 + int(45 * k / max(iters, 1))
        elif t == "repair_start":
            k = int(e.get("iteration") or 1)
            pct = 40 + int(45 * (k - 1) / max(iters, 1))
        if pct is not None:
            bar.progress(min(pct, 95), text=lines[-1] if lines else "running…")

    pc = dict(cfg, provider=provider, model=model or cfg["model"], requests_per_minute=rpm or None)
    with st.spinner("Running mission — planning, verifying, repairing…"):
        try:
            r = run_session(m, pc, LoopConfig(name="ui", feedback_level=level, repair_mode=mode, regression_guard=guard, max_iterations=iters), int(seed), on_event=on_event)
        except LLMError as e:
            st.error(f"The provider refused or failed: {e}\n\nThis is a service limit, not a planning result. Lower the requests-per-minute cap, wait for the quota reset, or use Ollama locally.")
            st.stop()
        except Exception as e:  # noqa
            st.error(f"Run failed: {type(e).__name__}: {e}")
            st.stop()
    log_box.empty()
    bar.progress(100, text="run complete")
    fv = r["final_verification"]
    color = COLORS.get(r["status"], INK)
    st.markdown(f'<div style="--c:{color}" class="stamp mono">{r["status"]}</div> &nbsp; <span class="mono" style="font-size:.8rem">{r["provider"]} / {r["model"]} &middot; '
                f'{r["n_llm_calls"]} model call(s) &middot; {r["wall_time_s"]:.1f}s{" &middot; MOCK" if r["is_mock"] else ""}</span>', unsafe_allow_html=True)
    if fv:
        a, b, c, d = st.columns(4)
        a.metric("Constraints met", "yes" if fv["valid"] else "no")
        b.metric("Mission intact", "yes" if fv["mission_preserved"] else "no")
        c.metric("Route length", f"{fv['route_distance']:.0f} / {m.max_distance:g}")
        d.metric("Open violations", fv["score"])
    t1, t2, t3, t4 = st.tabs(["Revision strips", "Map history", "Plans and feedback", "Final plan"])
    with t1:
        for it in r["iterations"]:
            col = COLORS.get(it["outcome"], INK)
            det = "parse failure" if it["verification"] is None else f'{it["score"]} violation(s)' + ("  [regression]" if it["regression"] else "")
            dist = f'{it["verification"]["route_distance"]:.0f}' if it["verification"] else "-"
            st.markdown(f'<div class="strip" style="--c:{col}"><div>R{it["iteration"]}</div><div><span class="stamp" style="--c:{col}">{it["outcome"]}</span></div>'
                        f'<div>{det}</div><div>len {dist}</div></div>', unsafe_allow_html=True)
            if it["verification"]:
                for v in it["verification"]["violations"] + it["verification"]["preservation_violations"]:
                    st.markdown(f'<div class="vio"><b>{v["type"]}</b> at {v["location"]}: {v["details"]}</div>', unsafe_allow_html=True)
                if it["verification"]["success"]:
                    st.markdown('<div class="ok">All constraints satisfied and all required waypoints kept.</div>', unsafe_allow_html=True)
    with t2:
        st.pyplot(r["history_fig"])
    with t3:
        for it in r["iterations"]:
            with st.expander(f'Round {it["iteration"]}: {it["outcome"]}'):
                if it["feedback"]:
                    st.markdown('<div class="lbl">Feedback sent to the planner (built from verifier output only)</div>', unsafe_allow_html=True)
                    st.json(it["feedback"]["payload"])
                st.markdown('<div class="lbl">Plan</div>', unsafe_allow_html=True)
                st.json(it["plan"])
                st.caption(f'plan diff: {it["diff"]}')
                if it["error"]:
                    st.warning(it["error"])
    with t4:
        st.json(r["final_plan"])
        st.download_button("Download session (JSON)", json.dumps({k: v for k, v in r.items() if k != "history_fig"}, default=str), "session.json")


st.markdown("---")
st.markdown('<div class="lbl">Mid-flight change (robustness)</div>', unsafe_allow_html=True)
with st.expander("Fly part of a valid plan, change the environment, repair what is left"):
    ch_label = {"add_zone": "A new restricted area appears on the route", "enlarge_zone": "An existing zone is enlarged",
                "tighten_distance": "Remaining distance allowance is cut", "reduce_energy": "Battery drops"}
    change = st.selectbox("Change", list(ch_label), format_func=lambda k: ch_label[k])
    prog = st.slider("Share of the route already flown", 0.1, 0.9, 0.4)
    if st.button("Apply change and re-plan"):
        pc2 = dict(cfg, provider=provider, model=model or cfg["model"], requests_per_minute=rpm or None)
        try:
            d = run_dynamic_session(m, change, float(prog), pc2, LoopConfig(name="ui-dyn", feedback_level=level, repair_mode=mode, regression_guard=guard, max_iterations=iters), int(seed))
        except LLMError as e:
            st.error(f"The provider refused or failed: {e}")
            st.stop()
        if d is None:
            st.info("This change cannot be applied to this scenario (for example no zone to enlarge, or no slack left to remove).")
        else:
            st.markdown(f'<div class="mono" style="font-size:.78rem;border:1px solid #cfc8b8;padding:.5rem;background:#faf8f2">{d["description"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="--c:{COLORS.get(d["status"], INK)}" class="stamp mono">{d["status"]}</div> &nbsp; <span class="mono" style="font-size:.8rem">{d["n_llm_calls"]} model call(s)</span>', unsafe_allow_html=True)
            st.json({"state at the moment of change": d["state"], "change": d["info"], "feasibility after change": d["feasibility"]})
            st.pyplot(d["history_fig"])
