import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import subprocess
import tempfile
import os
import shutil

st.set_page_config(
    page_title="ML Branch Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0a0a0f; color: #e2e8f0; }
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem; font-weight: 700; color: #00ff88;
        letter-spacing: -1px; margin-bottom: 0;
        text-shadow: 0 0 30px rgba(0,255,136,0.3);
    }
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; color: #4a5568;
        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2rem;
    }
    .metric-card {
        background: #111118; border: 1px solid #1e2030;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        text-align: center; position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 2px; background: linear-gradient(90deg, #00ff88, #0088ff);
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem; font-weight: 700; color: #00ff88;
    }
    .metric-label {
        font-size: 0.75rem; color: #4a5568;
        text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px;
    }
    .section-title {
        font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
        color: #00ff88; text-transform: uppercase; letter-spacing: 3px;
        margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1e2030;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #0088ff 100%);
        color: #0a0a0f; border: none; border-radius: 8px;
        font-family: 'JetBrains Mono', monospace; font-weight: 700;
        font-size: 0.85rem; letter-spacing: 1px;
        padding: 0.6rem 1.5rem; width: 100%;
    }
    .result-box {
        background: #111118; border: 1px solid #1e2030; border-radius: 12px;
        padding: 1.5rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; color: #a0aec0; line-height: 1.8;
    }
    .highlight-green { color: #00ff88; font-weight: 700; }
    .highlight-blue  { color: #0088ff; font-weight: 700; }
    .highlight-orange{ color: #ff8800; font-weight: 700; }
    div[data-testid="stSidebar"] { background: #080810; border-right: 1px solid #1e2030; }
    .stTabs [data-baseweb="tab-list"] { background: #111118; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; letter-spacing: 1px; color: #4a5568; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #1e2030 !important; color: #00ff88 !important; }
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00ff88, #0088ff);
        color: #0a0a0f; font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 700; padding: 2px 10px;
        border-radius: 20px; letter-spacing: 1px;
    }
    .log-box {
        background: #080810; border: 1px solid #1e2030; border-radius: 8px;
        padding: 1rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; color: #4a5568; line-height: 1.6;
        max-height: 200px; overflow-y: auto;
    }
    .stTextArea textarea {
        background: #080810 !important; color: #00ff88 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important; border: 1px solid #1e2030 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────
DARK   = '#0a0a0f'
CARD   = '#111118'
GREEN  = '#00ff88'
BLUE   = '#0088ff'
ORANGE = '#ff8800'
GRID   = '#1e2030'

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK, plot_bgcolor=CARD,
    font=dict(family='JetBrains Mono', color='#a0aec0', size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)

DEFAULT_CPP = """\
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

int binarySearch(vector<int>& arr, int target) {
    int lo = 0, hi = arr.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int main() {
    int N = 2000;
    vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = rand() % 10000;
    bubbleSort(data);
    for (int i = 0; i < 30000; i++)
        binarySearch(data, rand() % 10000);
    cout << "Done\\n";
    return 0;
}
"""

# ── helpers ────────────────────────────────────────────────────────────────────

def get_pin_root():
    pin = os.environ.get("PIN_ROOT", "")
    if pin and os.path.isdir(pin):
        return pin
    home = os.path.expanduser("~")
    for d in os.listdir(home):
        if d.startswith("pin-") and os.path.isdir(os.path.join(home, d)):
            return os.path.join(home, d)
    return ""


def compile_and_trace(cpp_code, pin_root):
    workdir    = tempfile.mkdtemp(prefix="bpred_")
    logs       = []
    src_path   = os.path.join(workdir, "target.cpp")
    bin_path   = os.path.join(workdir, "target")
    csv_path   = os.path.join(workdir, "branch_data.csv")
    pintool_so = os.path.join(pin_root, "source/tools/MyPinTool/obj-intel64/BranchDataGen.so")
    pin_bin    = os.path.join(pin_root, "pin")

    with open(src_path, "w") as f:
        f.write(cpp_code)
    logs.append("✓ Source written")

    r = subprocess.run(["g++", src_path, "-o", bin_path, "-O2"], capture_output=True, text=True)
    if r.returncode != 0:
        return False, "\n".join(logs) + "\n✗ Compile error:\n" + r.stderr, ""
    logs.append("✓ Compiled successfully")

    if not os.path.isfile(pintool_so):
        return False, "\n".join(logs) + f"\n✗ BranchDataGen.so not found at:\n  {pintool_so}", ""
    logs.append("✓ Pin tool found")

    r = subprocess.run(
        [pin_bin, "-t", pintool_so, "--", bin_path],
        capture_output=True, text=True, cwd=workdir, timeout=600
    )
    logs.append(r.stdout.strip())
    if not os.path.isfile(csv_path):
        return False, "\n".join(logs) + "\n✗ branch_data.csv not created.\n" + r.stderr, ""
    logs.append("✓ Trace generated")
    return True, "\n".join(logs), csv_path


def load_csv(path_or_buf, max_rows=300_000):
    df = pd.read_csv(path_or_buf)
    df.columns = [c.strip() for c in df.columns]

    def to_int(v):
        try:
            return int(str(v), 16) if str(v).startswith("0x") else int(v)
        except:
            return 0

    df["PC_int"]     = df["PC"].apply(to_int)
    df["Target_int"] = df["Target"].apply(to_int)
    df["Taken"]      = pd.to_numeric(df["Taken"], errors="coerce").fillna(0).astype(int)
    df["History_1"]  = df.groupby("PC_int")["Taken"].shift(1).fillna(0)
    df["History_2"]  = df.groupby("PC_int")["Taken"].shift(2).fillna(0)
    df["taken_mean"] = df.groupby("PC_int")["Taken"].transform("mean")
    return df


def run_ml_predictor(df):
    history = defaultdict(lambda: {"last": 0.0, "second": 0.0})
    correct, rows = 0, []
    for _, r in df.iterrows():
        pc = r["PC_int"]; actual = int(r["Taken"]); h = history[pc]
        score = 0.4*h["last"] + 0.2*h["second"] + 0.3*r["taken_mean"] + 0.1*(1 if pc%1024<512 else 0)
        pred = 1 if score >= 0.5 else 0
        ok = pred == actual; correct += ok
        rows.append({"predicted": pred, "actual": actual, "correct": ok, "pc": pc})
        history[pc]["second"] = history[pc]["last"]
        history[pc]["last"] = float(actual)
    return correct / len(df) * 100, pd.DataFrame(rows)


def run_2bit_predictor(df):
    counters = defaultdict(lambda: 2)
    correct, rows = 0, []
    for _, r in df.iterrows():
        pc = r["PC_int"]; actual = int(r["Taken"])
        pred = 1 if counters[pc] >= 2 else 0
        ok = pred == actual; correct += ok
        rows.append({"predicted": pred, "actual": actual, "correct": ok, "pc": pc})
        counters[pc] = min(3, counters[pc]+1) if actual==1 else max(0, counters[pc]-1)
    return correct / len(df) * 100, pd.DataFrame(rows)


def run_all(df):
    ml_acc,  ml_preds  = run_ml_predictor(df)
    tbt_acc, tbt_preds = run_2bit_predictor(df)
    st.session_state.update({
        "df": df, "ml_preds": ml_preds, "tbt_preds": tbt_preds,
        "ml_acc": ml_acc, "tbt_acc": tbt_acc, "ready": True
    })


def result_box(ml_acc, tbt_acc, n_rows):
    diff   = abs(ml_acc - tbt_acc)
    winner = "ML" if ml_acc >= tbt_acc else "2-bit"
    st.markdown(f"""
    <div class="result-box">
        ┌──────────────────────────────────┐<br>
        │  SIMULATION COMPLETE             │<br>
        ├──────────────────────────────────┤<br>
        │  Branches : <span class="highlight-green">{n_rows:,}</span><br>
        │  ML acc   : <span class="highlight-green">{ml_acc:.2f}%</span><br>
        │  2-bit acc: <span class="highlight-blue">{tbt_acc:.2f}%</span><br>
        │  Diff     : <span class="highlight-orange">{diff:.2f}%</span><br>
        │  Winner   : <span class="highlight-green">{winner}</span><br>
        └──────────────────────────────────┘
    </div>""", unsafe_allow_html=True)
    fig = go.Figure()
    for val, name, color in [(ml_acc,"ML Model",GREEN),(tbt_acc,"2-bit",BLUE)]:
        fig.add_trace(go.Bar(x=[name], y=[val], marker_color=color, width=0.35,
            text=[f"{val:.2f}%"], textposition="outside",
            textfont=dict(family="JetBrains Mono", color=color, size=13)))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
        yaxis=dict(range=[70,100],gridcolor=GRID), showlegend=False, bargap=0.5)
    st.plotly_chart(fig, use_container_width=True)


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">// Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🖥️  Run C++ Code", "📁  Upload CSV", "📊  Visualizations"],
                label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-title">// PIN_ROOT</div>', unsafe_allow_html=True)
    pin_root_input = st.text_input("Pin path", value=get_pin_root(),
                                   help="e.g. /home/user/pin-3.28-...-gcc-linux")
    if pin_root_input:
        st.session_state["pin_root"] = pin_root_input
    st.markdown("---")
    if st.session_state.get("ready"):
        ml_acc  = st.session_state["ml_acc"]
        tbt_acc = st.session_state["tbt_acc"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;line-height:1.9;">
        <span style="color:#4a5568;">// Last run</span><br>
        ML  : <span style="color:{GREEN}">{ml_acc:.2f}%</span><br>
        2bit: <span style="color:{BLUE}">{tbt_acc:.2f}%</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Run C++ Code
# ══════════════════════════════════════════════════════════════════════════════
if "C++" in page:
    st.markdown('<div class="main-header">ML Branch Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// Paste C++ → compile → trace → predict</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="section-title">// Your C++ code</div>', unsafe_allow_html=True)
        cpp_code = st.text_area("", value=DEFAULT_CPP, height=430, label_visibility="collapsed")
        run_btn  = st.button("▶  COMPILE → TRACE → PREDICT")

    with col_right:
        st.markdown('<div class="section-title">// Pipeline log</div>', unsafe_allow_html=True)
        log_slot    = st.empty()
        result_slot = st.empty()

    if run_btn:
        pin_root = st.session_state.get("pin_root", "")
        if not pin_root or not os.path.isdir(pin_root):
            st.error("Set PIN_ROOT correctly in the sidebar first.")
            st.stop()

        log_slot.markdown('<div class="log-box">Compiling...</div>', unsafe_allow_html=True)

        with st.spinner("Running full pipeline..."):
            ok, log, csv_path = compile_and_trace(cpp_code, pin_root)

        log_slot.markdown(
            f'<div class="log-box">{log.replace(chr(10),"<br>")}</div>',
            unsafe_allow_html=True)

        if not ok:
            result_slot.error("Pipeline failed — see log.")
            st.stop()

        df = load_csv(csv_path)
        shutil.rmtree(os.path.dirname(csv_path), ignore_errors=True)

        import time
        import threading

        start = time.time()
        timer_slot = st.empty()
        done = threading.Event()

        def update_timer():
            while not done.is_set():
                elapsed = time.time() - start
                timer_slot.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#4a5568; padding:0.5rem 0;">
                ⏱ Running... <span style="color:#00ff88;">{elapsed:.1f}s</span>
                </div>""", unsafe_allow_html=True)
                time.sleep(0.1)

        t = threading.Thread(target=update_timer)
        t.start()

        with st.spinner("Running predictors..."):
            run_all(df)

        done.set()
        t.join()

        elapsed = time.time() - start
        timer_slot.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#4a5568; padding:0.5rem 0;">
        ✓ Completed in <span style="color:#00ff88;">{elapsed:.1f}s</span>
        </div>""", unsafe_allow_html=True)

        with result_slot:
            result_box(st.session_state["ml_acc"], st.session_state["tbt_acc"], len(df))

        st.success("✓ Done! Switch to Visualizations or ML vs 2-bit in the sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Upload CSV
# ══════════════════════════════════════════════════════════════════════════════
elif "Upload" in page:
    st.markdown('<div class="main-header">Upload Trace</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// Upload any branch_data.csv from Intel Pin</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your branch_data.csv here", type=["csv"])
    if uploaded:
        with st.spinner("Loading..."):
            df = load_csv(uploaded)

        c1, c2, c3 = st.columns(3)
        for col, val, label in zip([c1,c2,c3],
            [f"{len(df):,}", f"{df['PC_int'].nunique():,}", f"{df['Taken'].mean()*100:.1f}%"],
            ["Total branches","Unique PCs","Branches taken"]):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df[["PC","Target","Taken"]].head(10), use_container_width=True, hide_index=True)

        if st.button("▶  RUN BOTH PREDICTORS"):
            import time
            start = time.time()
            timer_slot = st.empty()
            
            with st.spinner("Running predictors..."):
                import threading
                done = threading.Event()
                
                def update_timer():
                    while not done.is_set():
                        elapsed = time.time() - start
                        timer_slot.markdown(f"""
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#4a5568; padding:0.5rem 0;">
                        ⏱ Running... <span style="color:#00ff88;">{elapsed:.1f}s</span>
                        </div>""", unsafe_allow_html=True)
                        time.sleep(0.1)
                
                t = threading.Thread(target=update_timer)
                t.start()
                run_all(df)
                done.set()
                t.join()
            
            elapsed = time.time() - start
            timer_slot.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#4a5568; padding:0.5rem 0;">
            ✓ Completed in <span style="color:#00ff88;">{elapsed:.1f}s</span>
            </div>""", unsafe_allow_html=True)
            
            result_box(st.session_state["ml_acc"], st.session_state["tbt_acc"], len(df))
            st.success("✓ Done! Switch to Visualizations in the sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
elif "Visual" in page:
    st.markdown('<div class="main-header">Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">// Branch behavior analysis</div>', unsafe_allow_html=True)

    if not st.session_state.get("ready"):
        st.warning("Run a prediction first using **Run C++ Code** or **Upload CSV**.")
        st.stop()

    df        = st.session_state["df"]
    ml_preds  = st.session_state["ml_preds"]
    tbt_preds = st.session_state["tbt_preds"]

    tab1, tab2, tab3, tab4 = st.tabs(["Rolling Accuracy", "Distribution", "Hot Branches", "Taken Rate"])

    with tab1:
        window = st.slider("Rolling window", 500, 20000, 5000, step=500)
        ml_r   = ml_preds["correct"].rolling(window).mean() * 100
        tbt_r  = tbt_preds["correct"].rolling(window).mean() * 100
        
        # Downsample to max 10,000 points for browser performance
        max_points = 10000
        step = max(1, len(ml_r) // max_points)
        ml_r_plot  = ml_r.iloc[::step]
        tbt_r_plot = tbt_r.iloc[::step]
        x = list(range(len(ml_r_plot)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ml_r_plot, name="ML Model",
            line=dict(color=GREEN, width=1.5), fill="tozeroy", fillcolor="rgba(0,255,136,0.05)"))
        fig.add_trace(go.Scatter(x=x, y=tbt_r_plot, name="2-bit",
            line=dict(color=BLUE, width=1.5, dash="dot")))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
            title=dict(text="Accuracy over trace (downsampled for performance)", font=dict(color=GREEN, size=12)),
            yaxis=dict(range=[40,100],gridcolor=GRID),
            legend=dict(bgcolor="#111118",bordercolor=GRID))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        taken = int(df["Taken"].sum()); not_t = len(df)-taken
        fig = go.Figure(go.Pie(
            labels=["Taken","Not Taken"], values=[taken, not_t], hole=0.65,
            marker=dict(colors=[GREEN,BLUE], line=dict(color=DARK,width=2)),
            textfont=dict(family="JetBrains Mono",size=12), textinfo="label+percent"))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
            title=dict(text="Branch direction distribution",font=dict(color=GREEN,size=12)),
            showlegend=False,
            annotations=[dict(text=f"{taken/len(df)*100:.1f}%<br>taken",
                x=0.5,y=0.5,showarrow=False,
                font=dict(size=16,color=GREEN,family="JetBrains Mono"))])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        pc_counts = df.groupby("PC_int")["Taken"].count().sort_values(ascending=False).head(15)
        pc_counts.index = [hex(i) for i in pc_counts.index]
        table_df = pd.DataFrame({
            "Branch Address (PC)": pc_counts.index,
            "Execution Count": pc_counts.values,
            "% of Total": (pc_counts.values / len(df) * 100).round(2)
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    with tab4:
        pc_taken = df.groupby("PC_int")["Taken"].agg(["mean","count"])
        pc_taken = pc_taken[pc_taken["count"] > 50].sort_values("count", ascending=False).head(20)
        pc_taken.index = [hex(i) for i in pc_taken.index]
        table_df2 = pd.DataFrame({
            "Branch Address (PC)": pc_taken.index,
            "Times Executed": pc_taken["count"].values,
            "Taken Rate (%)": (pc_taken["mean"] * 100).round(2).values,
            "Behavior": ["Biased Taken" if x > 70 else "Biased Not Taken" if x < 30 else "Unpredictable" for x in (pc_taken["mean"]*100)]
        })
        st.dataframe(table_df2, use_container_width=True, hide_index=True)


