import streamlit as st

st.set_page_config(
    page_title="TriAuth — Multimodal Authentication",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    padding: 24px 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.06);
}
.metric-card .value { font-size: 2.2rem; font-weight: 700; margin: 0; }
.metric-card .label { font-size: 0.95rem; margin: 4px 0 0; }
.metric-card .sub   { font-size: 0.78rem; margin: 2px 0 0; opacity: 0.8; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; margin-right: 4px;
}
.section-header {
    font-size: 1.15rem; font-weight: 700; margin: 0 0 10px;
    padding-bottom: 6px; border-bottom: 2px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px'>
  <h1 style='margin:0;font-size:2rem'>🔐 TriAuth</h1>
  <p style='margin:4px 0 0;font-size:1.1rem;color:#6b7280'>
    Trimodal biometric authentication &nbsp;·&nbsp;
    <span style='color:#1a56db'>Face</span> &nbsp;·&nbsp;
    <span style='color:#7e3af2'>Voice</span> &nbsp;·&nbsp;
    <span style='color:#057a55'>Keystroke</span>
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='metric-card' style='background:#f0fdf4;border-color:#bbf7d0'>
        <p class='value' style='color:#15803d'>97.89%</p>
        <p class='label' style='color:#166534'>Best accuracy</p>
        <p class='sub'   style='color:#4ade80'>Trimodal majority vote</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card' style='background:#eff6ff;border-color:#bfdbfe'>
        <p class='value' style='color:#1d4ed8'>0.9971</p>
        <p class='label' style='color:#1e40af'>Best AUC-ROC</p>
        <p class='sub'   style='color:#60a5fa'>Hybrid trimodal fusion</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card' style='background:#faf5ff;border-color:#e9d5ff'>
        <p class='value' style='color:#6d28d9'>0% FAR</p>
        <p class='label' style='color:#5b21b6'>False Accept Rate</p>
        <p class='sub'   style='color:#a78bfa'>Trimodal majority vote</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column content ────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("<p class='section-header'>How it works</p>", unsafe_allow_html=True)
    st.markdown("""
    TriAuth verifies whether a user is **genuine or an attacker** by combining three
    independent anti-spoofing models. Each modality votes independently — majority voting
    fuses the decisions for a final verdict.

    | Modality | Model | Accuracy |
    |:---|:---|---:|
    | 👤 Face liveness | ResNet18 binary classifier | 89.47% |
    | 🎤 Voice anti-spoofing | CNN binary classifier | 83.16% |
    | ⌨️ Keystroke dynamics | XGBoost classifier | 95.74% |

    Fusing all three with **majority voting** achieves **97.89% accuracy with 0% FAR** —
    the system never grants access to an attacker.
    """)

with right:
    st.markdown("<p class='section-header'>Attack scenarios tested</p>", unsafe_allow_html=True)
    st.markdown("""
    <span class='badge' style='background:#fee2e2;color:#991b1b'>TTS</span> Text-to-speech voice synthesis<br><br>
    <span class='badge' style='background:#fef3c7;color:#92400e'>Replay</span> Recorded & replayed audio/video<br><br>
    <span class='badge' style='background:#ede9fe;color:#5b21b6'>Logical</span> Deepfake / face swap attacks<br><br>
    <span class='badge' style='background:#dcfce7;color:#166534'>Synthetic</span> Perturbed keystroke timing
    """, unsafe_allow_html=True)

    st.markdown("<br><p class='section-header'>Fusion strategies</p>", unsafe_allow_html=True)
    st.markdown("""
    - **Score-level** — weighted average of spoof probabilities
    - **Feature-level** — Random Forest / Logistic Regression meta-classifier
    - **Decision-level** — majority vote (best FAR: 0%)
    - **Hybrid** — 3-stage cascade (best AUC-ROC: 0.9971)
    """)

st.markdown("---")
st.caption("👈 Navigate to **Live Demo** to run authentication or **Results** to explore model performance.")
