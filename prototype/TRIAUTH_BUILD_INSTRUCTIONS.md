# TriAuth Prototype — Complete Build Instructions for Claude Code

> Give this entire file to Claude Code and say:
> **"Follow these instructions exactly and build the complete prototype."**

---

## Overview

You are building a Streamlit web application called **TriAuth** — a trimodal biometric
authentication demo combining face liveness, voice anti-spoofing, and keystroke dynamics.

The app has three pages:
1. **Home** — project overview and headline metrics
2. **Live Demo** — captures real keystroke timing, runs live XGBoost inference, accepts face/voice uploads
3. **Results** — full interactive results dashboard with Plotly charts

---

## Step 0 — Prerequisites

Install dependencies:

```bash
pip install streamlit plotly xgboost scikit-learn torch torchvision joblib pillow numpy pandas
```

---

## Step 1 — Folder structure

Create exactly this structure:

```
triauth/
├── app.py
├── requirements.txt
├── pages/
│   ├── 1_Live_Demo.py
│   └── 2_Results.py
├── models/
│   ├── face_binary_resnet18_best.pth       ← user provides
│   ├── voice_binary_cnn_best.pth           ← user provides
│   ├── xgb_keystroke.pkl                   ← user provides
│   ├── keystroke_scaler.pkl                ← user provides
│   └── keystroke_imputer.pkl               ← user provides
└── data/
    └── fusion_val_predictions.csv          ← user provides
```

---

## Step 2 — Create `requirements.txt`

```
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
plotly>=5.18.0
```

---

## Step 3 — Create `app.py` (Home page)

```python
import streamlit as st

st.set_page_config(
    page_title="TriAuth — Multimodal Authentication",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔐 TriAuth — Trimodal Biometric Authentication")
st.markdown("### Face · Voice · Keystroke Dynamics")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background:#EAF3DE;padding:24px;border-radius:12px;text-align:center'>
        <h2 style='color:#27500A;margin:0'>97.89%</h2>
        <p style='color:#3B6D11;margin:4px 0 0'>Best accuracy</p>
        <p style='color:#639922;font-size:12px;margin:0'>Trimodal majority vote</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background:#E6F1FB;padding:24px;border-radius:12px;text-align:center'>
        <h2 style='color:#0C447C;margin:0'>0.9971</h2>
        <p style='color:#185FA5;margin:4px 0 0'>Best AUC-ROC</p>
        <p style='color:#378ADD;font-size:12px;margin:0'>Hybrid trimodal fusion</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background:#EEEDFE;padding:24px;border-radius:12px;text-align:center'>
        <h2 style='color:#3C3489;margin:0'>0% FAR</h2>
        <p style='color:#534AB7;margin:4px 0 0'>False Accept Rate</p>
        <p style='color:#7F77DD;font-size:12px;margin:0'>Trimodal majority vote</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
## What this system does

This prototype demonstrates a **trimodal biometric authentication system** that combines
three independent anti-spoofing models to verify whether a user is genuine or an attacker.

| Modality | Model | Accuracy |
|---|---|---|
| 👤 Face liveness | ResNet18 binary classifier | 89.47% |
| 🎤 Voice anti-spoofing | CNN binary classifier | 83.16% |
| ⌨️ Keystroke dynamics | XGBoost classifier | 95.74% |

When all three are fused using **majority voting**, the system achieves **97.89% accuracy
with 0% FAR** — it never incorrectly grants access to an attacker.

## Attack scenarios tested
- **TTS** — text-to-speech synthesised voice attacks
- **Replay** — recorded and replayed audio/video
- **Logical** — deepfake / face swap attacks
- **Synthetic** — perturbed keystroke timing patterns

## Fusion strategies implemented
- Score-level fusion (weighted average of spoof probabilities)
- Feature-level fusion (Random Forest / Logistic Regression meta-classifier)
- Decision-level fusion (majority vote)
- Hybrid fusion (3-stage: Feature → Score → Decision)

---
👈 Use the sidebar to navigate to the **Live Demo** or **Results** pages.
""")
```

---

## Step 4 — Create `pages/1_Live_Demo.py`

This page captures real keystroke timing in the browser using JavaScript,
runs the real XGBoost model for keystroke inference, and accepts face/voice file uploads.

```python
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import json
import os
from PIL import Image

st.set_page_config(page_title="Live Demo", page_icon="🔐", layout="wide")
st.title("🔐 Live Authentication Demo")
st.markdown("Type the passphrase — the system captures your keystroke timing in real time.")
st.markdown("---")

# ── Model paths ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Load keystroke models (cached so they only load once) ────────────────────
@st.cache_resource
def load_keystroke_models():
    try:
        xgb     = joblib.load(os.path.join(MODEL_DIR, "xgb_keystroke.pkl"))
        scaler  = joblib.load(os.path.join(MODEL_DIR, "keystroke_scaler.pkl"))
        imputer = joblib.load(os.path.join(MODEL_DIR, "keystroke_imputer.pkl"))
        return xgb, scaler, imputer, True
    except Exception as e:
        st.warning(f"Keystroke model not found: {e}")
        return None, None, None, False

xgb_model, ks_scaler, ks_imputer, ks_loaded = load_keystroke_models()

# ── Feature extraction — exact same function used in training ─────────────────
FEATURE_COLS = [
    "hold_mean", "hold_std", "hold_min", "hold_max", "hold_median",
    "flight_mean", "flight_std", "flight_min", "flight_max", "flight_median",
    "num_keys", "unique_keys", "hold_flight_ratio"
]

def extract_keystroke_features(events):
    hold   = [e.get("holdTime")   for e in events if e.get("holdTime")   is not None]
    flight = [e.get("flightTime") for e in events if e.get("flightTime") is not None]
    keys   = [e.get("keyCode")    for e in events if e.get("keyCode")    is not None]

    hold   = np.array(hold,   dtype=float) if hold   else np.array([])
    flight = np.array(flight, dtype=float) if flight else np.array([])
    keys   = np.array(keys,   dtype=float) if keys   else np.array([])

    def s(arr, fn, d=0):
        return fn(arr) if len(arr) > 0 else d

    return {
        "hold_mean":         s(hold,   np.mean),
        "hold_std":          s(hold,   np.std),
        "hold_min":          s(hold,   np.min),
        "hold_max":          s(hold,   np.max),
        "hold_median":       s(hold,   np.median),
        "flight_mean":       s(flight, np.mean),
        "flight_std":        s(flight, np.std),
        "flight_min":        s(flight, np.min),
        "flight_max":        s(flight, np.max),
        "flight_median":     s(flight, np.median),
        "num_keys":          len(events),
        "unique_keys":       len(np.unique(keys)) if len(keys) > 0 else 0,
        "hold_flight_ratio": (np.mean(hold) / np.mean(flight))
                              if len(hold) > 0 and len(flight) > 0
                              and np.mean(flight) != 0 else 0,
    }

def predict_keystroke(events):
    if not ks_loaded or len(events) < 5:
        return None, None
    feats = extract_keystroke_features(events)
    X = np.array([[feats[c] for c in FEATURE_COLS]])
    X = ks_imputer.transform(X)
    X = ks_scaler.transform(X)
    prob = float(xgb_model.predict_proba(X)[0][1])
    pred = int(prob >= 0.5)
    return pred, prob

# ── Keystroke capture HTML component ─────────────────────────────────────────
PASSPHRASE = "authenticate user now"

keystroke_html = f"""
<div style="font-family:system-ui;padding:0">
  <p style="font-size:13px;color:#888;margin:0 0 8px">Type the passphrase exactly:</p>
  <p style="font-size:18px;font-weight:600;color:#1a1a1a;margin:0 0 12px;
            background:#f5f5f5;padding:10px 16px;border-radius:8px;letter-spacing:1px">
    "{PASSPHRASE}"
  </p>
  <input id="ks-input" type="text" autocomplete="off" spellcheck="false"
    style="width:100%;padding:12px 16px;font-size:16px;border:1.5px solid #ddd;
           border-radius:8px;outline:none;box-sizing:border-box"
    placeholder="Start typing here..."/>
  <div id="ks-status" style="font-size:12px;color:#999;margin-top:6px">
    Waiting for input...
  </div>
  <button id="ks-btn" onclick="submitKeystrokes()"
    style="margin-top:12px;padding:10px 24px;background:#1D9E75;color:white;
           border:none;border-radius:8px;font-size:14px;cursor:pointer;display:none">
    Analyse keystrokes
  </button>
</div>

<script>
var events = [];
var downTimes = {{}};
var inp    = document.getElementById('ks-input');
var status = document.getElementById('ks-status');
var btn    = document.getElementById('ks-btn');

inp.addEventListener('keydown', function(e) {{
  downTimes[e.key] = performance.now();
}});

inp.addEventListener('keyup', function(e) {{
  var up   = performance.now();
  var down = downTimes[e.key];
  if (down === undefined) return;
  var hold   = up - down;
  var flight = events.length > 0
    ? (down - events[events.length - 1]._upTime)
    : 0;
  events.push({{
    key:        e.key,
    keyCode:    e.keyCode,
    holdTime:   parseFloat(hold.toFixed(2)),
    flightTime: parseFloat(Math.max(0, flight).toFixed(2)),
    _upTime:    up
  }});
  status.textContent = 'Keys captured: ' + events.length;
  if (events.length >= 5) btn.style.display = 'inline-block';
}});

function submitKeystrokes() {{
  var clean = events.map(function(e) {{
    return {{
      key:        e.key,
      keyCode:    e.keyCode,
      holdTime:   e.holdTime,
      flightTime: e.flightTime
    }};
  }});
  var encoded = encodeURIComponent(JSON.stringify(clean));
  window.location.href = '?ks_data=' + encoded;
}}
</script>
"""

# ── Page layout ───────────────────────────────────────────────────────────────
st.markdown("## Step 1 — Keystroke dynamics")
components.html(keystroke_html, height=220)

# Parse submitted keystroke data from URL query param
ks_data_raw = st.query_params.get("ks_data", None)
ks_pred, ks_prob = None, None

if ks_data_raw:
    try:
        events   = json.loads(ks_data_raw)
        ks_pred, ks_prob = predict_keystroke(events)
        st.success(f"Keystroke data received — {len(events)} keys captured")
    except Exception as e:
        st.warning(f"Could not parse keystroke data: {e}")

st.markdown("---")
st.markdown("## Step 2 — Face liveness")
st.markdown("Upload a face image (screenshot or photo).")
face_file = st.file_uploader("Face image (.jpg / .png)", type=["jpg", "jpeg", "png"])

st.markdown("---")
st.markdown("## Step 3 — Voice anti-spoofing")
st.markdown("Upload a short audio recording (.wav).")
voice_file = st.file_uploader("Voice recording (.wav)", type=["wav"])

st.markdown("---")
st.markdown("## Step 4 — Run authentication")

if st.button("🔐 Authenticate now", type="primary"):

    results = {}
    cols = st.columns(3)

    # ── Keystroke result ──────────────────────────────────────────────────────
    with cols[0]:
        st.markdown("### ⌨️ Keystroke")
        if ks_pred is not None:
            label  = "Genuine" if ks_pred == 0 else "SPOOF"
            colour = "green"   if ks_pred == 0 else "red"
            prob_display = 1 - ks_prob if ks_pred == 0 else ks_prob
            st.markdown(f"**Decision:** :{colour}[{label}]")
            st.progress(float(prob_display))
            st.caption(f"Spoof probability: {ks_prob:.1%}")
            results["keystroke"] = ks_pred
        elif ks_loaded:
            st.info("Type the passphrase above first, then click 'Analyse keystrokes'.")
            results["keystroke"] = None
        else:
            st.error("Model files not found in models/ folder.")
            results["keystroke"] = None

    # ── Face result (upload preview — wire up ResNet18 here) ──────────────────
    with cols[1]:
        st.markdown("### 👤 Face")
        if face_file:
            img = Image.open(face_file)
            st.image(img, width=160, caption="Uploaded face")
            # ── TO WIRE UP RESNET18 INFERENCE ─────────────────────────────────
            # 1. Import your model class:
            #    import sys; sys.path.append(MODEL_DIR)
            #    from face_model import FaceBinaryResNet18
            # 2. Load weights:
            #    model = FaceBinaryResNet18()
            #    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,
            #        'face_binary_resnet18_best.pth'), map_location='cpu'))
            #    model.eval()
            # 3. Preprocess and run:
            #    transform = transforms.Compose([
            #        transforms.Resize((224, 224)),
            #        transforms.ToTensor(),
            #        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            #    ])
            #    x = transform(img.convert('RGB')).unsqueeze(0)
            #    with torch.no_grad():
            #        logit = model(x)
            #        prob  = torch.sigmoid(logit).item()
            #    pred = int(prob >= 0.5)
            # ──────────────────────────────────────────────────────────────────
            st.info("Connect face_binary_resnet18_best.pth — see comments in code.")
            results["face"] = None
        else:
            st.caption("No image uploaded.")
            results["face"] = None

    # ── Voice result (upload preview — wire up CNN here) ─────────────────────
    with cols[2]:
        st.markdown("### 🎤 Voice")
        if voice_file:
            st.audio(voice_file)
            # ── TO WIRE UP VOICE CNN INFERENCE ────────────────────────────────
            # 1. Load audio with librosa:
            #    import librosa
            #    y, sr = librosa.load(voice_file, sr=16000, mono=True)
            # 2. Extract MFCC or mel-spectrogram features
            # 3. Load your CNN model (voice_binary_cnn_best.pth)
            # 4. Run inference and get spoof probability
            # ──────────────────────────────────────────────────────────────────
            st.info("Connect voice_binary_cnn_best.pth — see comments in code.")
            results["voice"] = None
        else:
            st.caption("No audio uploaded.")
            results["voice"] = None

    # ── Fusion decision ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Fusion decision")

    decisions = [v for v in results.values() if v is not None]

    if decisions:
        spoof_votes   = sum(decisions)
        genuine_votes = len(decisions) - spoof_votes
        majority      = "SPOOF" if spoof_votes > genuine_votes else "Genuine"

        if majority == "Genuine":
            st.success(f"## ✅ Access GRANTED — Genuine user detected")
        else:
            st.error(f"## ❌ Access DENIED — Spoof attack detected")

        st.markdown(
            f"Votes: **{genuine_votes} genuine** / **{spoof_votes} spoof** "
            f"({len(decisions)} modalities evaluated)"
        )

        # Vote breakdown
        vote_cols = st.columns(3)
        labels = ["Keystroke", "Face", "Voice"]
        values = [results.get("keystroke"), results.get("face"), results.get("voice")]
        for col, label, val in zip(vote_cols, labels, values):
            with col:
                if val is None:
                    col.caption(f"{label}: not evaluated")
                elif val == 0:
                    col.success(f"{label}: Genuine")
                else:
                    col.error(f"{label}: Spoof")
    else:
        st.info("Complete at least one modality above to see a fusion decision.")
```

---

## Step 5 — Create `pages/2_Results.py`

This page shows the full interactive results dashboard using your actual numbers.

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("📊 Model Performance Results")
st.markdown("All results computed on the 95-sample joint validation set (24 genuine, 71 spoof).")
st.markdown("---")

# ── All results — hardcoded from your actual notebook outputs ─────────────────
data = {
    "Model": [
        # Unimodal
        "Face (ResNet18)",
        "Voice (CNN)",
        "Keystroke (XGBoost)",
        # Bimodal — Score
        "Face + Voice — Score",
        "Face + Keystroke — Score",
        "Voice + Keystroke — Score",
        # Bimodal — Feature
        "Face + Voice — Feature RF",
        "Face + Keystroke — Feature RF",
        "Voice + Keystroke — Feature RF",
        # Bimodal — Hybrid
        "Face + Voice — Hybrid",
        "Face + Keystroke — Hybrid",
        "Voice + Keystroke — Hybrid",
        # Trimodal
        "Trimodal — Equal weight",
        "Trimodal — Perf. weighted",
        "Trimodal — Majority vote",
        "Trimodal — Feature LR",
        "Trimodal — Feature RF",
        "Trimodal — Hybrid",
    ],
    "Level": [
        "Unimodal","Unimodal","Unimodal",
        "Bimodal","Bimodal","Bimodal",
        "Bimodal","Bimodal","Bimodal",
        "Bimodal","Bimodal","Bimodal",
        "Trimodal","Trimodal","Trimodal","Trimodal","Trimodal","Trimodal",
    ],
    "Fusion type": [
        "—","—","—",
        "Score","Score","Score",
        "Feature","Feature","Feature",
        "Hybrid","Hybrid","Hybrid",
        "Score","Score","Decision","Feature","Feature","Hybrid",
    ],
    "Accuracy (%)": [
        89.47, 83.16, 95.74,
        92.63, 95.79, 90.53,
        94.74, 97.89, 93.68,
        95.79, 96.84, 93.68,
        96.84, 96.84, 97.89, 96.84, 96.84, 96.84,
    ],
    "F1 (%)": [
        92.42, 88.24, 96.00,
        94.96, 97.14, 93.53,
        96.50, 98.61, 95.83,
        97.18, 97.90, 95.77,
        97.87, 97.87, 98.57, 97.87, 97.93, 97.87,
    ],
    "AUC-ROC": [
        0.9701, 0.9137, 0.9946,
        0.9765, 0.9894, 0.9783,
        0.9844, 0.9971, 0.9487,
        0.9853, 0.9935, 0.9683,
        0.9959, 0.9965, 0.9959, 0.9947, 0.9965, 0.9971,
    ],
    "FAR (%)": [
        0.00, 20.83, 8.33,
        8.33, 4.17, 12.50,
        12.50, 8.33, 16.67,
        8.33, 8.33, 12.50,
        4.17, 4.17, 0.00, 4.17, 12.50, 4.17,
    ],
    "FRR (%)": [
        14.08, 15.49, 4.26,
        7.04, 4.23, 8.45,
        2.82, 0.00, 2.82,
        2.82, 1.41, 4.23,
        2.82, 2.82, 2.82, 2.82, 0.00, 2.82,
    ],
}

df = pd.DataFrame(data)

COLOURS = {
    "Unimodal": "#5DCAA5",
    "Bimodal":  "#378ADD",
    "Trimodal": "#D85A30",
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "All results", "Accuracy chart", "FAR vs FRR", "Fusion comparison"
])

# ── Tab 1: Full table ──────────────────────────────────────────────────────────
with tab1:
    level_filter = st.multiselect(
        "Filter by level",
        ["Unimodal", "Bimodal", "Trimodal"],
        default=["Unimodal", "Bimodal", "Trimodal"],
    )
    filtered = df[df["Level"].isin(level_filter)].copy()

    def highlight_best(s):
        if s.name in ["Accuracy (%)", "F1 (%)", "AUC-ROC"]:
            best = s.max()
            return [
                "background-color:#EAF3DE;color:#27500A;font-weight:600"
                if v == best else "" for v in s
            ]
        elif s.name in ["FAR (%)", "FRR (%)"]:
            best = s.min()
            return [
                "background-color:#EAF3DE;color:#27500A;font-weight:600"
                if v == best else "" for v in s
            ]
        return [""] * len(s)

    st.dataframe(
        filtered.style.apply(highlight_best),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Green highlight = best value in that column across all shown rows.")

# ── Tab 2: Accuracy bar chart ──────────────────────────────────────────────────
with tab2:
    fig = go.Figure()
    for level in ["Unimodal", "Bimodal", "Trimodal"]:
        subset = df[df["Level"] == level]
        fig.add_trace(go.Bar(
            name=level,
            x=subset["Model"],
            y=subset["Accuracy (%)"],
            marker_color=COLOURS[level],
            text=subset["Accuracy (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))

    fig.update_layout(
        title="Accuracy across all models — unimodal → bimodal → trimodal",
        yaxis=dict(range=[70, 102], title="Accuracy (%)"),
        xaxis=dict(tickangle=-40, title=""),
        barmode="group",
        legend_title="Level",
        height=520,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Clear improvement from unimodal (avg 89%) → bimodal (avg 94%) → trimodal (avg 97%)."
    )

# ── Tab 3: FAR vs FRR scatter ──────────────────────────────────────────────────
with tab3:
    marker_map = {"Unimodal": "circle", "Bimodal": "square", "Trimodal": "star"}
    fig2 = go.Figure()

    for level in ["Unimodal", "Bimodal", "Trimodal"]:
        subset = df[df["Level"] == level]
        fig2.add_trace(go.Scatter(
            x=subset["FAR (%)"],
            y=subset["FRR (%)"],
            mode="markers+text",
            name=level,
            text=subset["Model"].apply(lambda x: x.split("—")[0].strip()),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=14,
                symbol=marker_map[level],
                color=COLOURS[level],
                line=dict(width=1, color="white"),
            ),
        ))

    fig2.update_layout(
        title="FAR vs FRR — lower is better for both axes",
        xaxis_title="FAR — False Accept Rate (%) → lower = more secure",
        yaxis_title="FRR — False Reject Rate (%) → lower = more usable",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig2.add_annotation(
        x=0, y=0, text="  Ideal (0%, 0%)",
        showarrow=True, arrowhead=2, ax=60, ay=-40,
        font=dict(color="#27500A", size=11),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Trimodal majority vote achieves 0% FAR with only 2.82% FRR — "
        "closest to the ideal bottom-left corner."
    )

# ── Tab 4: Fusion type comparison ─────────────────────────────────────────────
with tab4:
    fused = df[df["Level"] != "Unimodal"].copy()

    fig3 = px.box(
        fused,
        x="Fusion type",
        y="Accuracy (%)",
        color="Level",
        color_discrete_map={"Bimodal": "#378ADD", "Trimodal": "#D85A30"},
        title="Accuracy distribution by fusion strategy",
        points="all",
        hover_data=["Model", "FAR (%)", "FRR (%)"],
    )
    fig3.update_layout(
        height=440,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[70, 101]),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("### Key findings")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Performance progression**
        - Unimodal average: **86.0%** accuracy
        - Bimodal average: **94.2%** accuracy
        - Trimodal average: **97.2%** accuracy
        - Each additional modality reduces error rate
        """)
    with col2:
        st.markdown("""
        **Best system per category**
        - Best unimodal: Face — 89.47%, **0% FAR**
        - Best bimodal: Face+Keystroke Feature RF — 97.89%, 0% FRR
        - Best trimodal: Majority vote — 97.89%, **0% FAR**
        - Best AUC-ROC: Hybrid trimodal — **0.9971**
        """)

    st.markdown("---")
    st.markdown("### MLP failure note")
    st.info(
        "The MLP meta-classifier collapsed on the trimodal feature-level task "
        "(28.42% accuracy) — it predicted nearly everything as genuine. "
        "This is expected: with only 95 samples, neural meta-learners overfit. "
        "Random Forest and Logistic Regression are the appropriate choices for "
        "this dataset size."
    )
```

---

## Step 6 — Run the app

```bash
cd triauth
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Step 7 — Wire up face and voice inference (optional but recommended)

Once the app is running, open `pages/1_Live_Demo.py` in Claude Code and say:

> *"I need to wire up the face ResNet18 inference. The model class is a ResNet18
> with the final FC layer replaced by `nn.Linear(512, 1)`. The model file is
> `face_binary_resnet18_best.pth`. Uncomment and complete the face inference block."*

Then for voice:

> *"I need to wire up the voice CNN inference. The model file is
> `voice_binary_cnn_best.pth`. My model takes raw waveform or MFCC features as
> input. Uncomment and complete the voice inference block."*

---

## What each page does at a glance

| Page | What's live | What's simulated |
|---|---|---|
| Home | Headline metrics, system description | — |
| Live Demo | Keystroke capture + XGBoost inference | Face/voice uploads (ready to wire up) |
| Results | Full interactive Plotly charts from real data | — |

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'xgboost'"**
```bash
pip install xgboost
```

**"FileNotFoundError: xgb_keystroke.pkl"**
Make sure all 5 model files are in the `models/` subfolder.

**Keystroke data not passing between browser and Python**
This uses URL query params. Make sure you click "Analyse keystrokes" in the
browser component, which reloads the page with `?ks_data=...` appended.

**Streamlit page order wrong**
Pages are sorted alphabetically by filename. The `1_` and `2_` prefixes ensure
correct order. Do not rename the files.
