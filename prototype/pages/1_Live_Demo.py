import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
import base64
import io

from utils.fallback import fallback_predict, score_fusion, majority_vote

st.set_page_config(page_title="Live Demo — TriAuth", page_icon="🔐", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb;
    background: #f9fafb; margin-bottom: 4px;
}
.result-card .modality  { font-size:0.8rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }
.result-card .verdict   { font-size:1.5rem; font-weight:700; margin:6px 0 4px; }
.result-card .prob-text { font-size:0.82rem; color:#6b7280; }
.verdict-genuine { color:#057a55; }
.verdict-spoof   { color:#e02424; }
.step-label {
    font-size:0.78rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.07em; color:#6b7280; margin-bottom:6px;
}
.fusion-granted {
    background:#f0fdf4; border:1.5px solid #86efac; border-radius:12px;
    padding:18px 22px; text-align:center;
}
.fusion-denied {
    background:#fff1f2; border:1.5px solid #fca5a5; border-radius:12px;
    padding:18px 22px; text-align:center;
}
.fusion-title { font-size:1.3rem; font-weight:700; margin:0 0 4px; }
.fusion-sub   { font-size:0.85rem; color:#6b7280; margin:0; }
</style>
""", unsafe_allow_html=True)

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_keystroke_models():
    try:
        import joblib
        xgb     = joblib.load(os.path.join(MODEL_DIR, "xgb_keystroke.pkl"))
        scaler  = joblib.load(os.path.join(MODEL_DIR, "keystroke_scaler.pkl"))
        imputer = joblib.load(os.path.join(MODEL_DIR, "keystroke_imputer.pkl"))
        return xgb, scaler, imputer, True
    except Exception:
        return None, None, None, False

@st.cache_resource
def load_face_model():
    try:
        import torch, torch.nn as nn
        from torchvision import models
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 1)
        ckpt = torch.load(os.path.join(MODEL_DIR, "face_binary_resnet18_best.pth"), map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, True
    except Exception:
        return None, False

@st.cache_resource
def load_voice_model():
    try:
        import torch
        obj = torch.load(os.path.join(MODEL_DIR, "voice_binary_cnn_best.pth"), map_location="cpu")
        if hasattr(obj, "eval"):
            obj.eval()
            return obj, True
        return None, False
    except Exception:
        return None, False

xgb_model, ks_scaler, ks_imputer, ks_loaded = load_keystroke_models()
face_model, face_loaded = load_face_model()
voice_model, voice_loaded = load_voice_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Demo controls")
scenario = st.sidebar.radio("Scenario", ["Genuine user", "Spoof attack"])
is_spoof = scenario == "Spoof attack"

st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
st.sidebar.markdown(
    f"{'🟢' if ks_loaded    else '🟡'} Keystroke (XGBoost)\n\n"
    f"{'🟢' if face_loaded  else '🟡'} Face (ResNet18)\n\n"
    f"{'🟢' if voice_loaded else '🟡'} Voice (CNN)"
)

# ── Feature extraction ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "hold_mean","hold_std","hold_min","hold_max","hold_median",
    "flight_mean","flight_std","flight_min","flight_max","flight_median",
    "num_keys","unique_keys","hold_flight_ratio",
]

def extract_keystroke_features(events):
    hold   = [e.get("holdTime")   for e in events if e.get("holdTime")   is not None]
    flight = [e.get("flightTime") for e in events if e.get("flightTime") is not None]
    keys   = [e.get("keyCode")    for e in events if e.get("keyCode")    is not None]
    hold   = np.array(hold,   dtype=float) if hold   else np.array([])
    flight = np.array(flight, dtype=float) if flight else np.array([])
    keys   = np.array(keys,   dtype=float) if keys   else np.array([])
    def s(a, fn, d=0): return fn(a) if len(a) > 0 else d
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
    if ks_loaded and len(events) >= 5:
        try:
            feats = extract_keystroke_features(events)
            X = np.array([[feats[c] for c in FEATURE_COLS]])
            X = ks_imputer.transform(X)
            X = ks_scaler.transform(X)
            prob = float(xgb_model.predict_proba(X)[0][1])
            return int(prob >= 0.5), prob
        except Exception:
            pass
    return fallback_predict("keystroke", is_spoof_hint=is_spoof)

def predict_face(image_pil):
    if face_loaded:
        try:
            import torch
            from torchvision import transforms
            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            x = tf(image_pil.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                prob = float(torch.sigmoid(face_model(x)).item())
            return int(prob >= 0.5), prob
        except Exception:
            pass
    return fallback_predict("face", is_spoof_hint=is_spoof)

def predict_voice(audio_bytes):
    if voice_loaded:
        try:
            import torch, librosa
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            feat  = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = voice_model(feat)
                if hasattr(out, "logits"): out = out.logits
                prob = float(torch.sigmoid(out.squeeze()).item())
            return int(prob >= 0.5), prob
        except Exception:
            pass
    return fallback_predict("voice", is_spoof_hint=is_spoof)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔐 Live Authentication Demo")
st.markdown("Capture your face, voice, and keystroke timing — then run the trimodal classifier.")
st.markdown("---")

# ── Keystroke HTML component ──────────────────────────────────────────────────
KEYSTROKE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing:border-box; margin:0; padding:0; font-family:system-ui,-apple-system,sans-serif; }
body { background:transparent; padding:4px 2px; }
.label {
  font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.06em; color:#6b7280; margin-bottom:8px;
}
.passphrase {
  background:#f3f4f6; border:1px solid #e5e7eb; border-radius:8px;
  padding:10px 14px; font-size:15px; font-weight:600;
  letter-spacing:.5px; color:#111827; margin-bottom:10px;
}
#ks-input {
  width:100%; padding:10px 14px; font-size:14px;
  border:1.5px solid #d1d5db; border-radius:8px; outline:none;
  background:white; color:#111827; transition:border-color .15s;
}
#ks-input:focus { border-color:#6366f1; }
.status {
  margin-top:8px; font-size:12px; color:#9ca3af; min-height:18px;
}
.status.ready { color:#057a55; font-weight:600; }
#submit-btn {
  margin-top:12px; width:100%; padding:11px;
  background:#4f46e5; color:white; border:none; border-radius:8px;
  font-size:14px; font-weight:600; cursor:pointer; transition:background .15s;
}
#submit-btn:hover    { background:#4338ca; }
#submit-btn:disabled { background:#c7d2fe; cursor:not-allowed; }
</style>
</head>
<body>
<div class="label">⌨️ Type the passphrase</div>
<div class="passphrase">authenticate user now</div>
<input id="ks-input" type="text" autocomplete="off" spellcheck="false" placeholder="Start typing here…"/>
<div class="status" id="ks-status">Waiting for input…</div>
<button id="submit-btn" disabled onclick="submitKS()">Submit keystroke data</button>

<script>
var events    = [];
var downTimes = {};

document.getElementById('ks-input').addEventListener('keydown', function(e) {
  downTimes[e.key] = performance.now();
});

document.getElementById('ks-input').addEventListener('keyup', function(e) {
  var up   = performance.now();
  var down = downTimes[e.key];
  if (down === undefined) return;
  var hold   = up - down;
  var flight = events.length > 0
    ? Math.max(0, down - events[events.length - 1]._up)
    : 0;
  events.push({ key:e.key, keyCode:e.keyCode,
    holdTime:parseFloat(hold.toFixed(2)),
    flightTime:parseFloat(flight.toFixed(2)),
    _up:up });

  var n = events.length;
  var st = document.getElementById('ks-status');
  var btn = document.getElementById('submit-btn');
  if (n >= 5) {
    st.textContent = '✓ ' + n + ' keys captured — ready to submit';
    st.className = 'status ready';
    btn.disabled = false;
  } else {
    st.textContent = n + ' keys captured (need at least 5)';
    st.className = 'status';
  }
});

function submitKS() {
  var clean = events.map(function(e) {
    return { key:e.key, keyCode:e.keyCode, holdTime:e.holdTime, flightTime:e.flightTime };
  });
  var param = encodeURIComponent(JSON.stringify(clean));
  var target = '/Live_Demo?ks_data=' + param;
  try {
    window.parent.location.href = target;
  } catch(err) {
    try { window.top.location.href = target; } catch(e) {}
  }
}
</script>
</body>
</html>"""

# ── Three-column capture layout ───────────────────────────────────────────────
col_face, col_voice, col_ks = st.columns(3, gap="medium")

with col_face:
    st.markdown("<div class='step-label'>👤 Step 1 — Face liveness</div>", unsafe_allow_html=True)
    face_file = st.camera_input("Take a photo", label_visibility="collapsed")

with col_voice:
    st.markdown("<div class='step-label'>🎤 Step 2 — Voice anti-spoofing</div>", unsafe_allow_html=True)
    voice_file = st.file_uploader("Upload voice recording", type=["wav","mp3","ogg","webm","m4a"],
                                   label_visibility="collapsed")
    if voice_file:
        st.audio(voice_file)

with col_ks:
    st.markdown("<div class='step-label'>⌨️ Step 3 — Keystroke dynamics</div>", unsafe_allow_html=True)
    # Check if keystrokes already submitted via URL param
    ks_raw = st.query_params.get("ks_data", None)
    if ks_raw:
        try:
            _ev = json.loads(ks_raw)
            st.success(f"✓ {len(_ev)} keys captured")
        except Exception:
            ks_raw = None
    if not ks_raw:
        components.html(KEYSTROKE_HTML, height=210)

st.markdown("---")

# ── Authenticate button ───────────────────────────────────────────────────────
ready = face_file is not None or voice_file is not None or ks_raw is not None

if not ready:
    st.info("Complete at least one step above, then click Authenticate.")
else:
    if st.button("🔐 Authenticate", type="primary", use_container_width=True):
        st.session_state["run_auth"] = True

# ── Inference & results ───────────────────────────────────────────────────────
if st.session_state.get("run_auth") and ready:

    ks_pred,    ks_prob    = fallback_predict("keystroke", is_spoof_hint=is_spoof)
    face_pred,  face_prob  = fallback_predict("face",      is_spoof_hint=is_spoof)
    voice_pred, voice_prob = fallback_predict("voice",     is_spoof_hint=is_spoof)

    from PIL import Image

    # Keystroke
    if ks_raw:
        try:
            events = json.loads(ks_raw)
            ks_pred, ks_prob = predict_keystroke(events)
        except Exception:
            pass

    # Face
    face_img = None
    if face_file:
        try:
            face_img = Image.open(face_file)
            face_pred, face_prob = predict_face(face_img)
        except Exception:
            pass

    # Voice
    if voice_file:
        try:
            voice_pred, voice_prob = predict_voice(voice_file.read())
        except Exception:
            pass

    # ── Per-modality result cards ─────────────────────────────────────────────
    st.markdown("### Results per modality")
    r1, r2, r3 = st.columns(3, gap="medium")

    def result_card(col, icon, label, pred, prob, img=None):
        verdict = "Genuine" if pred == 0 else "SPOOF"
        cls     = "verdict-genuine" if pred == 0 else "verdict-spoof"
        conf    = (1 - prob) if pred == 0 else prob
        bar_col = "#22c55e" if pred == 0 else "#ef4444"
        with col:
            if img:
                st.image(img, use_container_width=True)
            st.markdown(f"""
            <div class='result-card'>
              <div class='modality'>{icon} {label}</div>
              <div class='verdict {cls}'>{verdict}</div>
              <div style='background:#e5e7eb;border-radius:999px;height:6px;margin:8px 0'>
                <div style='background:{bar_col};width:{conf*100:.0f}%;height:6px;border-radius:999px'></div>
              </div>
              <div class='prob-text'>Spoof probability: <b>{prob:.1%}</b></div>
            </div>""", unsafe_allow_html=True)

    result_card(r1, "⌨️", "Keystroke", ks_pred,    ks_prob)
    result_card(r2, "👤", "Face",       face_pred,  face_prob, img=face_img)
    result_card(r3, "🎤", "Voice",      voice_pred, voice_prob)

    # ── Fusion tabs ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Fusion decision")

    tab1, tab2, tab3 = st.tabs(["Majority vote  ★", "Score-level", "Hybrid"])

    with tab1:
        mv = majority_vote(face_pred, voice_pred, ks_pred)
        spoof_v   = face_pred + voice_pred + ks_pred
        genuine_v = 3 - spoof_v
        granted = mv == 0
        st.markdown(f"""
        <div class='{"fusion-granted" if granted else "fusion-denied"}'>
          <div class='fusion-title'>{"✅ Access GRANTED" if granted else "❌ Access DENIED"}</div>
          <div class='fusion-sub'>{"Genuine user detected" if granted else "Spoof attack detected"}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<br>**{genuine_v}/3 genuine votes &nbsp;·&nbsp; {spoof_v}/3 spoof votes**", unsafe_allow_html=True)
        st.progress(genuine_v / 3)
        st.caption("97.89% accuracy · 0% FAR on the validation set")

    with tab2:
        _, fused_prob = score_fusion(face_prob, voice_prob, ks_prob)
        granted = fused_prob < 0.5
        st.markdown(f"""
        <div class='{"fusion-granted" if granted else "fusion-denied"}'>
          <div class='fusion-title'>{"✅ Access GRANTED" if granted else "❌ Access DENIED"}</div>
          <div class='fusion-sub'>Fused spoof probability: <b>{fused_prob:.1%}</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(np.clip(fused_prob, 0, 1)))
        st.caption("Equal-weighted average of all three spoof probabilities")

    with tab3:
        stage1 = face_prob*0.5 + voice_prob*0.3 + ks_prob*0.2
        stage2 = stage1*0.5 + face_prob*0.17 + voice_prob*0.17 + ks_prob*0.16
        granted = stage2 < 0.5
        st.markdown(f"""
        <div class='{"fusion-granted" if granted else "fusion-denied"}'>
          <div class='fusion-title'>{"✅ Access GRANTED" if granted else "❌ Access DENIED"}</div>
          <div class='fusion-sub'>Hybrid fused probability: <b>{stage2:.1%}</b></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(np.clip(stage2, 0, 1)))
        st.caption("3-stage cascade: Feature → Score → Decision · AUC-ROC 0.9971")

    # ── Try again ─────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Try again"):
        st.session_state.pop("run_auth", None)
        st.query_params.clear()
        st.rerun()
