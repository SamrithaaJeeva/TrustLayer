Build me a complete Streamlit web application called TriAuth — a trimodal biometric authentication demo. Here are the full requirements:

---

## FOLDER STRUCTURE

Create this exact structure in the current directory:

```
triauth/
├── app.py
├── requirements.txt
├── utils/
│   └── fallback.py
├── pages/
│   ├── 1_Live_Demo.py
│   └── 2_Results.py
├── models/          (create empty folder — I will add files here)
└── data/            (create empty folder — I will add files here)
```

---

## FILE 1 — requirements.txt

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
librosa>=0.10.0
soundfile>=0.12.0
opencv-python-headless>=4.8.0
```

---

## FILE 2 — utils/fallback.py

This is a fallback inference engine. When real models fail to load, it samples from probability distributions extracted from my real validation dataset. The distributions are:

- Genuine face spoof prob: mean=0.14, std=0.14, clip to [0.00, 0.38]
- Genuine voice spoof prob: mean=0.27, std=0.28, clip to [0.01, 0.97]
- Genuine keystroke spoof prob: mean=0.12, std=0.26, clip to [0.00, 0.98]
- Spoof face spoof prob: mean=0.79, std=0.26, clip to [0.29, 1.00]
- Spoof voice spoof prob: mean=0.80, std=0.26, clip to [0.24, 1.00]
- Spoof keystroke spoof prob: mean=0.83, std=0.33, clip to [0.04, 0.99]

```python
import numpy as np

_GENUINE_DISTS = {
    "face":      (0.14, 0.14, 0.00, 0.38),
    "voice":     (0.27, 0.28, 0.01, 0.97),
    "keystroke": (0.12, 0.26, 0.00, 0.98),
}

_SPOOF_DISTS = {
    "face":      (0.79, 0.26, 0.29, 1.00),
    "voice":     (0.80, 0.26, 0.24, 1.00),
    "keystroke": (0.83, 0.33, 0.04, 0.99),
}

def _sample(mean, std, lo, hi, seed=None):
    rng = np.random.default_rng(seed)
    return float(np.clip(rng.normal(mean, std), lo, hi))

def fallback_predict(modality: str, is_spoof_hint: bool = False, seed=None):
    dist = _SPOOF_DISTS[modality] if is_spoof_hint else _GENUINE_DISTS[modality]
    prob = _sample(*dist, seed=seed)
    return int(prob >= 0.5), prob

def score_fusion(face_prob, voice_prob, ks_prob):
    fused = (face_prob + voice_prob + ks_prob) / 3.0
    return int(fused >= 0.5), float(fused)

def majority_vote(face_pred, voice_pred, ks_pred):
    return int((face_pred + voice_pred + ks_pred) >= 2)
```

---

## FILE 3 — app.py (Home page)

This is the Streamlit home page. Use st.set_page_config with layout="wide". Show three metric cards in a row using st.columns(3) styled with inline HTML:

- Card 1: "97.89%" headline, "Best accuracy", "Trimodal majority vote" — green background (#EAF3DE), text #27500A
- Card 2: "0.9971" headline, "Best AUC-ROC", "Hybrid trimodal fusion" — blue background (#E6F1FB), text #0C447C  
- Card 3: "0% FAR" headline, "False Accept Rate", "Trimodal majority vote" — purple background (#EEEDFE), text #3C3489

Below the cards, show a markdown table of the three modalities and their individual accuracies:
- Face liveness | ResNet18 binary classifier | 89.47%
- Voice anti-spoofing | CNN binary classifier | 83.16%
- Keystroke dynamics | XGBoost classifier | 95.74%

Also list the four attack types tested: TTS, Replay, Logical, Synthetic.

List the four fusion strategies: Score-level, Feature-level, Decision-level (majority vote), Hybrid.

End with a note pointing to the sidebar for navigation.

---

## FILE 4 — pages/1_Live_Demo.py

This is the most important page. It must:

### A. Load models with silent fallback

Use @st.cache_resource for all model loading. If any model fails to load for any reason, silently fall back — never show an error to the user.

**Keystroke model** — load three joblib files from models/ folder:
- xgb_keystroke.pkl
- keystroke_scaler.pkl  
- keystroke_imputer.pkl

**Face model** — load face_binary_resnet18_best.pth using torchvision ResNet18 with the final FC layer replaced by nn.Linear(512, 1). Handle both raw state_dict and wrapped checkpoints (check for "state_dict" and "model_state_dict" keys). Use strict=False when loading.

**Voice model** — load voice_binary_cnn_best.pth using torch.load. If it loads an object with an .eval() method, use it directly.

### B. Sidebar scenario selector

Add to st.sidebar:
- Title: "Demo controls"
- A radio button called "Select scenario" with options "Genuine user" and "Spoof attack"
- A caption explaining this controls the fallback distribution and has no effect when real models are loaded
- This sets a boolean variable `is_spoof` used by fallback_predict()

### C. Keystroke feature extraction

Use this EXACT function — it must match the training notebook precisely:

```python
FEATURE_COLS = [
    "hold_mean", "hold_std", "hold_min", "hold_max", "hold_median",
    "flight_mean", "flight_std", "flight_min", "flight_max", "flight_median",
    "num_keys", "unique_keys", "hold_flight_ratio",
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
```

predict_keystroke(events): if real model loaded and len(events)>=5, extract features, run through imputer then scaler then xgb_model.predict_proba, return (pred, prob). Otherwise call fallback_predict("keystroke", is_spoof_hint=is_spoof).

predict_face(image_pil): if real model loaded, apply torchvision transforms (Resize 224x224, ToTensor, Normalize with ImageNet mean/std), run through model, apply sigmoid. Otherwise fallback.

predict_voice(audio_bytes): if real model loaded, try librosa.load at sr=16000, extract MFCC (n_mfcc=40), run through model. Otherwise fallback.

### D. Live recording HTML component

Use streamlit.components.v1.components.html() to embed a self-contained HTML page that does all three captures. The component height should be 700. It must:

**Face capture:**
- Button "Start camera" calls navigator.mediaDevices.getUserMedia({video: true})
- Show live preview in a <video> element
- Button "Capture frame" draws the video frame to a hidden <canvas>, converts to base64 JPEG, stores in variable faceData

**Voice recording:**
- Button "Start recording" calls navigator.mediaDevices.getUserMedia({audio: true}), creates a MediaRecorder
- Button "Stop recording" stops the recorder, assembles chunks into a Blob, converts to base64 via FileReader, stores in voiceData. Shows an <audio> playback element so user can confirm recording.

**Keystroke capture:**
- Show the passphrase "authenticate user now" in a styled box
- An <input> element captures keydown/keyup events
- Calculate holdTime (keyup - keydown in ms) and flightTime (keydown - previous keyup in ms) for each key
- Store all events in array ksEvents

**Submit button:**
- Only appears when faceData is set AND voiceData is set AND ksEvents.length >= 5
- Calls submitAll() which builds URL query params:
  - ks_data: JSON.stringify of clean keystroke events (key, keyCode, holdTime, flightTime only)
  - face_data: faceData (base64 string)
  - voice_data: voiceData (base64 string)
- Sets window.location.href to the current page URL with these params

Style the component cleanly: card divs with light gray background, rounded corners, system-ui font. Buttons styled with appropriate colors (green for start, red for stop, blue for submit).

### E. Parse submitted data and run inference

After the component, check st.query_params for ks_data, face_data, voice_data.

If any of these exist:
1. Parse ks_data (URL-decoded JSON) → run predict_keystroke
2. Decode face_data (base64) → PIL Image → run predict_face, display the captured image
3. Decode voice_data (base64) → bytes → run predict_voice

If a parse fails for any reason, fall back silently.

### F. Display results

Show three columns (Keystroke, Face, Voice) each with:
- The captured image (face only)
- Decision label colored green (Genuine) or red (SPOOF)
- A st.progress bar showing confidence (1-prob if genuine, prob if spoof)
- A caption with the spoof probability as percentage

### G. Fusion decision with three tabs

Show tabs: "Score-level fusion", "Majority vote (best)", "Hybrid fusion"

**Score-level tab:** call score_fusion(face_prob, voice_prob, ks_prob) from utils.fallback. Show green st.success "Access GRANTED" or red st.error "Access DENIED". Show st.metric with fused probability. Caption: "Equal weighted average of all three spoof probabilities."

**Majority vote tab:** call majority_vote(face_pred, voice_pred, ks_pred). Show GRANTED/DENIED. Show vote count as "X genuine / Y spoof". St.progress showing genuine votes / 3. Caption: "Achieves 97.89% accuracy and 0% FAR on the validation set."

**Hybrid fusion tab:** compute stage1_prob = face_prob*0.5 + voice_prob*0.3 + ks_prob*0.2, then stage2_prob = stage1_prob*0.5 + face_prob*0.17 + voice_prob*0.17 + ks_prob*0.16, then pred = int(stage2_prob >= 0.5). Show GRANTED/DENIED. Caption: "3-stage cascade: Feature → Score → Decision. AUC-ROC 0.9971 on validation set."

### H. Clear / try again button

After the results, show a "Try again" button that calls st.query_params.clear() then st.rerun().

If no query params exist yet, show st.info telling the user to complete all three steps above.

---

## FILE 5 — pages/2_Results.py

Full interactive results dashboard using Plotly. Four tabs: "All results", "Accuracy chart", "FAR vs FRR", "Fusion comparison".

Use this exact data (these are my real validation numbers):

```python
data = {
    "Model": [
        "Face (ResNet18)", "Voice (CNN)", "Keystroke (XGBoost)",
        "Face + Voice — Score", "Face + Keystroke — Score", "Voice + Keystroke — Score",
        "Face + Voice — Feature RF", "Face + Keystroke — Feature RF", "Voice + Keystroke — Feature RF",
        "Face + Voice — Hybrid", "Face + Keystroke — Hybrid", "Voice + Keystroke — Hybrid",
        "Trimodal — Equal weight", "Trimodal — Perf. weighted", "Trimodal — Majority vote",
        "Trimodal — Feature LR", "Trimodal — Feature RF", "Trimodal — Hybrid",
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
        89.47,83.16,95.74,
        92.63,95.79,90.53,
        94.74,97.89,93.68,
        95.79,96.84,93.68,
        96.84,96.84,97.89,96.84,96.84,96.84,
    ],
    "F1 (%)": [
        92.42,88.24,96.00,
        94.96,97.14,93.53,
        96.50,98.61,95.83,
        97.18,97.90,95.77,
        97.87,97.87,98.57,97.87,97.93,97.87,
    ],
    "AUC-ROC": [
        0.9701,0.9137,0.9946,
        0.9765,0.9894,0.9783,
        0.9844,0.9971,0.9487,
        0.9853,0.9935,0.9683,
        0.9959,0.9965,0.9959,0.9947,0.9965,0.9971,
    ],
    "FAR (%)": [
        0.00,20.83,8.33,
        8.33,4.17,12.50,
        12.50,8.33,16.67,
        8.33,8.33,12.50,
        4.17,4.17,0.00,4.17,12.50,4.17,
    ],
    "FRR (%)": [
        14.08,15.49,4.26,
        7.04,4.23,8.45,
        2.82,0.00,2.82,
        2.82,1.41,4.23,
        2.82,2.82,2.82,2.82,0.00,2.82,
    ],
}
```

Use these colors: Unimodal="#5DCAA5", Bimodal="#378ADD", Trimodal="#D85A30". Set plot_bgcolor and paper_bgcolor to "rgba(0,0,0,0)" on all charts.

**Tab 1 — All results:** multiselect to filter by level. Style the dataframe so the best value in each metric column is highlighted green (#EAF3DE background, #27500A text, font-weight 600). For accuracy/F1/AUC highlight the max; for FAR/FRR highlight the min.

**Tab 2 — Accuracy chart:** grouped bar chart, one bar group per level, x-axis is Model name (tickangle -40), y-axis range 70–103, text labels above bars showing the percentage. Title: "Accuracy — unimodal → bimodal → trimodal". Height 520.

**Tab 3 — FAR vs FRR scatter:** scatter plot with x=FAR(%), y=FRR(%), each level gets a different marker symbol (Unimodal=circle, Bimodal=square, Trimodal=star), text labels showing model name (split on "—" and take first part). Add annotation at (0,0) saying "Ideal (0%,0%)". Caption: "Trimodal majority vote sits at 0% FAR, 2.82% FRR — closest to the ideal corner."

**Tab 4 — Fusion comparison:** box plot using plotly express, x=Fusion type, y=Accuracy(%), color=Level, points="all", hover shows Model/FAR/FRR. Below the chart, two columns with key findings text. Then st.info explaining that MLP collapsed on trimodal (28.42% accuracy) because 95 samples is too small for a neural meta-learner — this is expected and was discussed in the report.

---

## AFTER CREATING ALL FILES

1. Install all dependencies:
```bash
pip install streamlit plotly xgboost scikit-learn torch torchvision joblib pillow numpy pandas librosa soundfile opencv-python-headless
```

2. Run the app:
```bash
cd triauth
streamlit run app.py
```

3. Confirm the app opens at http://localhost:8501 with three sidebar pages: Home, Live Demo, Results.

4. Tell me what model files to place in the models/ folder and what CSV files to place in data/.

---

## IMPORTANT NOTES

- The pages/ filenames must be exactly "1_Live_Demo.py" and "2_Results.py" — the number prefix controls sidebar order.
- All model loading must be wrapped in try/except with silent fallback — never crash or show an error to the user.
- The HTML recording component must be fully self-contained — no external JS libraries, no fetch calls.
- The face_data and voice_data base64 strings are passed via URL query params — if they are too large (>2MB), show a warning asking the user to keep recordings under 5 seconds.
- Use sys.path to make the utils/ module importable from within the pages/ subdirectory.
- All Plotly charts must have transparent backgrounds so they work in both light and dark Streamlit themes.
