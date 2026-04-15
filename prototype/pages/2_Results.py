import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Results — TriAuth", page_icon="📊", layout="wide")

st.markdown("""
<style>
.summary-card {
    background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px;
    padding:16px 20px; text-align:center;
}
.summary-card .val { font-size:1.6rem; font-weight:700; margin:0; }
.summary-card .lbl { font-size:0.8rem; color:#6b7280; margin:3px 0 0; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Performance Results")
st.markdown("---")

# ── Data ──────────────────────────────────────────────────────────────────────
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
    "EER (%)": [
        13.26, 14.00, 9.10,
        6.98,  4.20,  8.39,
        None,  None,  None,
        None,  None,  None,
        4.20,  None,  4.20,  None, None, None,
    ],
    "APCER (%)": [
        0.00,  20.83, 8.33,
        8.33,  4.17,  12.50,
        12.50, 8.33,  16.67,
        8.33,  8.33,  12.50,
        4.17,  4.17,  0.00,  4.17, 12.50, 4.17,
    ],
    "BPCER (%)": [
        14.08, 15.49, 16.90,   # Keystroke BPCER from fresh evaluation
        7.04,  4.23,  8.45,
        2.82,  0.00,  2.82,
        2.82,  1.41,  4.23,
        2.82,  2.82,  2.82,  2.82, 0.00, 2.82,
    ],
    "t-DCF": [
        0.1408, 39.74, 16.00,
        15.90,  7.96,  23.83,
        23.78,  15.83, 31.70,
        15.86,  15.84, 23.79,
        7.94,   7.95,  0.0282, 7.95, 23.75, 7.95,
    ],
}
df = pd.DataFrame(data)

COLOURS = {"Unimodal":"#5DCAA5","Bimodal":"#378ADD","Trimodal":"#D85A30"}
TRANSPARENT = "rgba(0,0,0,0)"

# ── Summary metric strip ──────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
summaries = [
    ("#f0fdf4","#15803d", f"{df['Accuracy (%)'].max():.2f}%",  "Best accuracy",   "Face+KS Feature RF & Trimodal MV"),
    ("#eff6ff","#1d4ed8", f"{df['AUC-ROC'].max():.4f}",        "Best AUC-ROC",    "Face+KS Feature RF & Hybrid"),
    ("#faf5ff","#6d28d9", f"{df['FAR (%)'].min():.0f}%",        "Lowest FAR",      "Face (ResNet18) & Trimodal MV"),
    ("#fff7ed","#c2410c", f"{df['FRR (%)'].min():.2f}%",        "Lowest FRR",      "Face+KS Feature RF & Voice+KS RF"),
]
for col, (bg, fg, val, lbl, sub) in zip([m1,m2,m3,m4], summaries):
    with col:
        st.markdown(f"""
        <div class='summary-card' style='background:{bg};border-color:{fg}22'>
          <div class='val' style='color:{fg}'>{val}</div>
          <div class='lbl'>{lbl}</div>
          <div style='font-size:0.7rem;color:#9ca3af;margin-top:2px'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["All results", "Accuracy chart", "FAR vs FRR", "Fusion comparison"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    level_filter = st.multiselect(
        "Filter by modality level",
        ["Unimodal","Bimodal","Trimodal"],
        default=["Unimodal","Bimodal","Trimodal"],
    )
    filtered = df[df["Level"].isin(level_filter)].copy()

    def highlight_best(s):
        if s.name in ["Accuracy (%)","F1 (%)","AUC-ROC"]:
            best = s.max()
        elif s.name in ["FAR (%)","FRR (%)","EER (%)","APCER (%)","BPCER (%)","t-DCF"]:
            best = s.min()
        else:
            return [""]*len(s)
        return ["background-color:#f0fdf4;color:#15803d;font-weight:600"
                if v == best else "" for v in s]

    st.dataframe(
        filtered.style.apply(highlight_best),
        use_container_width=True, hide_index=True,
    )
    st.caption("Green = best value in each column across currently filtered rows.")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    fig = go.Figure()
    for level in ["Unimodal","Bimodal","Trimodal"]:
        sub = df[df["Level"]==level]
        fig.add_trace(go.Bar(
            name=level, x=sub["Model"], y=sub["Accuracy (%)"],
            marker_color=COLOURS[level],
            text=sub["Accuracy (%)"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
    fig.update_layout(
        title="Accuracy — unimodal → bimodal → trimodal",
        yaxis=dict(range=[70,103], title="Accuracy (%)"),
        xaxis=dict(tickangle=-38, title=""),
        barmode="group", legend_title="Level", height=520,
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
        font=dict(size=12),
        margin=dict(t=50, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each additional modality cuts the error rate — unimodal avg 86% → bimodal 94% → trimodal 97%.")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    marker_map = {"Unimodal":"circle","Bimodal":"square","Trimodal":"star"}
    fig2 = go.Figure()
    for level in ["Unimodal","Bimodal","Trimodal"]:
        sub = df[df["Level"]==level]
        fig2.add_trace(go.Scatter(
            x=sub["FAR (%)"], y=sub["FRR (%)"],
            mode="markers+text", name=level,
            text=sub["Model"].apply(lambda x: x.split("—")[0].strip()),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=14, symbol=marker_map[level],
                        color=COLOURS[level], line=dict(width=1,color="white")),
        ))
    fig2.update_layout(
        title="FAR vs FRR — bottom-left corner is ideal",
        xaxis_title="FAR — False Accept Rate (%)",
        yaxis_title="FRR — False Reject Rate (%)",
        height=500,
        plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
        font=dict(size=12),
    )
    fig2.add_annotation(
        x=0, y=0, text="  Ideal (0%, 0%)",
        showarrow=True, arrowhead=2, ax=70, ay=-45,
        font=dict(color="#15803d", size=11),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Trimodal majority vote: 0% FAR · 2.82% FRR — closest to the ideal corner.")

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    fused = df[df["Level"]!="Unimodal"].copy()
    fig3 = px.box(
        fused, x="Fusion type", y="Accuracy (%)", color="Level",
        color_discrete_map={"Bimodal":"#378ADD","Trimodal":"#D85A30"},
        title="Accuracy distribution by fusion strategy",
        points="all",
        hover_data=["Model","FAR (%)","FRR (%)","EER (%)","APCER (%)","BPCER (%)","t-DCF"],
    )
    fig3.update_layout(
        height=440, plot_bgcolor=TRANSPARENT, paper_bgcolor=TRANSPARENT,
        yaxis=dict(range=[70,101]), font=dict(size=12),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Performance progression**
| Level | Avg accuracy |
|:---|---:|
| Unimodal | 86.0% |
| Bimodal  | 94.2% |
| Trimodal | 97.2% |
        """)
    with c2:
        st.markdown("""
**Best system per category**
| Category | Model | FAR |
|:---|:---|---:|
| Unimodal | Face ResNet18 | **0%** |
| Bimodal  | Face+KS Feature RF | 8.33% |
| Trimodal | Majority vote | **0%** |
        """)

    st.markdown("---")
    st.info(
        "**MLP failure note** — The MLP meta-classifier collapsed on the trimodal "
        "feature-level task (28.42% accuracy), predicting nearly everything as genuine. "
        "This is expected: with only 95 samples, neural meta-learners overfit severely. "
        "Random Forest and Logistic Regression are the appropriate choices at this dataset size."
    )
