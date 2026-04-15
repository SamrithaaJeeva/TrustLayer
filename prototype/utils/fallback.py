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
