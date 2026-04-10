import math
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from collections import Counter

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parola Güçlülük Analizi",
    page_icon="🔐",
    layout="centered",
)

# ── Model & Scaler loading ────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("password_model.h5")
    scaler = joblib.load("password.joblib")
    return model, scaler


# ── Feature extraction (must match notebook exactly) ─────────────────────────
def extract_features(password: str) -> dict:
    length = len(password)
    if length == 0:
        return {k: 0 for k in ["length", "unique_ratio", "digits", "upper", "lower", "special", "entropy", "char_types"]}

    digits = sum(c.isdigit() for c in password)
    upper = sum(c.isupper() for c in password)
    lower = sum(c.islower() for c in password)
    special = sum(not c.isalnum() for c in password)

    counts = Counter(password)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())

    return {
        "length": length,
        "unique_ratio": len(set(password)) / length,
        "digits": digits,
        "upper": upper,
        "lower": lower,
        "special": special,
        "entropy": entropy,
        "char_types": sum([digits > 0, upper > 0, lower > 0, special > 0]),
    }


# ── Prediction ────────────────────────────────────────────────────────────────
FEATURE_ORDER = ["length", "unique_ratio", "digits", "upper", "lower", "special", "entropy", "char_types"]

# Encoded label → (Strength_Level, label_tr, label_en, emoji, progress)
STRENGTH_MAP = {
    0: (1, "Zayıf",      "Weak",        "🔴", 0.25),
    1: (2, "Orta",       "Average",     "🟡", 0.50),
    2: (3, "Güçlü",      "Strong",      "🟢", 0.75),
    3: (4, "Çok Güçlü",  "Very Strong", "🛡️", 1.00),
}


def predict_strength(password: str, model, scaler):
    features = extract_features(password)
    feature_values = [[features[k] for k in FEATURE_ORDER]]
    feature_array = np.array(feature_values, dtype=float)
    scaled = scaler.transform(feature_array)
    probabilities = model.predict(scaled, verbose=0)[0]
    encoded_label = int(np.argmax(probabilities))
    confidence = float(probabilities[encoded_label])
    return encoded_label, confidence, features, probabilities


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Güçlü Şifre İpuçları")
    st.markdown(
        """
- En az **12 karakter** kullanın
- **Büyük** ve **küçük** harf karıştırın
- **Rakam** ekleyin (0-9)
- **Özel karakter** kullanın (`!@#$%^&*`)
- Sözlükte bulunan kelimelerden kaçının
- Her hesap için **benzersiz** şifre seçin
"""
    )
    st.divider()
    st.header("ℹ️ Uygulama Hakkında")
    st.markdown(
        """
Bu uygulama, PWLDS veri seti üzerinde eğitilmiş bir **derin öğrenme modeli** (Keras/TensorFlow) kullanarak
parolanın güçlülük seviyesini 4 kategoride tahmin eder:

| Seviye | Etiket |
|--------|--------|
| 1 | 🔴 Zayıf |
| 2 | 🟡 Orta |
| 3 | 🟢 Güçlü |
| 4 | 🛡️ Çok Güçlü |
"""
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔐 Parola Güçlülük Analizi")
st.markdown("Parolanızı girin ve güçlülük seviyesini öğrenin.")

model, scaler = load_model_and_scaler()

password = st.text_input(
    "Parolanızı girin:",
    type="password",
    placeholder="örn. MyP@ssw0rd!2024",
)

show_plain = st.checkbox("Parolayı göster")
if show_plain and password:
    st.code(password, language=None)

if st.button("🔍 Analiz Et", use_container_width=True):
    if not password:
        st.warning("⚠️ Lütfen bir parola girin.")
    else:
        with st.spinner("Analiz ediliyor…"):
            encoded_label, confidence, features, probabilities = predict_strength(password, model, scaler)

        strength_level, label_tr, label_en, emoji, progress_val = STRENGTH_MAP[encoded_label]

        st.divider()
        st.subheader("📊 Sonuç")

        # Coloured feedback box
        msg = f"{emoji} **{label_tr} ({label_en})** — Güç Seviyesi: {strength_level}/4"
        if strength_level == 1:
            st.error(msg + " — ⚠️ Tehlikeli")
        elif strength_level == 2:
            st.warning(msg + " — 🔧 Geliştirilmeli")
        elif strength_level == 3:
            st.success(msg + " — ✅ İyi")
        else:
            st.success(msg + " — 🏆 Mükemmel")

        # Confidence & progress bar
        st.metric("Güven Skoru", f"{confidence * 100:.1f}%")
        st.progress(progress_val)

        # Feature details
        with st.expander("🔬 Özellik Detayları"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Uzunluk", features["length"])
                st.metric("Büyük Harf", features["upper"])
                st.metric("Küçük Harf", features["lower"])
                st.metric("Rakam", features["digits"])
            with col2:
                st.metric("Özel Karakter", features["special"])
                st.metric("Entropi", f"{features['entropy']:.2f}")
                st.metric("Karakter Çeşitliliği", f"{features['char_types']} / 4")
                st.metric("Benzersizlik Oranı", f"{features['unique_ratio']:.2f}")

        # Probability distribution
        with st.expander("📈 Sınıf Olasılıkları"):
            for enc, (lvl, lbl_tr, lbl_en, em, _) in STRENGTH_MAP.items():
                prob_pct = float(probabilities[enc]) * 100
                st.write(f"{em} **{lbl_tr}** ({lbl_en}): {prob_pct:.1f}%")
                st.progress(float(probabilities[enc]))
