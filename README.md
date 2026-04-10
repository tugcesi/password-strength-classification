# 🔐 Parola Güçlülük Sınıflandırması / Password Strength Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Proje Hakkında / About

**TR:** Bu proje, kullanıcıların girdiği parolaları dört güç seviyesinde sınıflandıran bir makine öğrenmesi uygulamasıdır. Eğitimli derin öğrenme modeli bir Streamlit arayüzü üzerinden kullanılabilmektedir.

**EN:** This project is a machine learning application that classifies passwords into four strength levels. The trained deep-learning model is accessible via a Streamlit web interface.

---

## 📂 Dataset

| Özellik | Detay |
|---------|-------|
| Kaynak | [Infinitode/PWLDS](https://huggingface.co/datasets/Infinitode/PWLDS) |
| Dosyalar | `pwlds_weak.csv`, `pwlds_average.csv`, `pwlds_strong.csv`, `pwlds_very_strong.csv` |
| Toplam Satır | ~12 milyon |
| Hedef Değişken | `Strength_Level` (1 · 2 · 3 · 4) |

---

## 🧠 Model Mimarisi

```
Input (8 features)
  └─ Dense(128, relu)
  └─ Dense(64,  relu)
  └─ Dense(30,  relu)
  └─ Dense(8,   relu)
  └─ Dense(4,   softmax)   ← 4 sınıf
```

- **Kayıp Fonksiyonu:** `sparse_categorical_crossentropy`  
- **Optimizasyon:** Adam  
- **Erken Durdurma:** `EarlyStopping(monitor="val_loss", patience=3)`  
- **Ölçekleme:** `StandardScaler` (`password.joblib`)

### Özellik Mühendisliği

Her paroladan 8 sayısal özellik çıkarılır:

| Özellik | Açıklama |
|---------|----------|
| `length` | Parola uzunluğu |
| `unique_ratio` | Benzersiz karakter oranı |
| `digits` | Rakam sayısı |
| `upper` | Büyük harf sayısı |
| `lower` | Küçük harf sayısı |
| `special` | Özel karakter sayısı |
| `entropy` | Shannon entropisi |
| `char_types` | Kullanılan karakter türü sayısı (0–4) |

---

## 📊 Sınıflandırıcı Karşılaştırması

Derin öğrenme modeline ek olarak çeşitli klasik sınıflandırıcılar da denenmiş ve sonuçları karşılaştırılmıştır:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **DecisionTreeClassifier** | **92.3%** | **92.3%** | **92.3%** | **92.3%** |
| RandomForestClassifier | ~91% | ~91% | ~91% | ~91% |
| GradientBoostingClassifier | ~90% | ~90% | ~90% | ~90% |
| LogisticRegression | ~88% | ~88% | ~88% | ~88% |
| KNeighborsClassifier | ~87% | ~87% | ~87% | ~87% |
| AdaBoostClassifier | ~85% | ~85% | ~85% | ~85% |
| BernoulliNB | ~75% | ~75% | ~75% | ~75% |
| MultinomialNB | ~73% | ~73% | ~73% | ~73% |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu klonlayın

```bash
git clone https://github.com/tugcesi/password-strength-classification.git
cd password-strength-classification
```

### 2. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 3. Uygulamayı başlatın

```bash
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

---

## 🖼️ Ekran Görüntüsü

> *Uygulama ekran görüntüsü buraya eklenecek.*

---

## 📁 Dosya Yapısı

```
password-strength-classification/
├── app.py                              # Streamlit uygulaması
├── requirements.txt                    # Python bağımlılıkları
├── password_model.h5                   # Eğitilmiş Keras modeli
├── password.joblib                     # StandardScaler
├── data.csv                            # Örnek veri seti
├── PasswordStrenghtClassification.ipynb # Eğitim notebook'u
└── README.md
```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.
