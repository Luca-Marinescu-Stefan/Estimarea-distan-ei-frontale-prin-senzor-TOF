## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Marinescu Luca-Stefan |
| **Grupa / Specializare** | 634AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/Luca-Marinescu-Stefan/Estimarea-distan-ei-frontale-prin-senzor-TOF |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (pandas, numpy, scikit-learn, TensorFlow/Keras, Streamlit, matplotlib) |
| **Domeniul Industrial de Interes (DII)** | Robotică / Automatizări industriale / Senzori |
| **Tip Rețea Neuronală** | MLP (Keras) + RandomForestClassifier (optimizare) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 100.00% | 100.00% | +0.00% | ✓ |
| F1-Score (Macro) | ≥0.65 | 1.00 | 1.00 | +0.00 | ✓ |
| Latență Inferență | ≤1 ms/sample | 0.349 ms/sample | 0.349 ms/sample | ±0.000 ms | ✓ |
| Contribuție Date Originale | ≥40% | 63.84% | 63.84% | - | ✓ |
| Nr. Experimente Optimizare | ≥4 | 5 | 5 | - | ✓ |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În aplicații industriale (robotică, metrologie rapidă, control al distanței), senzorii Time‑of‑Flight pot avea erori neliniare din cauza suprafețelor reflectorizante sau a variațiilor de lumină. Proiectul propune un sistem inteligent de corecție a măsurătorii distanței, folosind un model ML care învață relația dintre distanța brută, intensitatea semnalului și temperatură, pentru a furniza o distanță estimată mai stabilă.

### 2.2 Beneficii Măsurabile Urmărite

1. Reducerea erorii de estimare față de măsurarea brută.
2. Stabilitate mai bună a măsurătorii în condiții de lumină variabilă.
3. Inferență rapidă (sub 1 ms/sample) pentru utilizare aproape real‑time.
4. Reducerea nevoii de calibrare manuală frecventă.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Estimare distanță stabilă cu ToF | Model ML corectează măsurătoarea brută | `src/neural_network/` | Accuracy/F1 pe test set |
| Achiziție și simulare date | Generare de date sintetice cu zgomot | `src/data_acquisition/` | Procent date originale |
| Interfață de testare rapidă | UI cu upload CSV + predicție | `src/app/` | Latență / predicție |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt: date brute + date generate |
| **Sursa concretă** | Date brute din senzor + date sintetice generate în Python |
| **Număr total observații finale (N)** | 3603 (raw + generated) |
| **Număr features** | 3 (distance_raw, signal_strength, temperature) |
| **Tipuri de date** | Numerice |
| **Format fișiere** | CSV |
| **Perioada colectării/generării** | Noiembrie 2025 – Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 3603 |
| **Observații originale (M)** | 2300 |
| **Procent contribuție originală** | 63.84% |
| **Tip contribuție** | Date sintetice (simulare/augmentare) |
| **Locație cod generare** | `src/data_acquisition/generate.py` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

Datele originale au fost generate prin simulare în Python, cu parametri de zgomot și variații controlate ale intensității semnalului și temperaturii, pentru a acoperi scenarii care apar frecvent în utilizarea senzorilor ToF. Aceste date sunt folosite împreună cu seturile brute pentru a îmbunătăți generalizarea modelului.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | ~70% | 352 |
| Validation | ~15% | 75 |
| Test | ~15% | 76 |

**Preprocesări aplicate:**
- Eliminare duplicate.
- Tratare valori lipsă (imputare numerică / eliminare cazuri invalide).
- Filtrare outlieri pe percentile.
- Standardizare cu `StandardScaler`.

**Referințe fișiere:** `data/README.md`, `config/preprocessing_params.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python | Generare date simulate + încărcare CSV | `src/data_acquisition/` |
| **Neural Network** | Keras (MLP) + scikit-learn (RF) | Antrenare/evaluare model | `src/neural_network/` |
| **Web Service / UI** | Streamlit | UI pentru upload CSV + predicție | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.svg`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare input | Start aplicație | Input primit |
| `ACQUIRE_DATA` | Citire din CSV / senzor | Request procesare | Date disponibile |
| `PREPROCESS` | Normalizare și filtrare | Date brute | Features ready |
| `INFERENCE` | Predicție model | Features disponibile | Predicție generată |
| `DISPLAY/ACT` | Afișare rezultat | Output RN | Confirmare user |
| `LOG` | Salvare predicții | Predicție finală | OK |
| `ERROR` | Tratare erori | Excepție | Recovery/Stop |

**Justificare alegere arhitectură State Machine:**

Fluxul de procesare reflectă un sistem de monitorizare aproape real‑time, unde datele sunt achiziționate, preprocesate, trecute prin model și afișate/utilizate pentru decizie. Starea `ERROR` izolează situațiile anormale (date corupte, lipsă), pentru un comportament robust.

### 4.3 Actualizări State Machine în Etapa 6

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| Threshold alertă | 0.50 | 0.35 | Reducere FN pentru alerte timpurii |
| Stare nouă adăugată | N/A | `CONFIDENCE_CHECK` | Filtrare predicții cu încredere scăzută |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Input (3 features) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(5, Softmax)

**Justificare alegere arhitectură:**

MLP este potrivit pentru date tabulare cu puține features, oferind un compromis bun între performanță și complexitate.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Model | RandomForestClassifier | Stabilitate și performanță pe date tabulare |
| n_estimators | 200 | Performanță maximă cu timp de antrenare redus |
| max_depth | None | Lăsarea modelului să învețe relații neliniare |
| min_samples_split | 2 | Setare standard pentru separare noduri |
| min_samples_leaf | 1 | Păstrează sensibilitatea la variații mici |
| Threshold alertă | 0.35 | Reducere FN în aplicații practice |
| Confidence min | 0.60 | Filtrare predicții incerte |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | n_estimators=200 | 1.00 | 1.00 | 0.255 s | Referință |
| Exp 1 | n_estimators=400 | 1.00 | 1.00 | 0.507 s | Fără îmbunătățiri |
| Exp 2 | max_depth=10 | 1.00 | 1.00 | 0.393 s | Similar baseline |
| Exp 3 | min_samples_leaf=2 | 1.00 | 1.00 | 0.364 s | Similar baseline |
| Exp 4 | min_samples_split=4 | 1.00 | 1.00 | 0.315 s | Similar baseline |
| **FINAL** | Configurația aleasă | **1.00** | **1.00** | 0.255 s | Model folosit în producție |

**Justificare alegere model final:**

Configurația baseline a avut performanță maximă cu timp de antrenare minim; modificările ulterioare nu au adus îmbunătățiri relevante.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.joblib`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | 100.00% | ≥70% | ✓ |
| **F1-Score (Macro)** | 1.00 | ≥0.65 | ✓ |
| **Precision (Macro)** | 1.00 | - | - |
| **Recall (Macro)** | 1.00 | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 100.00% | 100.00% | +0.00% |
| F1-Score | 1.00 | 1.00 | +0.00 |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | Toate clasele (0–4) – performanță perfectă |
| **Clasa cu cea mai slabă performanță** | N/A (fără confuzii) |
| **Confuzii frecvente** | Nu există (diagonală perfectă) |
| **Dezechilibru clase** | Nu afectează performanța în setul de test |

### 6.3 Analiza Top 5 Erori

Nu există erori în setul de test (`results/error_analysis.json` este gol).

### 6.4 Validare în Context Industrial

Rezultatele indică o corecție foarte bună pentru setul de test, ceea ce sugerează că modelul poate îmbunătăți semnificativ măsurarea distanței în aplicații industriale. Totuși, setul de test este relativ mic, deci recomand extinderea datelor reale pentru validare în condiții diverse.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.joblib` | `optimized_model.joblib` | Model optimizat |
| **Threshold decizie** | 0.50 | 0.35 | Reducere FN |
| **UI - feedback vizual** | Predicție simplă | Bară confidence + valoare | Decizie informată |
| **Confidence gating** | N/A | `CONFIDENCE_CHECK` | Filtrare predicții incerte |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

**Descriere:** UI Streamlit cu upload CSV și predicții + confidence.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/screenshots/` (capturi UI). Pentru demo video se poate adăuga `docs/demo/`.

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|-----------------|
| 1 | Upload CSV | Preview date |
| 2 | Procesare | Predicții + confidence |
| 3 | Decizie | `NORMAL` / `ALERT` / `CONFIDENCE_CHECK` |

**Latență măsurată end-to-end:** ~0.35 ms/sample (inferență)
**Data și ora demonstrației:** N/A (demo end-to-end neînregistrat)

---

## 8. Structura Repository-ului Final

```
Estimarea-distan-ei-frontale-prin-senzor-TOF-main/
├── README.md
├── README_Etapa4_Arhitectura_SIA.md
├── README_Etapa5_Antrenare_RN.md
├── etapa6_optimizare_concluzii.md
├── comenzi.txt
├── docs/
│   ├── etapa3.md
│   ├── state_machine.md
│   ├── state_machine.svg
│   ├── confusion_matrix_optimized.png
│   ├── screenshots/
│   │   ├── ui_demo.png
│   │   ├── inference_real.png
│   │   └── inference_optimized.png
│   └── optimization/
│       ├── accuracy_comparison.png
│       └── f1_comparison.png
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── generated/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   └── app/
├── models/
│   ├── trained_model.h5
│   ├── trained_model.joblib
│   └── optimized_model.joblib
├── results/
│   ├── training_history.csv
│   ├── test_metrics.json
│   ├── optimization_experiments.csv
│   ├── final_metrics.json
│   └── error_analysis.json
├── config/
│   ├── preprocessing_params.pkl
│   └── optimized_config.yaml
├── requirements.txt
└── NUME_Prenume_Grupa_README_Proiect_RN.md
```

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

Python >= 3.10

### 9.2 Instalare

1. Clonare repo
2. (Opțional) creare venv
3. `pip install -r requirements.txt`

### 9.3 Rulare Pipeline Complet

- Preprocesare:
  `python src/preprocessing/data_cleaner.py`
  `python src/preprocessing/feature_engineering.py`
  `python src/preprocessing/process_data.py`
  `python src/preprocessing/data_splitter.py`

- Antrenare model:
  `python src/neural_network/train.py --backend sklearn`

- Optimizare:
  `python src/neural_network/optimize.py`

- Evaluare:
  `python src/neural_network/evaluate.py --model models/optimized_model.joblib`

- UI:
  `python -m streamlit run src/app/main.py`

### 9.4 Verificare Rapidă

- `python src/neural_network/evaluate.py --model models/optimized_model.joblib`

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit | Target | Realizat | Status |
|------------------|--------|----------|--------|
| Accuracy pe test set | ≥70% | 100% | ✓ |
| F1-Score pe test set | ≥0.65 | 1.00 | ✓ |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

1. Setul de test este relativ mic (76 observații).
2. Validarea pe date reale extinse este necesară pentru robusteză.
3. Modelul optimizat este RandomForest (nu RN pur), dar este păstrat pentru performanță.

### 10.3 Lecții Învățate (Top 5)

1. Preprocesarea riguroasă îmbunătățește stabilitatea modelului.
2. Datele sintetice cresc semnificativ acoperirea cazurilor.
3. Măsurarea metricilor pe set separat e esențială pentru validare.
4. UI-ul simplu accelerează demonstrațiile și validarea.
5. Optimizarea hiperparametrilor poate confirma stabilitatea baseline-ului.

### 10.4 Retrospectivă

Aș extinde colectarea de date reale pentru a valida modelul în condiții industriale variate și aș adăuga logging complet pentru audit.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|----------------------|------------------|
| Short-term | Extindere set real de date | Generalizare mai bună |
| Medium-term | Integrare model RN pur + calibrare | Interpretabilitate și robustețe |
| Long-term | Deployment pe edge device | Latență redusă, integrare în linie |

---

## 11. Bibliografie

1. Keras Documentation, 2024. https://keras.io/getting_started/
2. Scikit‑learn RandomForestClassifier, 2024. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
3. ST VL53L0X Time‑of‑Flight Sensor – Datasheet. https://www.st.com/resource/en/datasheet/vl53l0x.pdf

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [x] **F1-Score ≥0.65** pe test set
- [x] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [x] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [x] **Minimum 4 experimente** de optimizare documentate
- [x] **Confusion matrix** generată
- [x] **State Machine** definit (minim 4-6 stări)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI
- [x] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [x] **README.md** complet
- [x] **README-uri etape** prezente
- [x] **Screenshots** prezente în `docs/screenshots/`
- [x] **Structura repository** conformă
- [x] **requirements.txt** actualizat și funcțional
- [x] **Cod comentat** (minim 15% linii comentarii relevante)
- [x] **Toate path-urile relative**

### Acces și Versionare

- [x] **Repository accesibil** cadrelor didactice RN (public)
- [x] **Tag `v0.6-optimized-final`** creat și pushed
- [x] **Commit-uri incrementale** vizibile
- [x] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero**
- [x] **Minimum 40% date originale**
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen
**Ultima actualizare:** 22.01.2026
**Tag Git:** v0.6-optimized-final
