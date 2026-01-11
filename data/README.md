# Descrierea dataset-ului

Acest folder conține datele utilizate pentru antrenarea și evaluarea rețelei neuronale pentru estimarea distanței cu senzor TOF.

Structură:

- `data/raw/` – date brute colectate de la senzor (CSV)
- `data/processed/` – date curățate și transformate
- `data/train/`, `data/validation/`, `data/test/` – split-urile finale

Conținutul fiecărui fișier CSV:
- `timestamp` – marcaj temporal
- `distance_raw` – măsurătoarea brută de la senzor (mm)
- `signal_strength` – intensitatea semnalului
- `temperature` – temperatură senzor (opțional)
- `distance_ref` – distanța de referință măsurată cu un etalon (dacă există)

Analiză și EDA (rezumat):
- Calcularea statisticilor descriptive (medie, mediană, std, min/max).
- Vizualizarea distribuțiilor (histograme) și corelațiilor între feature-uri.
- Detectarea și raportarea valorilor lipsă pe coloană.
- Identificarea outlierilor (IQR) și a oricăror anomalii consistente.

Preprocesare (implementată în `src/preprocessing/process_data.py`):
- Eliminarea duplicatelor.
- Tratarea valorilor lipsă: imputare cu mediană pentru feature-uri numerice, eliminare pentru coloane cu prea multe lipsuri.
- Filtrare/limitare outlieri pe baza percentilelor (de ex. 1–99%).
- Normalizare/standardizare aplicată pe `train` și salvată parametric în `config/`.
- Împărțirea în `train/` (70–80%), `validation/` (10–15%) și `test/` (10–15%) — stratificare unde e relevant.

Fișiere generate:
- Datele procesate în `data/processed/`.
- Split-urile în folderele dedicate (`train`, `validation`, `test`).
- Configurări de preprocesare (scaleri etc.) în `config/` (opțional).

Recomandări:
- Rulați EDA local înainte de orice antrenament.
- Păstrați statisticile de scalare generate pe `train` și folosiți-le pe celelalte seturi.

Pentru detalii de etapă și documentație extinsă, consultați:
- [Etapa 3 – Analiză și pregătire (documentație detaliată)](../docs/etapa3.md)
