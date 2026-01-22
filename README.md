
---

# 📘 README – Proiect Final RN (SIA ToF)

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Marinescu Luca-Stefan (634AB)  
**Data:** 22/01/2026

## Rezumat
Proiectul implementează un sistem inteligent pentru estimarea distanței frontale cu senzor Time‑of‑Flight (VL53L0X), folosind preprocesare + model ML pentru corecția erorilor neliniare și stabilizarea măsurătorilor.

## Stack Tehnologic
- Python: pandas, numpy, scikit‑learn, TensorFlow/Keras, Streamlit, matplotlib

## Rulare rapidă
1. Instalează dependențe: `pip install -r requirements.txt`
2. Rulează UI: `python -m streamlit run src/app/main.py`

## Pipeline complet (PowerShell / VS Code)
`python src/preprocessing/combine_datasets.py ; python src/preprocessing/data_cleaner.py ; python src/preprocessing/feature_engineering.py ; python src/preprocessing/process_data.py ; python src/preprocessing/data_splitter.py ; python src/neural_network/train.py --backend keras --epochs 50 --batch-size 32 --early-stopping --reduce-lr ; python src/neural_network/evaluate.py --model models/trained_model.h5`

## Rezultate (test set)
- Accuracy: 100.00%
- F1‑macro: 1.00
- Latență inferență: ~0.349 ms/sample

## Structură proiect (scurt)
- [src/](src/) – preprocesare, model, UI
- [data/](data/) – raw/processed/train/val/test
- [results/](results/) – metrici și experimente
- [docs/](docs/) – diagrame și capturi

## Documentație detaliată
- [data/README.md](data/README.md)
- [docs/etapa3.md](docs/etapa3.md)
- [README_Etapa4_Arhitectura_SIA.md](README_Etapa4_Arhitectura_SIA.md)
- [README_Etapa5_Antrenare_RN.md](README_Etapa5_Antrenare_RN.md)
- [etapa6_optimizare_concluzii.md](etapa6_optimizare_concluzii.md)
- [NUME_Prenume_Grupa_README_Proiect_RN.md](NUME_Prenume_Grupa_README_Proiect_RN.md)

---

