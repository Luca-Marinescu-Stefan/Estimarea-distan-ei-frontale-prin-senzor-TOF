# Modul Data Acquisition

Acest modul generează/achiziționează date brute pentru senzorul TOF.

## Scripturi
- `collect_data.py` – simulare achiziție tip „senzor real” (10 rânduri demo).
- `generate.py` – generator de date sintetice cu parametri configurabili.

## Rulare recomandată
```
python src/data_acquisition/generate.py --rows 500 --out data/raw/sample_raw.csv
```

## Output
CSV cu coloanele:
- `timestamp`
- `distance_raw`
- `signal_strength`
- `temperature`
- `distance_ref`

## Notă contribuție originală (Etapa 4)
Folosiți fișierele din `data/generated/` ca date originale (≥40%).
