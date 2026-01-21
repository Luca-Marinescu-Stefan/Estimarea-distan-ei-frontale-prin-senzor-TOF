# Modul Neural Network

## Scripturi
- `model.py` – definește model Keras simplu și poate salva `models/untrained_model.h5`.
- `train.py` – antrenare model (backend `sklearn` sau `keras`).
- `evaluate.py` – evaluare model și metrici.
- `optimize.py` – experimente de optimizare + model optimizat.
- `visualize.py` – grafice finale pentru Etapa 6.

## Rulare (Etapa 5)
```
python src/neural_network/train.py --backend keras --epochs 50 --batch-size 32 --early-stopping
python src/neural_network/evaluate.py --model models/trained_model.h5
```

## Rulare (Etapa 6)
```
python src/neural_network/optimize.py
python src/neural_network/evaluate.py --model models/optimized_model.joblib --detailed --out-metrics results/final_metrics.json
python src/neural_network/visualize.py
```
