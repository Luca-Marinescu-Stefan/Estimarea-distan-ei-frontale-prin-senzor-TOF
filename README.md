# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Marinescu Luca-Stefan  
**Data:** 27/11/2025

---

## Introducere

Acest proiect implementează un sistem inteligent de măsurare a distanței (SIA) care combină un senzor Time-of-Flight (VL53L0X) cu o rețea neuronală artificială. Scopul este corectarea erorilor neliniare și reducerea zgomotului de măsurare, în special pe suprafețe reflectorizante și în condiții variabile de lumină.

### Obiective

* Achiziția datelor brute (distanță, intensitate semnal, temperatură).
* Creșterea preciziei măsurătorilor cu 15–25%.
* Reducerea zgomotului de măsurare cu aprox. 40%.

## Arhitectura Sistemului

1. **Hardware:** Senzor VL53L0X + Microcontroler (Arduino/RPi).
2. **Software:** Python pentru preprocesare și TensorFlow/Keras pentru modelul neuronal (MLP).
3. **Flux date:** Senzor -> Procesare Serială -> Preprocesare -> Rețea Neuronală -> Distanță Estimată.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```
<img width="603" height="662" alt="image" src="https://github.com/user-attachments/assets/9c80709a-8f99-4ce3-9085-6f4cfc7d563f" />

