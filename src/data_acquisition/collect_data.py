# Cod pentru achiziția datelor de la senzorul VL53L0X
import csv
import random

# Simulam datele (sau citim de la senzor real)
def colectare():
    print("Se colectează date brute (Distanță, Intensitate, Temperatură)...")
    with open('../../data/raw/dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distanta_bruta', 'intensitate', 'temperatura', 'distanta_reala'])
        # Generăm 10 rânduri de test
        for i in range(10):
            writer.writerow([500 + i, 200, 24.5, 500])
    print("Fisier salvat in data/raw/dataset.csv")

if __name__ == "__main__":
    colectare()
