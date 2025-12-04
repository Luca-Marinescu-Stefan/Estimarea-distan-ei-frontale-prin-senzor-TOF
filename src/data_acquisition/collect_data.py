import serial
import time
import csv
import os

# --- CONFIGURARE ---
SERIAL_PORT = 'COM3'  # <--- ATENȚIE: Verifică în Arduino IDE ce port ai (ex: COM4, COM5)
BAUD_RATE = 9600

# Calculăm calea absolută ca să nu avem erori indiferent de unde rulăm scriptul
# Mergem 3 nivele mai sus: data_acquisition -> src -> root -> data -> raw
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, '..', '..', 'data', 'raw')
OUTPUT_FILE = os.path.join(output_dir, 'dataset_colectat.csv')

SAMPLES_PER_DISTANCE = 50  # Câte citiri luăm per distanță

def collect_data():
    # Ne asigurăm că folderul există
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Am creat folderul: {output_dir}")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Așteptăm resetarea Arduino
        print(f"Conectat la {SERIAL_PORT}")
        print(f"Datele vor fi salvate în: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Eroare: Nu pot deschide portul serial {SERIAL_PORT}.")
        print(f"Detalii: {e}")
        print("Verifică dacă Arduino e conectat și dacă ai închis monitorul serial din Arduino IDE.")
        return

    # Deschidem fișierul CSV (append mode)
    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Scriem antetul doar dacă fișierul e gol
        if file.tell() == 0:
            writer.writerow(["Distanta_Bruta_Senzor", "Intensitate", "Distanta_Reala_Reference"])

        while True:
            try:
                user_input = input("\nIntrodu distanța reală (mm) sau 'q' pentru ieșire: ")
                if user_input.lower() == 'q':
                    break
                
                real_distance = float(user_input)
                print(f"Colectez {SAMPLES_PER_DISTANCE} de mostre pentru {real_distance}mm...")

                count = 0
                while count < SAMPLES_PER_DISTANCE:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if line and "TIMEOUT" not in line and "Eroare" not in line:
                            parts = line.split(',')
                            if len(parts) >= 1:
                                raw_dist = parts[0]
                                intensity = parts[1] if len(parts) > 1 else 0
                                
                                writer.writerow([raw_dist, intensity, real_distance])
                                count += 1
                                # Feedback vizual la fiecare 10 citiri
                                if count % 10 == 0:
                                    print(f".", end="", flush=True)
                    except UnicodeDecodeError:
                        continue # Ignorăm erorile de biți pe serial
                
                print(f"\nTerminat pentru {real_distance}mm.")

            except ValueError:
                print("Te rog introdu un număr valid.")
            except KeyboardInterrupt:
                break
    
    ser.close()
    print("\nColectare finalizată.")

if __name__ == "__main__":
    collect_data()
