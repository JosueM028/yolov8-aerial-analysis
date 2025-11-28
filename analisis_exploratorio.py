import cv2
import matplotlib.pyplot as plt
import glob
import os
import random

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATASET_DIR = 'aerial-cars-dataset' 
IMG_EXTENSION = '*.jpg' 

def parse_yolo_line(line, img_width, img_height):
    parts = line.strip().split()
    
    # --- FILTRO DE SEGURIDAD ---
    # Si la linea no tiene al menos 5 partes o el primero no es un numero, es basura
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0]) # Intentar convertir a entero
    except ValueError:
        return None # Si falla (ej: es un nombre de archivo), ignorar
        
    x_c_norm, y_c_norm, w_norm, h_norm = map(float, parts[1:])

    # Conversión
    w_pixel = int(w_norm * img_width)
    h_pixel = int(h_norm * img_height)
    x_center_pixel = int(x_c_norm * img_width)
    y_center_pixel = int(y_c_norm * img_height)

    x_min = int(x_center_pixel - (w_pixel / 2))
    y_min = int(y_center_pixel - (h_pixel / 2))
    
    return class_id, x_min, y_min, w_pixel, h_pixel

def analyze_dataset():
    txt_files = glob.glob(os.path.join(DATASET_DIR, '*.txt'))
    
    if not txt_files:
        print(f"ERROR: No se encontraron archivos .txt en '{DATASET_DIR}'")
        return

    print(f"Analizando {len(txt_files)} archivos... (filtrando errores)")
    
    class_counts = {}
    total_valid_objects = 0
    
    # Usamos encoding='utf-8' y errors='ignore' para evitar el crash en Windows
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        # Verificar si es una clase válida (entero)
                        c_id = int(parts[0])
                        class_counts[c_id] = class_counts.get(c_id, 0) + 1
                        total_valid_objects += 1
                    except ValueError:
                        # Si la primera palabra no es un numero, ignoramos la linea
                        continue

    print("\n" + "="*30)
    print(" ESTADÍSTICAS REALES (LIMPIAS)")
    print("="*30)
    print(f"Total de objetos válidos: {total_valid_objects}")
    print("\nDistribución por Clase:")
    
    # Ordenar por ID de clase
    for cls in sorted(class_counts.keys()):
        print(f" -> Clase ID {cls}: {class_counts[cls]} instancias")

    print("\nGenerando visualización...")
    visualize_random_sample()

def visualize_random_sample():
    img_files = glob.glob(os.path.join(DATASET_DIR, IMG_EXTENSION))
    if not img_files:
        print("No imágenes encontradas.")
        return

    # Intentar hasta encontrar una imagen válida con etiquetas válidas
    max_retries = 10
    for _ in range(max_retries):
        sample_img_path = random.choice(img_files)
        label_path = sample_img_path.replace('.jpg', '.txt').replace('.png', '.txt')
        
        if os.path.exists(label_path):
            break
    
    # Cargar imagen
    img = cv2.imread(sample_img_path)
    if img is None:
        print("Error leyendo la imagen.")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    print(f"Visualizando: {os.path.basename(sample_img_path)}")

    # Dibujar
    found_valid_labels = False
    with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            result = parse_yolo_line(line, w, h)
            if result:
                cls, x, y, bw, bh = result
                cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
                cv2.putText(img, f"ID:{cls}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                found_valid_labels = True
    
    if not found_valid_labels:
        print("La imagen seleccionada no tenía etiquetas válidas formato YOLO, intentando otra...")
        return

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Muestra: {os.path.basename(sample_img_path)}")
    plt.show()

if __name__ == "__main__":
    analyze_dataset()