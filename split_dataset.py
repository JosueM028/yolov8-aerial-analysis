import os
import shutil
import random
import glob

# --- CONFIGURACIÓN ---
SOURCE_DIR = 'aerial-cars-dataset'      # Carpeta origen (sucia)
OUTPUT_DIR = 'dataset_final_yolo'       # Carpeta destino (limpia y organizada)
SPLIT_RATIO = 0.8                       # 80% Train, 20% Val

def is_valid_yolo_txt(txt_path):
    """
    Verifica si un archivo .txt tiene formato YOLO válido y no es basura.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if not lines: return False # Archivo vacío
            
            # Verificar la primera línea
            parts = lines[0].strip().split()
            if len(parts) < 5: return False
            
            # Intentar convertir el primer valor a int (Class ID) y el resto a float
            int(parts[0]) 
            [float(x) for x in parts[1:5]]
            
            return True
    except:
        return False

def split_and_clean():
    # 1. Crear estructura de carpetas YOLO
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for sub in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # 2. Obtener lista de imágenes
    # Buscamos tanto jpg como png por si acaso
    images = glob.glob(os.path.join(SOURCE_DIR, "*.jpg")) + glob.glob(os.path.join(SOURCE_DIR, "*.png"))
    random.shuffle(images)
    
    print(f"Total imágenes encontradas: {len(images)}")
    
    valid_pairs = []

    # 3. Filtrar pares válidos
    print("Verificando integridad de etiquetas...")
    for img_path in images:
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        
        # Buscar txt correspondiente
        txt_path = os.path.join(SOURCE_DIR, name_no_ext + ".txt")
        
        if os.path.exists(txt_path) and is_valid_yolo_txt(txt_path):
            valid_pairs.append((img_path, txt_path))
        else:
            # Opcional: imprimir qué se descarta
            # print(f"Descartado (sin etiqueta válida): {base_name}")
            pass

    print(f"Imágenes válidas para entrenamiento: {len(valid_pairs)}")

    # 4. Dividir
    split_idx = int(len(valid_pairs) * SPLIT_RATIO)
    train_set = valid_pairs[:split_idx]
    val_set = valid_pairs[split_idx:]

    # 5. Mover archivos
    def copy_files(dataset, subset_name):
        print(f"Copiando {len(dataset)} archivos a {subset_name}...")
        for img_src, txt_src in dataset:
            # Definir destinos
            img_dst = os.path.join(OUTPUT_DIR, 'images', subset_name, os.path.basename(img_src))
            txt_dst = os.path.join(OUTPUT_DIR, 'labels', subset_name, os.path.basename(txt_src))
            
            shutil.copy(img_src, img_dst)
            shutil.copy(txt_src, txt_dst)

    copy_files(train_set, 'train')
    copy_files(val_set, 'val')

    print("\n¡Proceso completado!")
    print(f"Tu dataset limpio está en la carpeta: {OUTPUT_DIR}/")

if __name__ == "__main__":
    split_and_clean()