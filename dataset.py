import os
import numpy as np
from PIL import Image  # Para cargar y manipular imágenes

# Mapa de etiquetas:
# - Asocia números del 0 al 61 con caracteres correspondientes.
# - Incluye: dígitos (0-9), letras mayúsculas (A-Z) y letras minúsculas (a-z).
LABEL_MAP = {
    **{i: str(i) for i in range(10)},  # Números: 0-9
    **{i + 10: chr(i + ord('A')) for i in range(26)},  # Mayúsculas: A-Z
    **{i + 36: chr(i + ord('a')) for i in range(26)},  # Minúsculas: a-z
}

# Tamaño al que se redimensionarán las imágenes
IMG_SIZE = (32, 32)


def extract_label_from_old_format(img_name):
    """
    Extrae etiquetas de archivos con formato "antiguo".
    Ejemplo de nombres de archivo antiguos: "img001.png", "img011.png", etc.

    - Las primeras 10 imágenes corresponden a dígitos (0-9).
    - Las siguientes 26 corresponden a letras mayúsculas (A-Z).
    - Las últimas 26 corresponden a letras minúsculas (a-z).

    Parámetros:
    - img_name: Nombre del archivo de la imagen.

    Retorna:
    - Etiqueta correspondiente (entero) o None si hay un error.
    """
    try:
        if img_name.startswith("img"):  # Verificar que el nombre siga el formato esperado
            img_index = int(img_name[3:6])  # Extraer índice numérico del nombre
            if 1 <= img_index <= 10:  # Dígitos 0-9
                return img_index - 1
            elif 11 <= img_index <= 36:  # Letras mayúsculas A-Z
                return img_index - 11 + 10
            elif 37 <= img_index <= 62:  # Letras minúsculas a-z
                return img_index - 37 + 36
            else:
                raise ValueError(f"Índice fuera de rango: {img_index}")
        else:
            raise ValueError(f"Formato inesperado: {img_name}")
    except Exception as e:
        print(f"Error al procesar el archivo {img_name}: {e}")
        return None


def extract_label_from_new_format(img_name):
    """
    Extrae etiquetas de archivos con formato "nuevo".
    Ejemplo de nombres de archivo nuevos: "image_0.png", "image_10.png", etc.

    - La etiqueta está codificada en el nombre del archivo como un número al final del nombre.

    Parámetros:
    - img_name: Nombre del archivo de la imagen.

    Retorna:
    - Etiqueta correspondiente (entero).
    """
    try:
        # Extraer la etiqueta como un número desde la última parte del nombre
        numeric_label = int(img_name.split('_')[-1].split('.')[0])
        return numeric_label
    except ValueError:
        raise ValueError(f"Error al procesar el nombre del archivo: {img_name}")


def load_dataset(dataset_path, dataset_type="new"):
    """
    Carga un conjunto de datos desde una carpeta específica.

    Pasos:
    1. Lee imágenes desde la carpeta especificada.
    2. Convierte las imágenes a escala de grises y las redimensiona a (32, 32).
    3. Extrae etiquetas de las imágenes usando el formato de nombre correspondiente.
    4. Normaliza las imágenes al rango [0, 1].

    Parámetros:
    - dataset_path: Ruta de la carpeta que contiene las imágenes.
    - dataset_type: Tipo de dataset ("new" o "old").
      - "new": Usa `extract_label_from_new_format` para obtener etiquetas.
      - "old": Usa `extract_label_from_old_format` para obtener etiquetas.

    Retorna:
    - images: Arreglo de imágenes normalizadas con forma (N, 32, 32, 1).
    - labels: Arreglo de etiquetas correspondientes.
    """
    images = []  # Lista para almacenar las imágenes procesadas
    labels = []  # Lista para almacenar las etiquetas

    print(f"Cargando dataset desde: {dataset_path}, Tipo: {dataset_type}")
    for img_name in sorted(os.listdir(dataset_path)):  # Iterar sobre los archivos en la carpeta
        img_path = os.path.join(dataset_path, img_name)  # Ruta completa de la imagen
        try:
            # Abrir la imagen y convertirla a escala de grises
            img = Image.open(img_path).convert("L")
            # Redimensionar la imagen al tamaño definido (32x32)
            img = img.resize(IMG_SIZE)
            # Normalizar los valores de la imagen a [0, 1] y agregarla a la lista
            images.append(np.array(img, dtype=np.float32) / 255.0)

            # Extraer la etiqueta según el tipo de dataset
            if dataset_type == "old":
                label = extract_label_from_old_format(img_name)
            elif dataset_type == "new":
                label = extract_label_from_new_format(img_name)
            else:
                raise ValueError("Tipo de dataset desconocido.")

            if label is not None:
                # Agregar la etiqueta a la lista
                labels.append(label)
                print(f"Imagen: {img_name}, Etiqueta asignada: {LABEL_MAP[label]}")
        except Exception as e:
            # Manejar errores durante el procesamiento de archivos
            print(f"Error al procesar el archivo {img_name}: {e}")

    # Convertir las listas a arreglos de numpy
    # - Las imágenes tendrán forma (N, 32, 32, 1), donde N es el número de imágenes.
    # - Las etiquetas tendrán forma (N,).
    return np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1), np.array(labels)
