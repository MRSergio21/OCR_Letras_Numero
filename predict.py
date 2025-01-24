import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(img):
    """
    Aplica preprocesamiento a una imagen para mejorar la segmentación y predicción.
    - Suaviza la imagen para reducir ruido utilizando un filtro Gaussiano.
    - Mejora el contraste de la imagen utilizando ecualización de histograma.
    """
    # Suavizado para reducir ruido y facilitar la segmentación
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Mejora el contraste en imágenes en escala de grises
    img = cv2.equalizeHist(img)
    return img


def segment_image(image_path):
    """
    Segmenta caracteres individuales de una imagen en escala de grises.
    Pasos principales:
    - Cargar la imagen en escala de grises.
    - Preprocesar la imagen (suavizado y ajuste de contraste).
    - Aplicar binarización y operaciones morfológicas para resaltar caracteres.
    - Detectar contornos para identificar posibles caracteres.
    - Recortar y redimensionar cada carácter identificado.
    """
    # Cargar la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)

    # Aplicar binarización inversa para destacar caracteres (blanco sobre fondo negro)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Mejorar los bordes usando operaciones morfológicas (dilatación y erosión)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Detectar contornos de los caracteres en la imagen binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []  # Lista para almacenar imágenes de caracteres
    bounding_boxes = []  # Lista para almacenar las coordenadas de los cuadros delimitadores
    for contour in contours:
        # Obtener cuadro delimitador del contorno
        x, y, w, h = cv2.boundingRect(contour)
        # Filtrar contornos pequeños para eliminar ruido
        if w > 5 and h > 5:
            # Recortar y redimensionar el carácter identificado
            char_img = img[y:y + h, x:x + w]
            char_img = cv2.resize(char_img, (32, 32)) / 255.0  # Normalizar valores a rango [0, 1]
            char_images.append(char_img)
            bounding_boxes.append((x, y, w, h))

    # Ordenar los caracteres por su posición horizontal en la imagen
    sorted_data = sorted(zip(bounding_boxes, char_images), key=lambda b: b[0][0])
    if sorted_data:
        bounding_boxes, char_images = zip(*sorted_data)
        # Retornar las imágenes y sus cuadros delimitadores
        return np.array(char_images).reshape(-1, 32, 32, 1), bounding_boxes
    else:
        return np.array([]), []


def predict_image(image_path, prediction_type, model=None):
    """
    Realiza predicciones para imágenes de caracteres o frases completas.
    - Carga y preprocesa la imagen.
    - Realiza segmentación si se trata de una frase.
    - Usa el modelo para predecir caracteres y decodificarlos.
    """
    if model is None:
        # Cargar modelo OCR preentrenado si no se proporciona uno
        model = tf.keras.models.load_model("ocr_model.h5")

    # Leer la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if prediction_type == "phrase":
        # Realizar segmentación de caracteres en la imagen
        char_images, bounding_boxes = segment_image(image_path)
        if char_images.size == 0:
            print("No se detectaron caracteres en la imagen.")
            return ""

        # Realizar predicciones para cada carácter segmentado
        predictions = model.predict(char_images)
        predicted_classes = np.argmax(predictions, axis=1)

        # Decodificar las clases predichas a caracteres alfanuméricos
        predicted_chars = []
        for cls in predicted_classes:
            if 0 <= cls <= 9:  # Números
                predicted_chars.append(chr(cls + ord('0')))
            elif 10 <= cls <= 35:  # Letras mayúsculas
                predicted_chars.append(chr(cls - 10 + ord('A')))
            elif 36 <= cls <= 61:  # Letras minúsculas
                predicted_chars.append(chr(cls - 36 + ord('a')))
            else:
                predicted_chars.append("?")  # Clase desconocida

        predicted_phrase = "".join(predicted_chars)

        # Mostrar resultados visuales con cuadros delimitadores y caracteres predichos
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h), char in zip(bounding_boxes, predicted_chars):
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img_color, char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicción (Frase): {predicted_phrase}")
        plt.axis("off")
        plt.show()

        print(f"Predicción (Frase): {predicted_phrase}")
        return predicted_phrase
    else:
        # Predicción para una sola letra o número
        img_resized = cv2.resize(img, (32, 32)) / 255.0  # Normalizar
        img_resized = img_resized.reshape(1, 32, 32, 1)
        prediction = model.predict(img_resized)
        predicted_class = np.argmax(prediction)

        # Decodificar predicción según el tipo (letra o número)
        if prediction_type == "letter":
            if 0 <= predicted_class <= 9:
                predicted_char = chr(predicted_class + ord('0'))
            elif 10 <= predicted_class <= 35:
                predicted_char = chr(predicted_class - 10 + ord('A'))
            elif 36 <= predicted_class <= 61:
                predicted_char = chr(predicted_class - 36 + ord('a'))
            else:
                predicted_char = "Desconocido"
        elif prediction_type == "number":
            if 0 <= predicted_class <= 9:
                predicted_char = chr(predicted_class + ord('0'))
            else:
                predicted_char = "No es un número"
        else:
            predicted_char = "Desconocido"

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicción: {predicted_char}")
        plt.axis("off")
        plt.show()

        print(f"Predicción: Clase {predicted_class}, Carácter: {predicted_char}")
        return predicted_char


def predict_folder(folder_path, model):
    """
    Procesa un directorio completo de imágenes para realizar predicciones.
    - Identifica etiquetas verdaderas desde los nombres de archivo si es posible.
    - Calcula precisión si se disponen de etiquetas.
    """
    predictions = []
    true_labels = []
    correct_predictions = 0
    total_predictions = 0

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        print(f"Procesando: {img_name}")

        if "label_" in img_name:
            # Extraer etiqueta verdadera desde el nombre del archivo
            label = img_name.split("label_")[1].split(".")[0]
            prediction_type = "letter" if len(label) == 1 and label.isalnum() else "number"
        else:
            # Inferir etiqueta desde la primera letra del nombre
            potential_label = img_name[0]
            if potential_label.isdigit():
                prediction_type = "number"
                label = potential_label
            elif potential_label.isalpha():
                prediction_type = "letter"
                label = potential_label
            else:
                print(f"No se pudo determinar la etiqueta para {img_name}.")
                continue
            
        try:
            # Realizar predicción para la imagen
            prediction = predict_image(img_path, prediction_type, model)
            predictions.append(prediction)

            # Comparar predicción con la etiqueta verdadera
            if "label_" in img_name:
                true_label = label
                true_labels.append(true_label)
                if prediction == true_label:
                    correct_predictions += 1
                total_predictions += 1
        except Exception as e:
            print(f"Error procesando {img_name}: {e}")

    # Calcular precisión
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Precisión: {accuracy:.2f}%")
        return accuracy
    else:
        print("No se procesaron imágenes válidas.")
        return 0