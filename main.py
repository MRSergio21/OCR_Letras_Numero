import tkinter as tk
from tkinter import filedialog  # Para diálogos de selección de archivos y carpetas
import tensorflow as tf
from model_training import train_model  # Función para entrenar el modelo
from predict import predict_image, predict_folder  # Funciones para predicción en imágenes y carpetas


def train_interface():
    """
    Llama al entrenamiento manualmente a través de la interfaz gráfica.
    Permite al usuario ejecutar el proceso de entrenamiento desde un botón en la interfaz.
    """
    train_model()  # Llama a la función de entrenamiento definida en model_training.py


def predict_letter():
    """
    Llama a la predicción de una letra desde una imagen seleccionada por el usuario.
    - Abre un diálogo para seleccionar la imagen.
    - Muestra la predicción en la consola.
    """
    # Mostrar un diálogo para seleccionar un archivo de imagen
    image_path = filedialog.askopenfilename(title="Selecciona una imagen")
    if image_path:
        # Llamar a la función predict_image para predecir una letra
        prediction = predict_image(image_path, "letter")
        print(f"Predicción (Letra): {prediction}")


def predict_number():
    """
    Llama a la predicción de un número desde una imagen seleccionada por el usuario.
    - Abre un diálogo para seleccionar la imagen.
    - Muestra la predicción en la consola.
    """
    # Mostrar un diálogo para seleccionar un archivo de imagen
    image_path = filedialog.askopenfilename(title="Selecciona una imagen")
    if image_path:
        # Llamar a la función predict_image para predecir un número
        prediction = predict_image(image_path, "number")
        print(f"Predicción (Número): {prediction}")


def predict_phrase():
    """
    Llama a la predicción de una frase completa desde una imagen seleccionada por el usuario.
    - Abre un diálogo para seleccionar la imagen.
    - Muestra la predicción en la consola.
    """
    # Mostrar un diálogo para seleccionar un archivo de imagen
    image_path = filedialog.askopenfilename(title="Selecciona una imagen")
    if image_path:
        # Llamar a la función predict_image para predecir una frase
        prediction = predict_image(image_path, "phrase")
        print(f"Predicción (Frase): {prediction}")


def predict_from_folder():
    """
    Llama a la predicción de múltiples imágenes desde una carpeta seleccionada por el usuario.
    - Abre un diálogo para seleccionar la carpeta.
    - Calcula la precisión en las imágenes seleccionadas usando el modelo cargado.
    """
    # Cargar el modelo previamente entrenado
    model = tf.keras.models.load_model("ocr_model.h5")  # Ruta del modelo guardado
    # Mostrar un diálogo para seleccionar una carpeta
    folder_path = filedialog.askdirectory(title="Selecciona la carpeta de imágenes")
    if folder_path:
        # Llamar a la función predict_folder para realizar predicciones en la carpeta
        accuracy = predict_folder(folder_path, model)
        if accuracy > 0:
            print(f"Precisión calculada: {accuracy:.2f}%")
        else:
            print("No se procesaron imágenes válidas o precisión es 0%.")


# Entrenamiento automático al inicio
print("Iniciando entrenamiento automático del modelo...")
train_model()  # Llama al entrenamiento automático al iniciar el programa
print("Entrenamiento completado.")


# Configuración de la interfaz gráfica
root = tk.Tk()  # Crear la ventana principal
root.title("OCR Predictor")  # Título de la ventana

# Botón para predecir letras
button_letter = tk.Button(root, text="Predecir Letra", command=predict_letter)
button_letter.pack()  # Agregar el botón a la ventana

# Botón para predecir números
button_number = tk.Button(root, text="Predecir Número", command=predict_number)
button_number.pack()  # Agregar el botón a la ventana

# Botón para predecir frases
button_phrase = tk.Button(root, text="Predecir Frase", command=predict_phrase)
button_phrase.pack()  # Agregar el botón a la ventana

# Botón para predecir desde una carpeta
button_folder = tk.Button(root, text="Predecir Carpeta", command=predict_from_folder)
button_folder.pack()  # Agregar el botón a la ventana

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
