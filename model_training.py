import numpy as np
import tensorflow as tf
from dataset import load_dataset
from model import create_optimized_model

def rotate_image(img, angle):
    """
    Rota una imagen manualmente utilizando TensorFlow.
    
    Parámetros:
    - img: Imagen a rotar (tensor de TensorFlow).
    - angle: Ángulo de rotación en radianes.
    
    Retorna:
    - Imagen rotada.
    """
    angle = tf.constant(angle, dtype=tf.float32)  # Convertir el ángulo a constante de TensorFlow
    # Rotar la imagen en múltiplos de 90 grados
    return tf.image.rot90(img, k=int(angle // (np.pi / 2)))


def augment_images(images):
    """
    Aplica augmentación de datos a un conjunto de imágenes utilizando TensorFlow.
    
    Operaciones realizadas:
    - Volteo horizontal aleatorio.
    - Ajuste aleatorio de brillo.
    - Ajuste aleatorio de contraste.
    - Rotación manual de 45 grados.
    
    Parámetros:
    - images: Conjunto de imágenes (numpy array o lista).
    
    Retorna:
    - Imágenes augmentadas como un numpy array.
    """
    augmented_images = []  # Lista para almacenar las imágenes augmentadas
    for img in images:
        # Aplicar operaciones de augmentación
        img = tf.image.random_flip_left_right(img)  # Volteo horizontal aleatorio
        img = tf.image.random_brightness(img, max_delta=0.1)  # Ajuste de brillo aleatorio
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)  # Ajuste de contraste aleatorio
        img = rotate_image(img, np.pi / 4)  # Rotación fija de 45 grados
        augmented_images.append(img.numpy())  # Convertir a formato numpy
    return np.array(augmented_images)


def train_model():
    """
    Entrena un modelo OCR (Reconocimiento Óptico de Caracteres) utilizando TensorFlow.
    
    Pasos:
    1. Carga de datasets de entrenamiento y validación.
    2. Augmentación de datos en el conjunto de entrenamiento.
    3. Creación del modelo con regularización (Dropout y L2).
    4. Configuración de entrenamiento con Early Stopping.
    5. Entrenamiento del modelo.
    6. Evaluación de la precisión en el conjunto de validación.
    7. Guardado del modelo entrenado.
    
    Retorna:
    - Historial de entrenamiento para análisis.
    """
    # Cargar el conjunto de entrenamiento
    print("Cargando dataset de entrenamiento...")
    train_images, train_labels = load_dataset("./dataset/DatasetCompleto2", dataset_type="new")

    # Cargar el conjunto de validación
    print("Cargando dataset de validación...")
    val_images, val_labels = load_dataset("./dataset/datasetCompleto", dataset_type="old")

    # Imprimir las dimensiones de los datasets para verificar la carga
    print(f"Dataset de entrenamiento: {train_images.shape}, {train_labels.shape}")
    print(f"Dataset de validación: {val_images.shape}, {val_labels.shape}")

    # Augmentación de datos (solo en el conjunto de entrenamiento)
    print("Aplicando augmentación de datos al dataset de entrenamiento...")
    train_images = augment_images(train_images)

    # Crear el modelo con regularización (Dropout y L2)
    model = create_optimized_model(
        input_shape=(32, 32, 1),  # Dimensiones de las imágenes de entrada (32x32, escala de grises)
        num_classes=62,  # Total de clases (10 dígitos + 26 letras mayúsculas + 26 letras minúsculas)
        use_dropout=True,  # Activar Dropout para prevenir sobreajuste
        l2_reg=0.001  # Regularización L2 para reducir el impacto de pesos grandes
    )

    # Configurar Early Stopping para detener el entrenamiento si no hay mejoras en validación
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Monitorizar la pérdida en el conjunto de validación
        patience=5,  # Detener si no mejora después de 5 épocas
        restore_best_weights=True  # Restaurar los mejores pesos al final
    )

    # Compilar el modelo con un learning rate reducido
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Learning rate bajo para más estabilidad
        loss="sparse_categorical_crossentropy",  # Función de pérdida para etiquetas enteras
        metrics=["accuracy"]  # Métrica de precisión
    )

    # Entrenar el modelo con el conjunto de entrenamiento y validación
    print("Entrenando el modelo con regularización y Early Stopping...")
    history = model.fit(
        train_images,  # Imágenes de entrenamiento
        train_labels,  # Etiquetas de entrenamiento
        validation_data=(val_images, val_labels),  # Validación durante el entrenamiento
        epochs=10,  # Máximo de 10 épocas
        batch_size=32,  # Tamaño del batch durante el entrenamiento
        callbacks=[early_stopping]  # Callback para Early Stopping
    )

    # Guardar el modelo entrenado en un archivo
    model.save("ocr_model.h5")
    print("Modelo guardado.")

    # Calcular la precisión en el conjunto de validación
    print("Calculando precisión en el conjunto de validación...")
    val_predictions = model.predict(val_images)  # Predecir en el conjunto de validación
    val_predicted_classes = np.argmax(val_predictions, axis=1)  # Obtener la clase predicha

    # Comparar las predicciones con las etiquetas reales
    correct_predictions = np.sum(val_predicted_classes == val_labels)
    total_predictions = len(val_labels)
    accuracy = (correct_predictions / total_predictions) * 100  # Calcular precisión como porcentaje

    print(f"Precisión en el conjunto de validación: {accuracy:.2f}%")

    # Retornar el historial de entrenamiento para análisis posterior
    return history
