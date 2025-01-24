import tensorflow as tf
from tensorflow.keras import layers, models

def create_optimized_model(input_shape, num_classes, use_dropout=False, l2_reg=0.001):
    """
    Crea y compila un modelo de red neuronal convolucional optimizado para clasificación de imágenes.
    
    Parámetros:
    - input_shape: Tuple que define la forma de entrada de las imágenes (altura, ancho, canales).
    - num_classes: Número de clases en el problema de clasificación.
    - use_dropout: Booleano para habilitar/deshabilitar el uso de Dropout.
    - l2_reg: Valor de regularización L2 para prevenir sobreajuste.
    
    Retorna:
    - Un modelo compilado de Keras listo para entrenamiento.
    """

    # Inicializamos un modelo secuencial, que permite añadir capas en orden
    model = models.Sequential([
        # Primera capa convolucional:
        # - Detecta características iniciales de la imagen (bordes, texturas, etc.).
        # - Utiliza 16 filtros de tamaño 3x3 con activación ReLU para añadir no linealidad.
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        # Normalización por lotes:
        # - Acelera el entrenamiento al mantener activaciones escaladas de manera consistente.
        layers.BatchNormalization(),
        # Capa de MaxPooling 2x2:
        # - Reduce la dimensionalidad espacial, manteniendo las características más importantes.
        layers.MaxPooling2D((2, 2)),

        # Segunda capa convolucional:
        # - Incrementa el número de filtros (64) para capturar características más complejas.
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Aplicamos normalización por lotes nuevamente.
        layers.BatchNormalization(),
        # MaxPooling para reducir dimensionalidad.
        layers.MaxPooling2D((2, 2)),

        # Tercera capa convolucional:
        # - Incrementamos aún más los filtros a 128, para detectar patrones avanzados.
        layers.Conv2D(128, (3, 3), activation='relu'),
        # Normalización por lotes.
        layers.BatchNormalization(),
        # MaxPooling 2x2 para reducir la representación espacial.
        layers.MaxPooling2D((2, 2)),

        # Capa Flatten:
        # - Convierte la salida 3D de las capas convolucionales a un vector 1D,
        #   lo que es necesario para las capas densas.
        layers.Flatten(),

        # Capa Dropout:
        # - Si `use_dropout` es True, se aplica Dropout con una tasa del 30% para prevenir sobreajuste.
        # - Si `use_dropout` es False, se añade una capa vacía para no afectar el modelo.
        layers.Dropout(0.3) if use_dropout else layers.Layer(),

        # Capa completamente conectada (densa):
        # - Contiene 256 unidades con activación ReLU.
        # - Se aplica regularización L2 para reducir el sobreajuste penalizando pesos grandes.
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),

        # Capa de salida:
        # - Número de unidades igual al número de clases (`num_classes`).
        # - Activación softmax para convertir los valores en probabilidades.
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilamos el modelo:
    # - Optimizador Adam: Adaptativo y eficiente para tareas de clasificación.
    # - Función de pérdida: sparse_categorical_crossentropy (para etiquetas enteras).
    # - Métrica: accuracy (para medir el rendimiento durante el entrenamiento).
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
