import keras as K
from src.evaluation_metrics import f1_m, precision_m, recall_m

# Define a function to build the model architecture
def build_model():
    base = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze all layers except the last 10
    for layer in base.layers[:-10]:
        layer.trainable = False

    model = K.layers.GlobalAveragePooling2D()(base.output)
    model = K.layers.Dense(512, activation=None)(model)
    model = K.layers.BatchNormalization()(model)
    model = K.layers.Activation('relu')(model)
    model = K.layers.Dropout(rate=0.4)(model)
    model = K.layers.Dense(3, activation='softmax')(model)
    model = K.models.Model(inputs=base.input, outputs=model)

    optimizer = K.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', precision_m, recall_m, f1_m]
    )

    return model
