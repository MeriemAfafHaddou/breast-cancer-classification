"""
Model definition for breast ultrasound classification using
a ResNet50 backbone with a custom classification head
and medical-focused evaluation metrics.
"""

import keras as K
import tensorflow as tf
import numpy as np
import cv2

from .evaluation_metrics import f1_m, precision_m, recall_m



def build_model():
    """
    Build the model architecture
    """
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


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create model that maps input -> last conv layer + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply feature maps by importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize between 0 and 1
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap /= max_val

    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if img.max() <= 1:
        img = np.uint8(255 * img)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img
