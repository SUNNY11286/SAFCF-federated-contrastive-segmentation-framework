import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def sum_scaled_weights(scaled_weight_list, length_ratio):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean * (1 - length_ratio))
    return avg_grad

def contrastive_loss(z_prev, z_present, z_serv, margin=0.2, lr=1e-3):
    z_prev = tf.math.l2_normalize(z_prev)
    z_present = tf.math.l2_normalize(z_present)
    dis = tf.sqrt(tf.reduce_sum(tf.square(z_present - z_prev)))
    dis = tf.sqrt(tf.math.maximum(dis, tf.keras.backend.epsilon()))
    sqpres = tf.math.square(z_present)
    sqMar = tf.math.square(tf.math.maximum(margin - z_present, 0))
    loss = tf.math.reduce_mean(z_prev * sqpres + (1 - z_prev) * sqMar)
    loss = loss - (loss * lr)
    srv = tf.sqrt(tf.reduce_sum(tf.square(z_serv)))
    srv = tf.math.log(srv)
    return loss / srv