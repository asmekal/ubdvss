#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Различные функции потерь для обучения
"""
import functools

import keras.backend as K
import tensorflow as tf

# TODO: подобрать константы
L_POSITIVE_WEIGHT = 15.
L_NEGATIVE_WEIGHT = 1.
L_HARD_NEGATIVE_WEIGHT = 5.
L_DETECTION_WEIGHT = 1.
L_CLASSIFICATION_WEIGHT = 1.


def get_loss(classification_mode=False):
    if classification_mode:
        return detection_and_classification_loss
    else:
        return detection_loss


def _prepare_detection_logits(y_true, y_pred):
    detection_true = K.cast(y_true > 0, tf.float32)
    detection_pred = K.sigmoid(y_pred[..., :1])
    return detection_true, detection_pred


def detection_loss(y_true, y_pred):
    """
    Вычисляет loss чисто за детекцию -
    смотрим первый канал y_pred считая его логитом вероятности того что здесь есть объект
    :param y_true: y_true.shape == (bs, h, w, 1) содержит целые числа от 0 до n_classes включительно
            y_true[..., 0] = 0 когда на этом месте нет объекта вообще
            y_true[..., 0] = i > 0 значит класс объекта на этом месте i-1
    :param y_pred: y_pred.shape == (bs, h, w, 1 + n_classes)
    :return:
    """
    detection_true, detection_pred = _prepare_detection_logits(y_true, y_pred)
    return binary_classification_loss(detection_true, detection_pred)


def detection_and_classification_loss(y_true, y_pred):
    """
    Суммарный лосс за детекцию + классификацию
    :param y_true: y_true.shape == (bs, h, w, 1) содержит целые числа от 0 до n_classes включительно
            y_true[..., 0] = 0 когда на этом месте нет объекта вообще
            y_true[..., 0] = i > 0 значит класс объекта на этом месте i-1
    :param y_pred: y_pred.shape == (bs, h, w, 1 + n_classes)
    :return:
    """
    loss_detection = detection_loss(y_true, y_pred)

    loss_classification = classification_loss(y_true, y_pred)

    loss = L_DETECTION_WEIGHT * loss_detection + L_CLASSIFICATION_WEIGHT * loss_classification

    return loss


def classification_loss(y_true, y_pred):
    """
    Вычисление кроссэнтропии для каждого пикселя который есть хоть в каком-то объекте
    (для пикселей фона лосс зануляется)
    :param y_true: y_true.shape == (bs, h, w, 1) содержит целые числа от 0 до n_classes включительно
            y_true[..., 0] = 0 когда на этом месте нет объекта вообще
            y_true[..., 0] = i > 0 значит класс объекта на этом месте i-1
    :param y_pred: y_pred.shape == (bs, h, w, 1 + n_classes)
    :return:
    """
    positive_mask = K.cast(y_true > 0, tf.float32)

    class_labels = (y_true - 1) * positive_mask
    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.squeeze(tf.cast(class_labels, tf.int64), axis=-1),
        logits=y_pred[..., 1:])
    positive_mask = tf.squeeze(positive_mask, axis=-1)
    class_loss = K.sum(class_loss * positive_mask) / K.maximum(K.sum(positive_mask), 1)
    return class_loss


def binary_classification_loss(y_true, y_pred):
    """
    Вычисляет лосс при сегментации на 2 класса 0 - фон, 1 - целевой класс
    этот лосс считает все 3 сразу (без перевычислений которые были бы если вызывать функции по отдельности)
    loss = average_loss_on_positive_pixels
        + average_loss_on_negative_pixels
        + worst_k_average_loss_on_negative_pixels, where k=num_positive_pixels
    :param y_true: y_true.shape == (bs, h, w, 1)
    :param y_pred: y_pred.shape == (bs, h, v, 1) after sigmoid
    :return:
    """
    crossentropy_loss = K.binary_crossentropy(y_true, y_pred)  # .shape = (batch_size, h, w)
    positive_mask = y_true
    n_positive_pixels = K.maximum(K.sum(positive_mask), 1)
    # средний лосс на положительных пикселях
    positive_loss = K.sum(crossentropy_loss * positive_mask) / n_positive_pixels

    negative_mask = 1 - positive_mask
    crossentropy_loss_masked_negatives = crossentropy_loss * negative_mask
    n_negative_pixels = K.maximum(K.sum(negative_mask), 1)
    # средний лосс на отрицательных пикселях
    negative_loss = K.sum(crossentropy_loss_masked_negatives) / n_negative_pixels

    # тут хотя бы один должен быть
    n_hard_max_pixels = K.minimum(n_positive_pixels, n_negative_pixels)
    max_negatives, _ = tf.nn.top_k(input=K.reshape(crossentropy_loss_masked_negatives, shape=(-1,)),
                                   k=K.cast(n_hard_max_pixels, dtype=tf.int32),
                                   sorted=False)
    # средний лосс на худших (наиболее путающихся с положительными) отрицательных пикселях
    # число худших примеров равно min(число положительных пикселей, число отрицательных пикселей)
    hard_negative_loss = K.mean(max_negatives)
    hard_negative_loss = tf.cond(
        tf.is_nan(hard_negative_loss),
        lambda: tf.Print(0., data=[], message="hard_negative_loss is nan (replaced by zero)"),
        lambda: hard_negative_loss
    )

    loss = L_POSITIVE_WEIGHT * positive_loss \
           + L_NEGATIVE_WEIGHT * negative_loss \
           + L_HARD_NEGATIVE_WEIGHT * hard_negative_loss
    return loss


def _prepare_detection_args(loss_fn):
    @functools.wraps(loss_fn)
    def loss_fn_wrapped(y_true, y_pred):
        detection_true, detection_pred = _prepare_detection_logits(y_true, y_pred)
        return loss_fn(detection_true, detection_pred)

    return loss_fn_wrapped


@_prepare_detection_args
def pixel_positive_loss(y_true, y_pred):
    """
    Вычисляет средний лосс по положительным (с точки зрения y_true) пикселям
    """
    crossentropy_loss = K.binary_crossentropy(y_true, y_pred)  # .shape = (batch_size, h, w)
    positive_mask = y_true
    n_positive_pixels = K.sum(positive_mask)
    # средний лосс на положительных пикселях
    positive_loss = K.sum(crossentropy_loss * positive_mask) / K.maximum(1., n_positive_pixels)

    return positive_loss


@_prepare_detection_args
def pixel_negative_loss(y_true, y_pred):
    """
    Вычисляет средний лосс по отрицательным (с точки зрения y_true) пикселям
    """
    crossentropy_loss = K.binary_crossentropy(y_true, y_pred)  # .shape = (batch_size, h, w)
    positive_mask = y_true

    negative_mask = 1 - positive_mask
    crossentropy_loss_masked_negatives = crossentropy_loss * negative_mask
    n_negative_pixels = K.sum(negative_mask)
    # средний лосс на отрицательных пикселях
    negative_loss = K.sum(crossentropy_loss_masked_negatives) / K.maximum(1., n_negative_pixels)

    return negative_loss


@_prepare_detection_args
def pixel_hard_negative_loss(y_true, y_pred):
    """
    Вычисляет средний лосс по худшим отрицательным(с точки зрения y_true) пикселям
    количество плохих пикселей берется max(1, min(n_positive_pixels, n_negative_pixels))
    """
    crossentropy_loss = K.binary_crossentropy(y_true, y_pred)  # .shape = (batch_size, h, w)
    positive_mask = y_true
    n_positive_pixels = K.sum(positive_mask)

    negative_mask = 1 - positive_mask
    crossentropy_loss_masked_negatives = crossentropy_loss * negative_mask
    n_negative_pixels = K.sum(negative_mask)

    n_hard_max_pixels = K.maximum(K.minimum(n_positive_pixels, n_negative_pixels), 1.)
    max_negatives, _ = tf.nn.top_k(input=K.reshape(crossentropy_loss_masked_negatives, shape=(-1,)),
                                   k=K.cast(n_hard_max_pixels, dtype=tf.int32),
                                   sorted=False)
    # средний лосс на худших (наиболее путающихся с положительными) отрицательных пикселях
    # число худших примеров равно min(число положительных пикселей, число отрицательных пикселей)
    hard_negative_loss = K.mean(max_negatives)

    return hard_negative_loss


def get_losses(classification_mode=False):
    """
    Возвращает список всех функций потерь
    :param classification_mode:
    :return:
    """
    losses = [
        detection_loss,
        pixel_positive_loss,
        pixel_negative_loss,
        pixel_hard_negative_loss
    ]
    if classification_mode:
        losses.append(classification_loss)
    return losses
