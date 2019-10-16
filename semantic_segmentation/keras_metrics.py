#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Различные метрики для keras подсчитываемые при обучении
"""

import functools

import keras.backend as K
import tensorflow as tf

from semantic_segmentation.losses import get_losses


def _squeeze_single_dims(*args):
    return [tf.squeeze(arg) for arg in args]


def _metric_wrapper(metric_fn):
    @functools.wraps(metric_fn)
    def metric_fn_wrapped(true, pred, weights=None):
        if weights is None:
            weights = tf.ones_like(true, tf.float32)
        _true, _pred, _weights = _squeeze_single_dims(true, pred, weights)

        metric_value = metric_fn(_true, _pred, _weights)

        return metric_value

    return metric_fn_wrapped


@_metric_wrapper
def _acc(true, pred, weights):
    equal = K.cast(K.equal(true, pred), tf.float32)
    return K.sum(equal * weights) / K.maximum(1., K.sum(weights))


def confusion_matrix(true, pred, weights):
    """
    Confusion matrix для бинарной классификации
    :param true:
    :param pred:
    :param weights:
    :return: tp, tn, fp, fn - confusion matrix
    """
    equal = K.equal(true, pred)

    def calculate_sum(metric):
        m = K.cast(metric, tf.float32)
        return K.sum(m * weights)

    tp = tf.logical_and(equal, K.equal(true, 1))
    tn = tf.logical_and(equal, K.equal(true, 0))
    fp = tf.logical_and(tf.logical_not(equal), K.equal(pred, 1))
    fn = tf.logical_and(tf.logical_not(equal), K.equal(pred, 0))

    tp = calculate_sum(tp)
    tn = calculate_sum(tn)
    fp = calculate_sum(fp)
    fn = calculate_sum(fn)

    return tp, tn, fp, fn


@_metric_wrapper
def precision(true, pred, weights):
    """
    Вычисляет precision c учетом весов
    :param true:
    :param pred:
    :param weights:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix(true, pred, weights)
    return tp / K.maximum(1., tp + fp)


@_metric_wrapper
def recall(true, pred, weights):
    """
    Вычисляет recall с учетом весов
    :param true:
    :param pred:
    :param weights:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix(true, pred, weights)
    return tp / K.maximum(1., tp + fn)


@_metric_wrapper
def f1(true, pred, weights):
    """
    Вычисляет f1-меру с учетом весов
    :param true:
    :param pred:
    :param weights:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix(true, pred, weights)
    precision = tp / K.maximum(1., tp + fp)
    recall = tp / K.maximum(1., tp + fn)
    return tf.cond(K.not_equal(precision + recall, 0.),
                   lambda: 2. * precision * recall / (precision + recall),
                   lambda: 0.)


def _get_detection_labels(y_true, y_pred):
    detection_true = K.cast(K.greater(y_true, 0), tf.int32)
    detection_pred = K.cast(K.greater(y_pred[..., 0], 0), tf.int32)
    return detection_true, detection_pred


def detection_pixel_acc(y_true, y_pred):
    """
    Вычисляет попиксельную accuracy детекции
    :param y_true:
    :param y_pred:
    :return:
    """
    detection_true, detection_pred = _get_detection_labels(y_true, y_pred)
    return _acc(detection_true, detection_pred)


def detection_pixel_precision(y_true, y_pred):
    """
    Вычисляет попиксельню точность (precision) детекции
    :param y_true:
    :param y_pred:
    :return:
    """
    detection_true, detection_pred = _get_detection_labels(y_true, y_pred)
    return precision(detection_true, detection_pred)


def detection_pixel_recall(y_true, y_pred):
    """
    Вычисляет попиксельню полноту (recall) детекции
    :param y_true:
    :param y_pred:
    :return:
    """
    detection_true, detection_pred = _get_detection_labels(y_true, y_pred)
    return recall(detection_true, detection_pred)


def detection_pixel_f1(y_true, y_pred):
    """
    Вычисляет попиксельню f1-меру детекции
    :param y_true:
    :param y_pred:
    :return:
    """
    detection_true, detection_pred = _get_detection_labels(y_true, y_pred)
    return f1(detection_true, detection_pred)


def classification_pixel_acc(y_true, y_pred):
    """
    Вычисляет попиксельную accuracy классификации
    считается только по y_true > 0 т.е. там где есть какой-то объект
    :param y_true:
    :param y_pred:
    :return:
    """
    mask = K.cast(y_true > 0, tf.float32)
    labels = K.cast((y_true - 1) * mask, tf.int64)
    class_p = tf.nn.softmax(y_pred[..., 1:], axis=-1)
    predictions = tf.argmax(class_p, axis=-1)
    return _acc(labels, predictions, weights=mask)


def get_all_metrics(classification_mode=False):
    """
    Возвращает список всех метрик
    :param classification_mode:
    :return:
    """
    all_metrics = [
        detection_pixel_acc,
        detection_pixel_precision,
        detection_pixel_recall,
        detection_pixel_f1
    ]
    if classification_mode:
        all_metrics.append(classification_pixel_acc)

    all_metrics += get_losses(classification_mode)
    return all_metrics
