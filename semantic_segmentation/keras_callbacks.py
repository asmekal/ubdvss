#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Все что связано с callbacks для обучения и валидации
"""
import os

import keras.backend as K
import tensorflow as tf
from keras import callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from semantic_segmentation.model_runner import ModelRunner


class SingleSplitLogCallback(callbacks.TensorBoard):
    """
    Callback для отдельного сохранения метрик в трейне/валидации
    Используется для отрисовки графиков обучения/валидации на одной оси
    """
    ONLY_TRAIN_LOGS_MODE = 'train'
    ONLY_VALID_LOGS_MODE = 'valid'

    def __init__(self, log_dir, mode=ONLY_TRAIN_LOGS_MODE):
        super().__init__(log_dir=log_dir)
        assert mode in (SingleSplitLogCallback.ONLY_VALID_LOGS_MODE, SingleSplitLogCallback.ONLY_TRAIN_LOGS_MODE)
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.is_train_log_mode():
            filter_fn = lambda k: not k.startswith('val')
            map_fn = lambda k, v: (k, v)
        else:
            filter_fn = lambda k: k.startswith('val')
            map_fn = lambda k, v: ('_'.join(k.split('_')[1:]), v)
        logs = dict(map_fn(k, v) for (k, v) in logs.items() if filter_fn(k))
        super().on_epoch_end(epoch=epoch, logs=logs)

    def is_train_log_mode(self):
        return self.mode == SingleSplitLogCallback.ONLY_TRAIN_LOGS_MODE

    @classmethod
    def get_callbacks(cls, train_log_dir, valid_log_dir):
        return [
            cls(train_log_dir, cls.ONLY_TRAIN_LOGS_MODE),
            cls(valid_log_dir, cls.ONLY_VALID_LOGS_MODE)
        ]


class EvaluationCallback(SingleSplitLogCallback):
    """
    Служит для подсчета целевых метрик и визуализации картинок в тензорборде при обучении
    """

    def __init__(self, log_dir, net_config, batch_generator, max_evaluated_images=-1,
                 mode=SingleSplitLogCallback.ONLY_TRAIN_LOGS_MODE):
        """
        :param log_dir: путь до папки с логами (тензорборда)
        :param net_config:
        :param batch_generator: объект типа BatchGenerator с оцениваемой выборкой
        :param max_evaluated_images: сколько максимум картинок использовать для оценки,
            если -1 все что есть в выборке, иначе min(картинок в выборке, max_evaluated_images)
        :param mode: 'train' / 'valid'
        """
        super().__init__(log_dir=log_dir, mode=mode)
        self.__net_config = net_config
        self.__batch_generator = batch_generator.generate(add_metainfo=True)
        self.__n_evaluated_images = batch_generator.get_images_per_epoch()
        if max_evaluated_images >= 0:
            self.__n_evaluated_images = min(self.__n_evaluated_images, max_evaluated_images)
        self.__epochs_count = 0
        self.__model_runner = ModelRunner(net_config=net_config, pixel_threshold=0.5)

    def on_epoch_end(self, epoch, logs=None):
        scalar_logs, visualizations = self.__model_runner.run(model=self.model,
                                                              data_generator=self.__batch_generator,
                                                              n_images=self.__n_evaluated_images,
                                                              save_dir=None,
                                                              save_visualizations=False)
        if not self.is_train_log_mode():
            # это такой чит чтобы в родителе отфильтровалось то что надо, а то что не надо наоборот осталось
            scalar_logs = dict(('val_' + k, v) for k, v in scalar_logs.items())

        for k, v in logs.items():
            scalar_logs[k] = v

        image_logs = dict()
        for key, value in visualizations.items():
            image_logs[f"{self.mode}_{key}"] = value
        if epoch < 1:
            for name, value in image_logs.items():
                images_placeholder = K.placeholder(shape=(None, None, None, 3), dtype=None, name=name)
                tf.summary.image(name, images_placeholder, max_outputs=10)

        summary_str = tf.summary.merge_all(
            key=tf.GraphKeys.SUMMARIES,
            scope=f"{self.mode}.*"
        )
        feed_dict = dict(("{}:0".format(key), value) for key, value in image_logs.items())
        self.writer.add_summary(self.sess.run([summary_str], feed_dict=feed_dict)[0], epoch)
        super().on_epoch_end(epoch, scalar_logs)

    @classmethod
    def get_callbacks(cls, net_config,
                      train_log_dir, valid_log_dir,
                      train_generator, valid_generator,
                      max_evaluated_images):
        return [
            cls(train_log_dir, net_config, train_generator, max_evaluated_images, mode=cls.ONLY_TRAIN_LOGS_MODE),
            cls(valid_log_dir, net_config, valid_generator, max_evaluated_images, mode=cls.ONLY_VALID_LOGS_MODE)
        ]


def build_callbacks_list(log_dir, net_config, training_generator, validation_generator, max_evaluated_images=-1):
    """
    Собирает список всех callbacks для обучение
    :param log_dir: основная директория с обучением
    :param net_config:
    :param training_generator: BatchGenerator по обучающим данным
    :param validation_generator: BatchGenerator по валидационным данным
    :param max_evaluated_images: максимальное количество изображений, использующееся для оценки целевых метрик
    :return:
    """
    backup_dir = os.path.join(log_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(backup_dir, "model_{epoch:03d}.h5"))

    last_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"))
    best_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "model_best.h5"), save_best_only=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, min_lr=1e-4)

    basic_callbacks = [
        last_checkpoint_callback,
        best_checkpoint_callback,
        backup_checkpoint_callback,
        reduce_lr_callback
    ]

    train_log_dir = os.path.join(log_dir, 'train')
    valid_log_dir = os.path.join(log_dir, 'valid')
    tensorboard_callbacks = EvaluationCallback.get_callbacks(
        net_config, train_log_dir, valid_log_dir, training_generator, validation_generator, max_evaluated_images)

    return basic_callbacks + tensorboard_callbacks
