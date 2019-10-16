#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Все что связано непосредственно с архитектурой сети
    - конфигурационная информация о сети в NetConfig
    - дополнительные слои и функции строящие и применяющие группы слоев
    - NetManager строящий/загружающий модель
"""
import copy
import logging
import os
import pickle
from enum import Enum

import keras.backend as K
import keras.initializers
import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Activation, \
    BatchNormalization, ZeroPadding2D, UpSampling2D, SeparableConv2D
from keras.models import Model, load_model

from semantic_segmentation import losses, keras_metrics


# относительная воспроизводимость (веса инициализируются теми же значениями)
# tf.set_random_seed(42)


class IdentityInitializer(keras.initializers.Initializer):
    """
    Инициализация как в статье https://arxiv.org/pdf/1511.07122.pdf
    """

    def __call__(self, shape, dtype=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0] // 2, shape[1] // 2
        for i in range(shape[2]):
            array[cx, cy, i, i] = 1
        return K.constant(array, shape=shape, dtype=dtype)


def ImageScaler(max_scale_power=1):
    """
    Возвращает keras.layer преобразующий батч входных изображений в
    [images, images // 2, images // 4, ..., images // 2**max_scale_power] список изображений в разных масштабах
    ("//" выше обозначает перемасштабирование а не деление)
    :param max_scale_power: сколько различных масштабов использовать
    :return:
    """
    # Bilinear interpolation
    return Lambda(lambda images:
                  [images] + [
                      tf.image.resize_images(images, size=(tf.shape(images)[1:3] // 2 ** scale))
                      for scale in range(1, max_scale_power + 1)
                  ],
                  name="ResizeImagesLayer"
                  )


class PreprocessingType(Enum):
    NONE = 0
    MOBILENET_LIKE = 1


supported_preprocessing_types = {
    "none": PreprocessingType.NONE,
    "mobilenet_like": PreprocessingType.MOBILENET_LIKE
}


class NetConfig:
    """
    Конфигурация сети и всей системы в целом
    """

    @staticmethod
    def from_others(base_config, side_multiple=None, max_image_side=None, min_pixels_for_detection=None):
        """
        Создает новый конфиг из base_config с заменой значений некоторых (архитектуро-независимых) параметров
        на те что в оставшихся параметрах
        :param base_config:
        :param side_multiple: новое значение или None
        :param max_image_side: новое значение или None
        :param min_pixels_for_detection: новое значение или None
        :return: new_config
        """
        new_config = copy.deepcopy(base_config)
        if side_multiple:
            new_config._side_multiple = side_multiple
        if max_image_side:
            new_config._max_side = max_image_side
        if min_pixels_for_detection:
            new_config._min_pixels_for_detection = min_pixels_for_detection
        return new_config

    def __init__(self,
                 object_types_fname=None,
                 scale=4, fml_compatible=True, no_classification=False,
                 side_multiple=64, max_image_side=512, min_pixels_for_detection=5,
                 preprocessing=PreprocessingType.NONE, grey=True):
        """

        :param scale: во сколько раз предсказываемая карта сегментаций меньше исходной кортинки по одному измерению
        :param fml_compatible: конвертируется ли в FML (правильные padding при stride > 1)
        :param side_multiple: размер картинок (должен быть) кратен этому параметру
        :param max_image_side: размер максимальной стороны изображений
        :param min_pixels_for_detection: минимальное количество пикселей в карте сегментации которые считается детекцией
        (т.е. если размер компоненты связности в карте сегментации не меньше этого значения - постпроцессим это как
        задетекченный блок - иначе СЧИТАЕМ ЧТО ТУТ НИЧЕГО НЕТ, мол это случайный шум)
        :param preprocessing: какой препроцессинг используется
        """

        if object_types_fname is None:
            self._class_names = None
            self._is_classification_supported = False
        else:
            self._is_classification_supported = not no_classification
            self._read_classnames_from_file(object_types_fname)

        self._grey = grey
        # во сколько раз предсказываемая карта сегментаций меньше исходной кортинки по одному измерению
        self._scale = scale
        # обучается/обучалась так, чтобы веса были конвертируемы в FML
        self._fml_compatible = fml_compatible
        self._preprocessing = preprocessing

        self._side_multiple = side_multiple
        # максимальный размер стороны изображения (длина или ширина)
        self._max_side = max_image_side
        self._min_pixels_for_detection = min_pixels_for_detection

    def log_classification_mode(self):
        if self.is_classification_supported():
            logging.info(f"Training classification with object types: {self._class_names}")
        elif self._class_names is not None:
            logging.info(f"Training WITHOUT classification, detection only for types: {self._class_names}")
        else:
            logging.info(f"Training WITHOUT classification, detection for any barcode in datasets")

    def is_grey(self):
        return self._grey

    def get_scale(self):
        return self._scale

    def get_min_pixels_for_detection(self):
        return self._min_pixels_for_detection

    def get_side_multiple(self):
        return self._side_multiple

    def get_max_side(self):
        return self._max_side

    def is_fml_compatible(self):
        return self._fml_compatible

    def get_preprocessing_type(self):
        return self._preprocessing

    def get_preprocessing_fn(self):
        if self._preprocessing == PreprocessingType.NONE:
            return lambda x: x
        elif self._preprocessing == PreprocessingType.MOBILENET_LIKE:
            return preprocess_image_mobilenet
        else:
            raise ValueError("Unknown preprocessing type")

    def get_depreprocessing_fn(self):
        if self._preprocessing == PreprocessingType.NONE:
            return lambda x: x
        elif self._preprocessing == PreprocessingType.MOBILENET_LIKE:
            return depreprocess_image_mobilenet
        else:
            raise ValueError("Unknown preprocessing type")

    def get_class_names(self):
        return self._class_names

    def get_n_classes(self):
        return len(self._class_names)

    def get_class_name(self, class_id):
        return self._class_names[class_id]

    def get_class_id(self, class_name):
        return self._class_name_to_id[class_name]

    def is_class_supported(self, class_name):
        return self._class_names is None or class_name in self._class_name_to_id

    def is_classification_supported(self):
        return self._is_classification_supported

    def _read_classnames_from_file(self, path):
        assert os.path.exists(path), f"File with object class names {path} does not exist"
        logging.info(f"Reading object types from {path}")
        class_names = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    class_names.append(line.strip())
        self._class_names = class_names
        self._class_name_to_id = dict((class_name, i) for i, class_name in enumerate(class_names))

    def __str__(self):
        sb = ["Net Config:"]
        for key in self.__dict__:
            if key.startswith('_'):
                sb.append("\t{key}={value}".format(key=key[1:], value=self.__dict__[key]))

        return '\n'.join(sb)


def preprocess_image_mobilenet(image):
    return (image - 127.5) / 127.5


def depreprocess_image_mobilenet(image):
    return image * 127.5 + 127.5


def conv_bn(inputs, n_filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=1,
            padding='same', activation='relu', use_bn=False, kernel_initializer='glorot_uniform',
            use_strides_compatible_with_fml=False, separable=False):
    x = inputs
    if use_strides_compatible_with_fml and max(strides) > 1:
        assert strides == (2, 2) and kernel_size == (3, 3) and padding == 'same', "padding can not be set properly"
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        padding = 'valid'

    Conv2DLayer = Conv2D
    if separable:
        Conv2DLayer = SeparableConv2D

    x = Conv2DLayer(
        n_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=None if use_bn else activation,
        kernel_initializer=kernel_initializer
    )(x)

    if use_bn:
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

    return x


class NetManager:
    """
    Строит/загружает модель
    """

    CURRENT_MODEL_FILENAME = "model.h5"
    INFERENCE_MODEL_FILENAME = "inference_model.h5"
    MODEL_WEIGHTS_FILENAME = "model_weights.h5"
    PICKLED_CONFIG_FILENAME = "config.pkl"

    def __init__(self, log_dir, net_config=None):
        self._log_dir = log_dir
        if net_config is not None:
            self._net_config = net_config
        else:
            self.load_config()
        self._model = None

    def build_model(self):
        self._build_dilated_conv_model()
        # self._build_multiscale_model(max_scale_power=3)
        return self._net_config

    def _build_dilated_conv_model(self):
        """
        Стороит нейросеть и добавляет ее параметры в конфиг
        текущая основная модель на dilated свертках
        :return: net_config
        """
        # основано на https://arxiv.org/pdf/1511.07122.pdf
        # все еще в процессе подбора оптимальной конфигурации
        image_channels = 1 if self._net_config.is_grey() else 3
        inputs = Input(shape=(None, None, image_channels))
        x = inputs
        n_filters = 24

        # уменьшаем разрешение
        x = conv_bn(x, n_filters, strides=(2, 2), separable=True,
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x = conv_bn(x, n_filters, dilation_rate=1, separable=True)
        x = conv_bn(x, n_filters, strides=(2, 2), separable=True,
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        # оригинальная статья
        x1 = conv_bn(x, n_filters, dilation_rate=1, separable=False)
        x2 = conv_bn(x1, n_filters, dilation_rate=2, separable=False)
        x3 = conv_bn(x2, n_filters, dilation_rate=4, separable=False)
        x4 = conv_bn(x3, n_filters, dilation_rate=8, separable=False)
        x5 = conv_bn(x4, n_filters, dilation_rate=16, separable=False)
        # возможно этот слой не нужен (в оригинальной статье есть)
        x6 = conv_bn(x5, n_filters, dilation_rate=1, separable=False)

        x = x6
        # last dense layer
        n_classes = 0
        if self._net_config.is_classification_supported():
            n_classes = self._net_config.get_n_classes()
        x = Conv2D(1 + n_classes, (1, 1), padding='same', activation=None)(x)

        self._model = Model(inputs=inputs, outputs=x, name='dilated_conv')
        self._net_config._scale = 4

    def _build_traditional_cnn(self):
        """
        Стороит нейросеть и добавляет ее параметры в конфиг
        основа как в vgg, затем объединяем признаки с разных пространственных разрешений
        это объединение основано на https://arxiv.org/abs/1611.06612 и http://bmvc2018.org/contents/papers/0494.pdf
        :return:
        """
        inputs = Input(shape=(None, None, 3))
        x = inputs
        n_filters = 32
        x = conv_bn(x, n_filters, strides=(2, 2),
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x = conv_bn(x, n_filters)

        x = conv_bn(x, n_filters * 2, strides=(2, 2),
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x = conv_bn(x, n_filters * 2)
        x1 = x  # scale=4

        x = conv_bn(x, n_filters * 4, strides=(2, 2),
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x = conv_bn(x, n_filters * 4)
        x2 = x  # scale=8

        x = conv_bn(x, n_filters * 8, strides=(2, 2),
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x = conv_bn(x, n_filters * 8)
        x3 = x  # scale=16

        x = conv_bn(x, n_filters * 16, strides=(2, 2),
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x = conv_bn(x, n_filters * 16)
        x4 = x  # scale=32

        x = conv_bn(x, n_filters * 16, strides=(2, 2),
                    use_strides_compatible_with_fml=self._net_config.is_fml_compatible())
        x5 = x  # scale=64

        x = self._fuse_multiscale_features(
            [x1, x2, x3, x4, x5],
            n_output_channels=n_filters * 16,
            activation='relu'
        )
        n_classes = 0
        if self._net_config.is_classification_supported():
            n_classes = self._net_config.get_n_classes()
        x = Conv2D(1 + n_classes, (1, 1), padding='same', activation=None)(x)

        self._model = Model(inputs=inputs, outputs=x, name='multiscale_cnn')
        self._net_config._scale = 4
        self._net_config._side_multiple = max(64, self._net_config._side_multiple)

    def _build_multiscale_model(self, max_scale_power=3):
        """
        Стороит нейросеть и добавляет ее параметры в конфиг
        эта модель просто подает одной и той же нейросети изображение в разных масштабах
        и усредняет предсказанный результат
        :param max_scale_power:
        :return:
        """
        self._build_dilated_conv_model()
        self._base_cnn = self._model
        input_images = Input(shape=(None, None, 3))
        images_decreasing_size_list = ImageScaler(max_scale_power=max_scale_power)(input_images)
        outputs = []
        for i, images_batch in enumerate(images_decreasing_size_list):
            output = self._base_cnn(images_batch)
            if i > 0:
                # TODO try bilinear interpolation
                output = UpSampling2D(size=(2 ** i, 2 ** i))(output)
            outputs.append(output)

        output = Lambda(lambda l: K.mean(K.concatenate(l, axis=-1), axis=-1, keepdims=True))(outputs)
        self._model = Model(inputs=input_images, outputs=output, name='multiscale_dilated')

    @staticmethod
    def _fuse_multiscale_features(features_list, n_output_channels, activation=None):
        """
        Собирает контекстную информацию из выходов разных слоев нейросети (с разным пространственным разрешением)
        Выдает новую карту признаков с пространственным разрешением как у наибольшей из входных карт признаков
            и количеством каналов n_output_channels
        :param features_list: список карт признаков, полученных их слоев с разным пространственным разрешением,
            отсортированы по убыванию пространственного размера со scale=2
            (каждая следующая карта признаков в 2 раза меньше предыдущей в пространственном разрешении)
        :param n_output_channels: количество выходов у карты признаков на выходе
        :param activation: активация, которую использовать
        :return:
        """
        prepared_features = []
        for i, x in enumerate(features_list):
            x = Conv2D(n_output_channels, (1, 1), padding='same', activation=None)(x)
            if i > 0:
                x = UpSampling2D((2 ** i, 2 ** i))(x)
            prepared_features.append(x)
        x = keras.layers.add(prepared_features)
        if activation:
            keras.layers.Activation(activation)(x)
        return x

    def get_keras_model(self):
        return self._model

    def save_model(self, step):
        self._model.save(os.path.join(self._log_dir, "model{:03d}.h5".format(step)))
        self._model.save(os.path.join(self._log_dir, NetManager.CURRENT_MODEL_FILENAME))

    def save_inference(self):
        weights_path = os.path.join(self._log_dir, NetManager.MODEL_WEIGHTS_FILENAME)
        self._model.save_weights(weights_path)
        self.build_model()
        self._model.load_weights(weights_path)
        self._model.save(os.path.join(self._log_dir, NetManager.INFERENCE_MODEL_FILENAME))

    def load_another_model(self, another_log_dir):
        """
        Загрузить модель из другой директории с тем же конфигом (теми же архитектуро-зависимыми параметрами)
        архитектуро-независимые параметры остаются теми же что и при инициализации текущего объекта NetManager
        :param another_log_dir: директория из которой подгружается модель
        :return:
        """
        another_net_manager = NetManager(another_log_dir, self._net_config)
        another_net_manager.load_config()
        another_net_manager.load_model()
        self._model = another_net_manager._model
        self._net_config = NetConfig.from_others(another_net_manager._net_config, self._net_config)
        return self._net_config

    def load_model(self, path_to_model=None):
        """
        Загрузить модель (из self._log_dir директории)
        :param path_to_model: путь до .h5 модели (абсолютный/относительный или от self._log_dir)
        :return:
        """
        # если вы действительно уверены что загружаете модель так чтобы конфиг был такой же
        # можете закомментить строчку внизу
        assert path_to_model is None, f"Programmer! Models are stored in log_dir, near their config. " \
                                      f"If you load model from other location model and config " \
                                      f"will not match most likely."
        if path_to_model is not None:
            if not os.path.exists(path_to_model):
                path_to_model = os.path.join(self._log_dir, path_to_model)
            return self._load_model(path_to_model)

        # загрузить из директории с логами, если ничего не подали
        possible_model_fnames = [NetManager.INFERENCE_MODEL_FILENAME, NetManager.CURRENT_MODEL_FILENAME]
        for model_fname in possible_model_fnames:
            path_to_model = os.path.join(self._log_dir, model_fname)
            if os.path.exists(path_to_model):
                return self._load_model(path_to_model)
        raise FileNotFoundError(f"Model not found in dir {self._log_dir}. Must contain "
                                f"at least one of the following files {possible_model_fnames}")

    def save_config(self):
        pickle.dump(self._net_config, open(os.path.join(self._log_dir, NetManager.PICKLED_CONFIG_FILENAME), 'wb'))

    def load_config(self):
        self._net_config = pickle.load(open(os.path.join(self._log_dir, NetManager.PICKLED_CONFIG_FILENAME), 'rb'))

    def _load_model(self, path_to_model):
        assert os.path.exists(path_to_model), f"model fname does not exist, {path_to_model}"
        logging.info(f"loading model from {path_to_model}")
        self._model = load_model(path_to_model,
                                 custom_objects={
                                     'tf': tf,
                                     'get_loss': losses.get_loss,
                                     'IdentityInitializer': IdentityInitializer,
                                     'detection_and_classification_loss': losses.detection_and_classification_loss,
                                     'detection_loss': losses.detection_loss,
                                     'pixel_negative_loss': losses.pixel_negative_loss,
                                     'pixel_positive_loss': losses.pixel_positive_loss,
                                     'pixel_hard_negative_loss': losses.pixel_hard_negative_loss,
                                     'classification_loss': losses.classification_loss,
                                     'detection_pixel_acc': keras_metrics.detection_pixel_acc,
                                     'detection_pixel_precision': keras_metrics.detection_pixel_precision,
                                     'detection_pixel_recall': keras_metrics.detection_pixel_recall,
                                     'detection_pixel_f1': keras_metrics.detection_pixel_f1,
                                     'classification_pixel_acc': keras_metrics.classification_pixel_acc
                                 })
        return self._net_config
