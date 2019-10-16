#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Генератор батчей для обучения сети
"""

import functools
import itertools
import logging
import multiprocessing

import numpy as np
from PIL import PngImagePlugin

from semantic_segmentation.markup_readers import XMLBarcodeMarkupReader, \
    MultiplePathXMLMarkupReader, SegmentationMapMarkupReader, DeployMarkupReader
from semantic_segmentation.segmap_manager import SegmapManager

# было обнаружено что некоторые картинки не пиклятся https://github.com/python-pillow/Pillow/issues/1434
# из-за чего мультипроцессинг может падать, так как он пиклит и распикливает результат
# следующая строчка костыль против такого
PngImagePlugin.iTXt.__getnewargs__ = lambda x: (x.__str__(), '', '')
sizes_cache_filename = "sizes_cache.h5py"

supported_markup_types = {
    "Barcode": XMLBarcodeMarkupReader,
    "BarcodeMultipath": MultiplePathXMLMarkupReader,
    "BarcodeSegmap": SegmentationMapMarkupReader,
    "BarcodeDeploy": DeployMarkupReader,
}


#############################################
class PreprocessedInfo:
    def __init__(self, image, target_map, meta_info):
        self.image = image
        self.target_map = target_map
        self.meta_info = meta_info


class MetaInfo:
    def __init__(self, filename, original_markup, xscale, yscale):
        """
        Дополнительная информация о картинке (не нужна при обучении, но полезна на валидации/тесте)
        :param filename: имя файла с изображением или (_, filename)
        :param original_markup: разметка на ИСХОДНОМ изображении (на большой не аугментированной картинке)
        :param xscale: original.shape[0] / resized.shape[0]
        :param yscale: original.shape[1] / resized.shape[1]
        """
        self.filename = filename[1] if isinstance(filename, (tuple, list)) else filename
        self.markup = original_markup
        self.xscale = xscale
        self.yscale = yscale


class BatchGenerator:
    """
    Генератор батчей для обучения сети.
    """

    def __init__(self, path, batch_size, markup_type, net_config,
                 use_augmentation=False,
                 n_workers=3, yield_incomplete_batches=True, prepare_batch_size=1000, name="TrainGenerator"):
        """
        :param path: путь до разметки
        :param batch_size: размер одного батча
        :param markup_type: один из доступных типов чтения разметки - список смотреть в assert
        :param net_config:
        :param use_augmentation:
        :param n_workers: количество потоков для предобработки изображений
        :param yield_incomplete_batches: если установлено True могут появляться батчи размера меньше batch_size,
                                         иначе размер батча константный и равен batch_size
        :param prepare_batch_size: при генерации батчей сначала берется prepare_batch_size изображений из датасета,
                                   затем они группируются по размеру, батчи формируются из полученных групп,
                                   когда предобработанные изображения заканчиваются берутся новые и все повторяется
        """
        self._batch_size = batch_size
        self._net_config = net_config
        self._use_augmentation = use_augmentation
        self._name = name
        self.__n_workers = n_workers
        self.__prepare_batch_size = prepare_batch_size
        self.__yield_incomplete_batches = yield_incomplete_batches

        assert markup_type in supported_markup_types, \
            f"now we support only one from {supported_markup_types.keys()} as markup_type"
        self._reader = supported_markup_types[markup_type](path, net_config)
        self._reader.read_markup()
        self._image_names = self._reader.get_list_of_images()

        logging.info(f"Loaded dataset '{path}' with {self.get_images_per_epoch()} images")

    def get_images_per_epoch(self):
        """
        Возвращает размер датасета
        """
        return len(self._image_names)

    def get_epoch_size(self):
        """
        Вычисляет примерное число итераций, составляющих одну эпоху
        """
        return self.get_images_per_epoch() // self._batch_size

    def generate(self, add_metainfo=False):
        """
        Генерирует батчи сгруппированные по размеру изображений
        :param add_metainfo: если стоит True кроме (x, y) возвращает meta_info (доп информацию)
        :return: (batch_images, batch_targets, [meta_info])
        """
        assert not (add_metainfo and self.is_augmentation_used()), "will produce invalid metainfo (scales)"
        preprocessing_fn = self._net_config.get_preprocessing_fn()
        preprocessed_data = None
        # при отсутствии аугментации можно сэкономить на постоянном считывании/предобработке картинок
        # просто запомнив их, если ин не много (prepare_batch_size >= len(self._image_names))
        is_cached = False
        while True:
            border_index = 0
            if self.is_augmentation_used():
                np.random.shuffle(self._image_names)

            while border_index < len(self._image_names):
                if (not is_cached
                        and not self.is_augmentation_used()
                        and preprocessed_data is not None
                        and len(preprocessed_data) == len(self._image_names)):
                    logging.info(f"data cached!, {len(self._image_names)} images are cached in generator {self._name}")
                    is_cached = True
                elif not is_cached:
                    preprocessed_data = self._prepare_images(
                        self._image_names[border_index:border_index + self.__prepare_batch_size],
                        with_meta=add_metainfo
                    )
                    preprocessed_data.sort(key=lambda x: x.image.size)
                else:
                    # все данные уже посчитаны, ничего не надо делать
                    pass

                for img_size, group in itertools.groupby(preprocessed_data, key=lambda x: x.image.shape):
                    group = list(group)
                    group_border_index = 0
                    while len(group) - group_border_index > 0:
                        if (not self.__yield_incomplete_batches
                                and len(group) - group_border_index != self._batch_size):
                            # пропустить неполный батч
                            break
                        images = [preprocessing_fn(prepr_info.image)
                                  for prepr_info in group[group_border_index:group_border_index + self._batch_size]]
                        targets = [prepr_info.target_map
                                   for prepr_info in group[group_border_index:group_border_index + self._batch_size]]
                        images, targets = np.array(images), np.array(targets)
                        logging.debug(f"images.shape={images.shape}, targets.shape={targets.shape}")
                        if add_metainfo:
                            meta = [prepr_info.meta_info
                                    for prepr_info in group[group_border_index:group_border_index + self._batch_size]]
                            yield images, np.expand_dims(targets, -1), meta
                        else:
                            yield images, np.expand_dims(targets, -1)
                        group_border_index += self._batch_size

                border_index += self.__prepare_batch_size

    def _prepare_image(self, image_name, with_meta=False):
        """
        Нужно предпосчитать изображениe, размер + разметку
        :return PreprocessedInfo
        """
        try:
            image = self._reader.get_image(image_name)
            markup = self._reader.get_image_markup(image_name)
            original_w, original_h = image.size
            image, _, target = SegmapManager.prepare_image_and_target(image, markup, self._net_config,
                                                                      augment=self._use_augmentation)

            if self._net_config.is_grey():
                image = image.convert('L')  # делаем серым
            image = np.asarray(image)
            if image.ndim != 3:
                image = np.expand_dims(image, -1)
                assert image.ndim == 3
            target = np.array(target, dtype=int)

            if with_meta:
                rescaled_h, rescaled_w = image.shape[:2]
                return PreprocessedInfo(
                    image,
                    target,
                    MetaInfo(image_name, markup, original_w / rescaled_w, original_h / rescaled_h)
                )
            return PreprocessedInfo(image, target, None)
        except Exception as e:
            logging.error(f"unknown error in preprocessing image {image_name}, error: {e}")
            return None

    def _prepare_images(self, image_names, with_meta=False):
        # если попросили запустить в одном потоке, либо молчали про параллельность,
        # не будем использовать multiprocessing
        map_fn = functools.partial(self._prepare_image, with_meta=with_meta)
        if self.__n_workers is not None and self.__n_workers > 1:
            pool = multiprocessing.Pool(self.__n_workers)
            preprocessed_data = [x for x in pool.map(map_fn, image_names) if x is not None]
            pool.close()
            pool.join()
            if not preprocessed_data:
                logging.error("no data was preprocessed")
            return preprocessed_data
        else:
            return [x for x in map(map_fn, image_names) if x is not None]

    def is_augmentation_used(self):
        return self._use_augmentation
