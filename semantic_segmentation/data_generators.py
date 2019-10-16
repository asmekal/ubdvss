#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2018. All rights reserved.
"""
Здесь генератор батчей для обучения сети
"""

import functools
import itertools
import logging
import multiprocessing

import numpy as np
from PIL import PngImagePlugin

from semantic_segmentation.markup_readers import BarcodeMarkupReader, \
    MultiplePathBarcodeMarkupReader, DeployMarkupReader
from semantic_segmentation.segmap_manager import SegmapManager

# было обнаружено что некоторые картинки не пиклятся https://github.com/python-pillow/Pillow/issues/1434
# из-за чего мультипроцессинг может падать, так как он пиклит и распикливает результат
# следующая строчка костыль против такого
PngImagePlugin.iTXt.__getnewargs__ = lambda x: (x.__str__(), '', '')
sizes_cache_filename = "sizes_cache.h5py"

supported_markup_types = {
    "Barcode": BarcodeMarkupReader,
    "Barcode_multipath": MultiplePathBarcodeMarkupReader,
    "Barcode_deploy": DeployMarkupReader,
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
        Доп информация о картинке (не нужна при обучении, но полезна на валидации/тесте)
        :param filename:
        :param xscale: original.shape / resized.shape
        :param yscale:
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
                 validation_mode=False, n_workers=3, yield_incomplete_batches=True, prepare_batch_size=1000):
        """
        :param path: путь до разметки
        :param batch_size: размер одного батча
        :param markup_type: один из доступных типов чтения разметки - список смотреть в assert
        """
        self._batch_size = batch_size
        self._net_config = net_config
        self._validation_mode = validation_mode
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
        возвращает размер датасета
        """
        return len(self._image_names)

    def get_epoch_size(self):
        """
        вычисляет примерное число итераций, составляющих одну эпоху
        """
        return self.get_images_per_epoch() // self._batch_size

    def generate(self, add_metainfo=False):
        """
        генерирует батчи сгруппированные по размеру изображений
        :param add_metainfo: если стоит True кроме (x, y) возвращает meta_info (доп информацию)
        :return: (batch_images, batch_targets, [meta_info])
        """
        preprocessing_fn = self._net_config.get_preprocessing_fn()
        while True:
            border_index = 0
            if not self._validation_mode:
                np.random.shuffle(self._image_names)

            while border_index < len(self._image_names):
                preprocessed_data = self._prepare_images(
                    self._image_names[border_index:border_index + self.__prepare_batch_size],
                    with_meta=add_metainfo
                )

                preprocessed_data.sort(key=lambda x: x.image.size)
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
        нужно предпосчитать изображениe, размер + разметку
        :return PreprocessedInfo
        """
        try:
            image = self._reader.get_image(image_name)
            markup = self._reader.get_image_markup(image_name)
            original_w, original_h = image.size
            image, _, target = SegmapManager.prepare_image_and_target(image, markup, self._net_config,
                                                                      is_training=not self._validation_mode)

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
