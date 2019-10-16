#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Классы для чтения разметки из различных форматов
"""
import logging
import os
import xml.etree.ElementTree
from abc import abstractmethod
from enum import Enum

import numpy as np
from PIL import Image

from semantic_segmentation import utils
from semantic_segmentation.data_markup import ObjectMarkup, ClassifiedObjectMarkup


class BarcodeType(Enum):
    BARCODES_1D = 0
    BARCODES_2D = 1
    BARCODES_2D_SQUARED = 2
    BARCODES_ALL = 3


BARCODE_1D_TYPES = ('Code128', 'Patch', 'EAN8', 'Code93', 'UCC128', 'EAN13',
                    'Industrial25', 'Code32', 'FullASCIICode',
                    'UPCE', 'MATRIX25', 'Code39', 'IATA25', 'UPCA', 'CODABAR', 'Interleaved25',
                    # типы баркодов ниже выглядят как postnet - примерно так iIIiiiIiIiIIiiiIIiiiI
                    'Postnet', 'AustraliaPost', 'Kix', 'IntelligentMail', 'RoyalMailCode')  # [:-5]

BARCODE_2D_SQUARED_TYPES = ('QRCode', 'Aztec', 'MaxiCode', 'DataMatrix')  # DataMatrix не всегда!
BARCODE_2D_TYPES = BARCODE_2D_SQUARED_TYPES + ('PDF417',)


class BaseMarkupReader:
    """
    Интерфейс для чтения разметки
    """

    def __init__(self, path, net_config):
        self._path = path
        self._net_config = net_config

    @abstractmethod
    def read_markup(self):
        pass

    @abstractmethod
    def get_list_of_images(self):
        """
        :return: список имен всех изобраений у этого ридера
        """
        pass

    @abstractmethod
    def get_image_markup(self, image_name):
        """
        :param image_name:
        :return: список ObjectMarkup разметок объектов на изображении
        """
        pass

    @abstractmethod
    def get_image(self, image_name):
        """
        :param image_name:
        :return: изображения Image
        """
        pass


class DeployMarkupReader(BaseMarkupReader):
    """
    Ридер для изображений без разметки, может использоваться для predict
    """

    def __init__(self, images_path, net_config):
        super().__init__(images_path, net_config)
        self.__images_folder_path = images_path
        self.__markup = dict()

    def read_markup(self):
        for image_file in os.listdir(self.__images_folder_path):
            image_name, image_ext = os.path.splitext(image_file)
            if utils.is_image_extension(image_ext):
                self.__markup[image_file] = []

    def get_list_of_images(self):
        return list(self.__markup.keys())

    def get_image_markup(self, image_name):
        return self.__markup[image_name]

    def get_image(self, image_name):
        return Image.open(os.path.join(self.__images_folder_path, image_name)).convert('RGB')


class FileMarkupReader(BaseMarkupReader):
    """
    Читает разметку в предположении, что есть отдельная директория с изображениями
    и отдельная директория с файлами разметки, файлы в которых соответствуют друг другу по имени,
    за исключением, возможно, расширения
    """

    def __init__(self, path, net_config, images_folder='Image', markup_folder='Markup'):
        super().__init__(path, net_config)
        self.__images_folder_path = os.path.join(path, images_folder)
        self.__markup_folder_path = os.path.join(path, markup_folder)
        self.__markup = dict()
        self.__full_filename = dict()

    def get_list_of_images(self):
        """
        :return: лист с именами изображений
        """
        return list(self.__markup.keys())

    def read_markup(self):
        n_errors = 0
        n_successfull_reads = 0
        logging.info("Reading markup from {}".format(self.__images_folder_path))
        for markup_filename in os.listdir(self.__markup_folder_path):
            fname, ext = os.path.splitext(markup_filename)
            if not self._is_markup_file_extension(ext):
                continue
            try:
                image_filename = utils.find_corresponding_image(self.__images_folder_path, fname)
                markup = self._read_markup_from_file(os.path.join(self.__markup_folder_path, markup_filename))
                self.__markup[fname] = markup
                self.__full_filename[fname] = image_filename
                n_successfull_reads += 1
            except Exception as e:
                n_errors += 1
                logging.error("{} can't read {}".format(n_errors, e))
        logging.info("{}/{} files read successfully".format(n_successfull_reads, n_successfull_reads + n_errors))

    @abstractmethod
    def _is_markup_file_extension(self, ext):
        """
        Является ли данное расширение валидным для файла разметки
        :param ext:
        :return:
        """
        pass

    @abstractmethod
    def _read_markup_from_file(self, markup_file_path, skip_empty_markup=True):
        """
        Читает разметку из файла
        :param markup_file_path: путь до файла с разметкой
        :param skip_empty_markup: пропускать файлы в которых не указана разметка
        :return: список объектов вида ObjectMarkup с информацией об объектах на изображении
        """
        pass

    def get_image_markup(self, image_name):
        """
        :param image_name:
        :return: разметка местоположения слов в виде списка объектов типа ObjectMarkup
        """
        return self.__markup[image_name]

    def get_image(self, image_name):
        """
        :param image_name:
        :return: изображения Image
        """
        return Image.open(os.path.join(self.__images_folder_path, self.__full_filename[image_name])).convert('RGB')


class XMLBarcodeMarkupReader(FileMarkupReader):
    """
    Читает разметку для баркодов из xml, генерируемых нашими разметчиками
    """

    def __init__(self, path, net_config, images_folder='Image', markup_folder='Markup',
                 barcode_type=BarcodeType.BARCODES_ALL):
        super().__init__(path, net_config, images_folder=images_folder, markup_folder=markup_folder)
        self.__barcode_type = barcode_type

    def _read_markup_from_file(self, markup_file_path, skip_empty_markup=True):
        """
        Читает разметку из файла
        :param markup_file_path: путь до файла с разметкой
        :param skip_empty_markup: пропускать файлы в которых не указана разметка
        :return: список объектов ObjectMarkup
        """
        markup = []
        words = xml.etree.ElementTree.parse(markup_file_path).getiterator("Barcode")
        for word in words:
            border_points = []
            barcode_type = word.get('Type')
            if not barcode_type or barcode_type.strip() == "":
                # это может быть валидный баркод который мы хотим распознать и тогда выкидывать его плохо
                # а может быть не валидный (или другого типа) и тогда его надо выкинуть
                # так что лучше заигнорить всю картинку целиком дабы не заморачиваться
                raise ValueError("Unknown barcode type (empty string). "
                                 "Image {} will be skipped".format(os.path.basename(markup_file_path)))
            elif self.__barcode_type == BarcodeType.BARCODES_1D \
                    and barcode_type not in BARCODE_1D_TYPES:
                logging.info("Skipping barcode type \"{}\". Only barcodes of types {} "
                             "are considered in markup".format(barcode_type,
                                                               str(BARCODE_1D_TYPES)))
                continue
            elif self.__barcode_type == BarcodeType.BARCODES_2D \
                    and barcode_type not in BARCODE_2D_TYPES:
                logging.info("Skipping barcode type \"{}\". Only 2D barcodes "
                             "are considered in markup".format(barcode_type))
                continue
            elif self.__barcode_type == BarcodeType.BARCODES_2D_SQUARED \
                    and barcode_type not in BARCODE_2D_SQUARED_TYPES:
                logging.info("Skipping barcode type \"{}\". Only squared 2D barcodes "
                             "are considered in markup".format(barcode_type))
                continue

            if not self._net_config.is_class_supported(barcode_type):
                logging.info(f"Skippping barcode type \"{barcode_type}\","
                             f" not in current classification object types: {self._net_config.get_class_names()}")
                continue

            for point in word.iter('Point'):
                border_points.append(
                    (int(point.get('X')), int(point.get('Y')))
                )
            assert len(border_points) == 4

            quad_vertices = utils.fix_quadrangle(np.array(border_points))
            assert quad_vertices.shape == (4, 2)

            if self.__barcode_type == BarcodeType.BARCODES_2D_SQUARED:
                sides_lengths = utils.get_polygon_sides_lengths(quad_vertices)
                if not utils.is_quad_square(quad_vertices):
                    # такое может быть с DataMatrix, лучше пропустить всю картинку
                    # т.к. она скорее всего из пакета с не квадратными изображениями
                    raise ValueError("Skipping file {} which contains not squared "
                                     "barcode {} with sides {}".format(markup_file_path,
                                                                       barcode_type, str(sides_lengths)))

            if self._net_config.is_classification_supported():
                markup.append(ClassifiedObjectMarkup(
                    quad_vertices.reshape((8,)),
                    self._net_config.get_class_id(barcode_type)
                ))
            else:
                markup.append(ObjectMarkup(quad_vertices.reshape((8,))))
        if not markup and skip_empty_markup:
            raise ValueError(
                "Skipping suspicious empty markup file (no barcodes in markup) {}".format(markup_file_path))
        return markup

    def _is_markup_file_extension(self, ext):
        return ext.lower() == '.xml'


class SegmentationMapMarkupReader(FileMarkupReader):
    """
    Читает разметку из карт сегментаций (в виде многоугольников-контуров)
    пока что было необходимо исключительно для чтения разметки из публичных датасетов,
    в которых тип везде один и тот же (EAN13 -> 5)
    """

    def __init__(self, path, net_config, images_folder='Image', markup_folder='Detection'):
        super().__init__(path, net_config, images_folder=images_folder, markup_folder=markup_folder)

    def _is_markup_file_extension(self, ext):
        return utils.is_image_extension(ext)

    def _read_markup_from_file(self, markup_file_path, skip_empty_markup=True):
        seg_map_image = Image.open(markup_file_path).convert('L')
        contours, _ = utils.get_contours_and_boxes(seg_map_image, min_area=-1)
        # приходится брать convex hull от контура,
        # иначе shapely отказывается работать на невыпуклых контурах (когда считает intersection в evaluation)
        # это вносит некоторую ошибку, но в большинстве случаев она будет пренебрежимо мала
        # пример карты сегментации из которой получится невыпуклый контур:
        # 0 0 1 0 0
        # 0 0 1 0 0
        # 1 1 1 1 1
        # 0 0 1 0 0
        # 0 0 1 0 0
        # также здесь отфильтровываются все контура в которых только одна точка
        # такие пока возникали только на датасете Artelab как артефакты разметки
        return [ClassifiedObjectMarkup(utils.get_convex_hull(cnt).reshape((-1,)), 5)
                for cnt in contours if len(utils.get_convex_hull(cnt).reshape((-1,))) > 2]


# TODO: можно написать общий multipath reader а этот будет partial от него
class MultiplePathXMLMarkupReader(BaseMarkupReader):
    def __init__(self, paths, net_config, *args, **kwargs):
        if isinstance(paths, str):
            paths = paths.split(',')
        logging.info(f"Reading markup from {paths}")
        self._paths = paths
        self._readers = [XMLBarcodeMarkupReader(path, net_config, *args, **kwargs) for path in paths]

    def get_list_of_images(self):
        return [
            (i, image_name) for i in range(len(self._readers)) for image_name in self._readers[i].get_list_of_images()
        ]

    def read_markup(self):
        for path, reader in zip(self._paths, self._readers):
            logging.info(f"Reading markup from {path}...")
            reader.read_markup()

    def get_image_markup(self, image_name):
        return self._readers[image_name[0]].get_image_markup(image_name[1])

    def get_image(self, image_name):
        return self._readers[image_name[0]].get_image(image_name[1])
