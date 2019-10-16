#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Отрисовка визуализаций
"""
import numpy as np
from PIL import Image

from semantic_segmentation import utils
from semantic_segmentation.segmap_manager import SegmapManager


class Visualizer:
    """
    Класс для отрисовки визуализаций
    """

    @staticmethod
    def compute_visualizations(images, gt_segmap, predicted_segmap, found_objects, pixel_classification_mask=None):
        """
        Построить визуализации для изображений:
        - gt - с картой сегментации из разметки
        - seg_map - с предсказанной картой сегментации
        - postprocessed - результаты детекции после постобработки
        - [classification_gt] - пиксели на объектах раскрашиваются
            в зеленый/красный(верно/неверно классифицированный по типу)
        :param images: ориганальные картинки (до препроцессинга)
        :param gt_segmap: карта сегментаций из разметки
        :param predicted_segmap: предсказанная карта сегментации
        :param found_objects: найденные объекты (после постпроцессинга)
        :param pixel_classification_mask: попиксельная маска с
            -1 (неверная классификация пикселя), 1 (верная), 0 (здесь нет ни одного объекта)
        :return: dict с визуализациями
        """
        visualizations = dict()
        unnormalized_images = images.astype(np.uint8)

        visualizations["gt"] = Visualizer.visualize_segmentation_maps(unnormalized_images,
                                                                      gt_segmap)
        visualizations["seg_map"] = Visualizer.visualize_segmentation_maps(unnormalized_images,
                                                                           predicted_segmap)
        visualizations["postprocessed"] = Visualizer.draw_bboxes(unnormalized_images,
                                                                 found_objects)
        if pixel_classification_mask is not None:
            visualizations["classification_gt"] = \
                Visualizer.visualize_classification_masks(unnormalized_images,
                                                          pixel_classification_mask)
        return visualizations

    @staticmethod
    def visualize_segmentation_maps(images, targets, threshold=0.5):
        """
        Возвращает список изображений с отрисованными картами сегментаций
        :param images: список изображений (np.ndarray)
        :param targets: список соответствующих изображениям карт сегментации (должен быть той же длины что и images)
        :param threshold: порог по которому разделяются положительный и отрицательный классы
        :return:
        """
        assert len(images) == len(targets)
        return [
            Visualizer.visualize_segmentation_map(
                image,
                target,
                threshold=threshold
            )
            for image, target in zip(images, targets)
        ]

    @staticmethod
    def visualize_classification_masks(images, targets):
        assert len(images) == len(targets)
        return [
            Visualizer.visualize_classification_mask(
                image,
                target
            )
            for image, target in zip(images, targets)
        ]

    @staticmethod
    def visualize_classification_mask(image, is_pixel_correct):
        pillow_image = utils.pillow_rgb_fromarray(image)
        target_true = np.where(is_pixel_correct == 1, 255, 0).astype(np.uint8)
        target_false = np.where(is_pixel_correct == -1, 255, 0).astype(np.uint8)

        visualized_markup = pillow_image
        for target, color in zip((target_true, target_false), ((0, 255, 0), (255, 0, 0))):
            target = utils.pillow_grey_fomarray(target, dsize=pillow_image.size)
            visualized_markup = Visualizer.draw_segmentation_map(visualized_markup, target, color=color)
        return np.array(visualized_markup)

    @staticmethod
    def draw_bboxes(images, image_markups):
        """
        Возвращает список изображений с отрисованной разметкой
        :param images: список изображений
        :param image_markups: соответствующие им разметки
        :return:
        """
        assert len(images) == len(image_markups)
        return [
            np.array(Visualizer.draw_markup(utils.pillow_rgb_fromarray(image), markup))
            for image, markup in zip(images, image_markups)
        ]

    @staticmethod
    def visualize_segmentation_map(image, target, threshold=0.5, result_fname=None):
        """
        Накладывает карту сегментации на изображение как маску, то что больше threshold красится в зеленый
        :param image: np.ndarray RGB картинка
        :param target: np.ndarray массив вероятностей (той же размерности что и image но с одним каналом)
        :param threshold: порог по которому разделяются положительный и отрицательный классы
        :param result_fname: куда сохранять картинку (если None - никуда)
        :return:
        """
        pillow_image = utils.pillow_rgb_fromarray(image)
        target_as_image = np.where(target > threshold, 255, 0).astype(np.uint8)
        target_as_image = utils.pillow_grey_fomarray(target_as_image, dsize=pillow_image.size)

        visualized_markup = Visualizer.draw_segmentation_map(pillow_image, target_as_image,
                                                             result_fname=result_fname)
        return np.array(visualized_markup)

    @staticmethod
    def draw_markup(image, markup, result_fname=None):
        """
        Возвращает изображение с отрисованной на нем разметкой
        :param image: Pillow image
        :param markup: список ObjectMarkup
        :param result_fname: если не None, сохраняет туда получившееся изображение
        :return: image_with_markup
        """
        seg_map = SegmapManager.build_segmentation_map(image, markup, scale=1, for_drawing=True)
        return Visualizer.draw_segmentation_map(image, seg_map, result_fname=result_fname)

    @staticmethod
    def draw_segmentation_map(image, seg_map, result_fname=None, color=(0, 255, 0)):
        """
        Возвращает изображение с отрисованной на нем картой сегментаций
        :param image: Pillow image
        :param seg_map: Pillow image of type 'L' or '1' такого же размера что и image
        :param result_fname: если не None, сохраняет туда получившееся изображение
        :param color: RGB цвет которым рисуется карта сегментации
        :return: image_with_segmentation_map
        """
        colored_image = Image.blend(image, Image.new('RGB', image.size, color=color), alpha=0.5)
        image_object = Image.composite(colored_image, image, seg_map)
        if result_fname:
            image_object.save(result_fname)
        return image_object
