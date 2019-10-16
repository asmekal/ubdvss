#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2018. All rights reserved.
"""
Работа с картами сегментаций - построение, препроцессинг, постпроцессинг, аугментация и т.д.
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from semantic_segmentation.augmentation import SegLinksImageAugmentation
from semantic_segmentation.data_markup import ObjectMarkup, ClassifiedObjectMarkup
from semantic_segmentation.utils import get_contours_and_boxes, np_softmax


class SegmapManager:
    """
    Класс для работы с картами сегментаций
    """

    @staticmethod
    def prepare_image_and_target(image, markup, net_config, is_training=False):
        """
        преобразовать изображение и разметку в формат принимаемый сеткой (заданный net_config),
        построить карту сегментаций
        :param image: исходное изображение
        :param markup: разметка в виде списка четырехугольников заданных в виде [x1, y1, ..., x4, y4]
        :param net_config: конфигурация сети
        :param is_training:
        :return: processed_image, processed_markup, segmentation_map
        """
        if is_training:
            image, markup = SegmapManager._augment(image, markup, net_config)
        rescaled_image, rescaled_markup = SegmapManager._rescale_image_and_markup(image, markup, net_config)
        segmentation_map = SegmapManager._build_segmentation_map(rescaled_image, rescaled_markup,
                                                                 scale=net_config.get_scale())
        return rescaled_image, rescaled_markup, segmentation_map

    @staticmethod
    def draw_markup(image, markup, result_fname=None):
        """
        возвращает изображение с отрисованной на нем разметкой
        :param image:
        :param markup:
        :param result_fname: если не None, сохраняет туда получившееся изображение
        :return: image_with_markup
        """
        seg_map = SegmapManager._build_segmentation_map(image, markup, scale=1, for_drawing=True)
        return SegmapManager.draw_segmentation_map(image, seg_map, result_fname=result_fname)

    @staticmethod
    def get_bboxes_from_segmentation_map(seg_map, scale=1, min_block_threshold=5):
        """
        выполняет постпроцессинг, т.е. по полученной карте сегментаций возвращает найденные объекты
        :param seg_map: карта сегментаций
        :param scale: отношение размера исходного изображения к карте сегментаций
            (например изображение было 200x200, а карта сегментаций 50x50, тогда scale=4)
        :param min_block_threshold: минимальное количество прилежащих друг к другу 1 в карте сегментации,
            чтобы считаться задетекченным объектом (таким образом повышается precision
            уменьшая количество случайных срабатываний)
        :return: разметка в виде списка четырехугольников заданных в виде [x1, y1, ..., x4, y4]
        """
        # ВНИМАНИЕ! opencv не всегда находит прямоугольник который покрывает прямо все точки контура
        # возможна ошибка 2-3 пикселя (обычно справа и снизу изображения)
        # из-за этого при постпроцессинге иногда теряются границы (особенно когда объекты прямо на границе изображения)
        # править это и переписывать вручную считаю нецелесообразным, т.к. этот проект больше для исследования
        # и для исследования такая ошибка не критична
        contours, boxes = get_contours_and_boxes(seg_map, min_area=min_block_threshold)
        boxes = [np.round(box * scale).astype(int) for box in boxes]
        return boxes

    @staticmethod
    def postprocess(seg_map, seg_map_class_logits=None, scale=1, min_block_threshold=5):
        contours, boxes = get_contours_and_boxes(seg_map, min_area=min_block_threshold)
        boxes = [np.round(box * scale).astype(int) for box in boxes]
        if seg_map_class_logits is None:
            return [ObjectMarkup(bbox) for bbox in boxes]

        seg_map_class_ps = np_softmax(seg_map_class_logits, axis=-1)

        class_ids = []
        for cnt in contours:
            mask = np.zeros(seg_map.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            class_logits = seg_map_class_ps[mask.astype(np.bool)].mean(axis=0)
            class_id = np.argmax(class_logits)
            class_ids.append(class_id)

        return [ClassifiedObjectMarkup(bbox, class_id) for bbox, class_id in zip(boxes, class_ids)]

    @staticmethod
    def draw_segmentation_map(image, seg_map, result_fname=None, color=(0, 255, 0)):
        """
        возвращает изображение с отрисованной на нем картой сегментаций
        :param image:
        :param seg_map:
        :param result_fname: если не None, сохраняет туда получившееся изображение
        :param color: RGB цвет которым рисуется карта сегментации
        :return: image_with_segmentation_map
        """
        colored_image = Image.blend(image, Image.new('RGB', image.size, color=color), alpha=0.5)
        image_object = Image.composite(colored_image, image, seg_map)
        if result_fname:
            image_object.save(result_fname)
        return image_object

    @staticmethod
    def _augment(image, markup, net_config):
        modifier = SegLinksImageAugmentation(image, markup, net_config)
        aug_image = modifier.get_modified_image()
        aug_markup = modifier.get_modified_markup()
        return aug_image, aug_markup

    @staticmethod
    def _build_segmentation_map(image, markup, scale=1, for_drawing=False):
        """

        :param image:
        :param markup:
        :param scale:
        :param for_drawing: если True в seg_map на местах bboxes будут 255, иначе то что стоит в markup.object_type
        :return: pillow image (mode='L') - segmentation map
        """
        w, h = image.size
        assert w % scale == 0 and h % scale == 0
        image_segmap = Image.new(mode='L', size=(w // scale, h // scale), color=0)
        draw = ImageDraw.Draw(image_segmap)
        for object_markup in markup:
            drawn_bbox = SegmapManager._proper_round(object_markup.bbox / scale)
            if for_drawing:
                fill_color = 255
            else:
                # добавляем 1 т.к. obj_type нумеруются с нуля
                fill_color = object_markup.object_type + 1 if isinstance(object_markup, ClassifiedObjectMarkup) else 1
                assert fill_color <= 255, "No more than 255 classes are supported"
            draw.polygon(drawn_bbox.tolist(), fill=fill_color)
        return image_segmap

    @staticmethod
    def _proper_round(markup_bbox):
        """
        надо правильно округлить, чтобы не потерять части объекта
        :param markup_bbox: [x1, y1, ..., x4, y4] (координаты возможно не целые)
        :return:
        """
        xs = markup_bbox[::2]
        ys = markup_bbox[1::2]
        assert len(xs) == len(ys) == 4

        # итак, надо правильно округлить чтобы не потерять части объекта - вверх или вниз?
        # если для x_i есть хотя бы 2 других x_j больше него, значит объект лежит "выше" по координатам чем x_i
        # соответственно если мы округлим вверх - можем потерять часть объекта - значит округляем вниз
        # и наоборот если есть хотя бы 2 других x_j меньше него округляем вверх
        # ситуация когда 1 больше 1 меньше 1 равен - округляется вверх (это не оптимально, но предполагается,
        # что такая ситуация почти не встречается)
        xs_greater = [sum(1 for _x in xs if _x > x) for x in xs]
        ys_greater = [sum(1 for _y in ys if _y > y) for y in ys]
        xs = [math.floor(x) if n_greater > 1 else math.ceil(x) for (x, n_greater) in
              zip(xs, xs_greater)]
        ys = [math.floor(y) if n_greater > 1 else math.ceil(y) for (y, n_greater) in
              zip(ys, ys_greater)]
        return np.ravel(list(zip(xs, ys))).astype(np.int32)

    @staticmethod
    def _rescale_image_and_markup(image, markup, net_config):
        w, h = image.size
        # кратно какому минимальному размеру могут быть стороны
        side_multiple = net_config.get_side_multiple()
        # максимальный размер максимальной стороны изображения
        max_side = net_config.get_max_side()

        # если изображение слишком высокого разрешения - уменьшим его чтобы работало быстрее
        if max(w, h) > max_side:
            # большую сторону делаем максимально разумно возможной, меньшую уменьшаем пропорционально
            downscale = max_side / max(h, w)
            if w > h:
                new_w = max_side
                new_h = max(1, round(h * downscale / side_multiple)) * side_multiple
            else:
                new_w = max(1, round(w * downscale / side_multiple)) * side_multiple
                new_h = max_side
        else:
            # если изображение не большое просто сдалаем размеры кратно side_multiple
            new_w = max(1, round(w / side_multiple)) * side_multiple
            new_h = max(1, round(h / side_multiple)) * side_multiple

        resized_image = image.resize(size=(new_w, new_h), resample=Image.BICUBIC)  # TODO: check bilinear
        if not markup:
            return resized_image, markup
        scales = np.array([new_w / w, new_h / h] * 4)
        resized_markup = [m.create_same_markup(m.bbox * scales) for m in markup]
        return resized_image, resized_markup
