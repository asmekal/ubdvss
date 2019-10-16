#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Запуск модели с
- подсчетом всех метрик
- отрисовкой результатов и ошибок
- сохранением результатов работы
"""
import os

import cv2
import numpy as np

from semantic_segmentation import utils
from semantic_segmentation.data_markup import ClassifiedObjectMarkup
from semantic_segmentation.evaluation import DatasetMetricCalculator, ImageResultCategories
from semantic_segmentation.segmap_manager import SegmapManager
from semantic_segmentation.visualizations import Visualizer


class ModelRunner:
    """
    Класс для прогона модели на наборе изображений с
    - подсчетом метрик по различным порогам iou (precision, recall, f1)
        и по картинкам (precision/recall/iou _by_area и detection rate)
    - сохранением результатов работы (найденные прямоугольники и разметка сохраняются в .txt файлы)
    - отрисовкой разметки, результатов и ошибок
    """

    def __init__(self, net_config, pixel_threshold=0.5):
        """
        :param net_config: конфигурация сети
        :param pixel_threshold: pixel_probability > pixel_threshold класс считается положительным
        """
        self._net_config = net_config
        eps = 1e-9
        self._logit_threshold = - np.log(1 / np.clip(pixel_threshold, eps, 1 - eps) - 1)

    def run(self, model, data_generator, n_images, save_dir=None, save_visualizations=False):
        """
        Основной метод, находящий результат, отрисовывающий визуализации и подсчитывающий метрики

        Подсчет метрик по различным порогам iou (precision, recall, f1) и отрисовка картинок с разметкой,
        предсказанными картами сегментаций и финальными детекциями после пост-процессинга
        :param model: оцениваемая модель
        :param data_generator: генератор батчей
        :param n_images: сколько изображений использовать для подсчета метрик
        :param save_dir: директория для сохранения результатов, если не указана, не сохраняется
        :param save_visualizations: нужно ли сохранять визуализации
        :return: (scalar_logs, visualizations)
        """
        evaluator = DatasetMetricCalculator(self._net_config)
        saver = ResultSaver(save_dir, save_visualizations)

        images_processed = 0
        visualizations = dict()
        # батч с этой картинкой будет нарисован в логах тензорборда
        image_num_to_visualize = np.random.randint(0, n_images)
        while images_processed < n_images:
            # не logging потому что там сложно с тем чтобы не переносить строчку
            # а так не замусоривается лог
            print("\rimages processed: {}/{}".format(images_processed, n_images), end='', flush=True)

            images, targets, meta_infos = next(data_generator)

            detection_logits, classification_logits, found_objects = self.predict(model, images, rescale=False)
            # эти будем отрисовывать чтобы не перемасштабировать картинки
            drawn_found_objects = found_objects
            # а на этих уже смотрим целевые метрики, перемасштабируя их так
            # чтобы они обозначали объекты на изначальном разрешении изображений
            found_objects = self.rescale(found_objects, meta_infos)
            # разметка
            gt_objects = [meta_info.markup for meta_info in meta_infos]
            image_metrics, pixel_classification_mask = evaluator.evaluate_batch(
                gt_objects, found_objects, gt_segmap=targets,
                classification_logits=classification_logits, meta_infos=meta_infos)
            denorm = self._net_config.get_depreprocessing_fn()
            if save_dir:
                saver.save_gt_and_prediction(gt_objects, found_objects, meta_infos)
                image_categories = [ImageResultCategories.get_categories(m) for m in image_metrics]
                # если визуализации в принципе нужно сохранять и
                # в этом батче нужно отрисовывать хотя бы одну картинку
                if save_visualizations and any(image_categories):
                    visualizations = Visualizer.compute_visualizations(
                        images=denorm(images),
                        gt_segmap=targets, predicted_segmap=detection_logits,
                        found_objects=drawn_found_objects,
                        pixel_classification_mask=pixel_classification_mask)
                    saver.save_visualizations(image_categories, meta_infos, visualizations)
            elif images_processed <= image_num_to_visualize < images_processed + len(images):
                visualizations = Visualizer.compute_visualizations(
                    images=denorm(images),
                    gt_segmap=targets, predicted_segmap=detection_logits,
                    found_objects=drawn_found_objects,
                    pixel_classification_mask=pixel_classification_mask)

            images_processed += len(images)
        # новая строка после images_processed:k/n
        print("\rimages processed: {}/{}\n".format(images_processed, n_images), flush=True)

        scalar_logs = evaluator.get_metrics()
        return scalar_logs, visualizations

    def predict(self, model, images, rescale=False, meta_infos=None):
        """
        Посчитать карту сегментации (для детекции и классификации) и сделать постобработку
        Внимание! found_objects будут относительно поданных (в этот метод) картинок, чтобы получить результат
            для оригинальных надо их еще перемасштабировать используя meta_info
        :param model:
        :param images:
        :param rescale: False - found_objects будут относительно поданных (в этот метод) картинок
                        True - found_objects будут относительно исходных картинок (для этого надо подать и meta_infos)
        :param meta_infos: метаинформация об изображениях, необходимо указывать если нужно установить rescale=True
        :return: detection_logits, classification_logits, found_objects
        """
        assert not rescale or (meta_infos is not None and len(images) == len(meta_infos))

        predicted_targets = model.predict(images)

        detection_logits = predicted_targets[..., :1]
        classification_logits = predicted_targets[..., 1:]

        detection_logits = np.where(detection_logits > self._logit_threshold, 1, 0)

        # эти отрисовываются (чтобы не перемасштабировать картинки)
        found_objects = [
            SegmapManager.postprocess(
                detection_logits[i],
                classification_logits[i] if self._net_config.is_classification_supported() else None,
                scale=self._net_config.get_scale(),
                min_area_threshold=self._net_config.get_min_pixels_for_detection()
            ) for i in range(predicted_targets.shape[0])
        ]
        if rescale:
            found_objects = self.rescale(found_objects, meta_infos)

        return detection_logits, classification_logits, found_objects

    @staticmethod
    def rescale(found_objects, meta_infos):
        assert len(found_objects) == len(meta_infos)
        return [[
            found_objects[i][j].create_same_markup(
                utils.rescale_bbox(found_objects[i][j].bbox,
                                   xscale=meta_infos[i].xscale,
                                   yscale=meta_infos[i].yscale)
            ) for j in range(len(found_objects[i]))] for i in range(len(found_objects))]


# Вспомогательные классы ########################################################


class ResultSaver:
    """
    Сохраняет результаты работы сети и визуализации
    """

    def __init__(self, save_dir=None, save_visualizations=False):
        """
        :param save_dir: путь куда сохранять результаты, если None - никуда
        :param save_visualizations: сохранять ли визуализации
        """
        self._save_dir = save_dir
        self._are_visualizations_saved = save_visualizations and save_dir is not None
        self._make_dirs()

    def _make_dirs(self):
        if self._save_dir:
            self._save_gt_dir = os.path.join(self._save_dir, 'markup')
            self._save_predictions_dir = os.path.join(self._save_dir, 'predictions')
            os.makedirs(self._save_gt_dir, exist_ok=True)
            os.makedirs(self._save_predictions_dir, exist_ok=True)
            if self._are_visualizations_saved:
                self._save_images_dir = os.path.join(self._save_dir, 'images')
                os.makedirs(self._save_images_dir, exist_ok=True)
                for category in ImageResultCategories.get_folders():
                    os.makedirs(os.path.join(self._save_images_dir, category), exist_ok=True)

    def save_gt_and_prediction(self, gt_objects, found_objects, meta_infos):
        """
        Сохраняет результат и разметку в файл
        :param gt_objects:
        :param found_objects:
        :param meta_infos:
        :return:
        """
        for i, meta_info in enumerate(meta_infos):
            ResultSaver.save_markup_csv(
                filename=os.path.join(self._save_gt_dir, meta_info.filename + '.txt'),
                markups=gt_objects[i]
            )
            ResultSaver.save_markup_csv(
                filename=os.path.join(self._save_predictions_dir, meta_info.filename + '.txt'),
                markups=found_objects[i]
            )

    def save_visualizations(self, need_visualize_categories, meta_infos, visualizations):
        """
        Сохраняет картинки, возможно в несколько папок одни и те же, возможно не все
        :param need_visualize_categories: список (для каждой картинки) списков имен директорий в которые сохранять,
            для изображений, которым соответствует пустой список не сохраняем визуализации
        :param meta_infos: метаинформация о изображениях, отсюда берется имя файла
        :param visualizations: сами визуализации
        :return:
        """
        for i, meta_info in enumerate(meta_infos):
            for category in need_visualize_categories[i]:
                for tag in visualizations:
                    save_fname = os.path.join(self._save_images_dir, category,
                                              f"{meta_info.filename}.{tag}.png")
                    cv2.imwrite(save_fname, cv2.cvtColor(visualizations[tag][i], cv2.COLOR_RGB2BGR))

    @staticmethod
    def save_markup_csv(filename, markups):
        """
        Запись в CSV массива четырехугольников
        """
        markup_str = ''
        for markup in markups:
            markup_str += '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},""'.format(
                *[int(xy) for xy in markup.bbox])
            if isinstance(markup, ClassifiedObjectMarkup):
                markup_str += f',{markup.object_type}\n'
            else:
                markup_str += f'\n'
        with open(filename, "w") as text_file:
            text_file.write(markup_str)
