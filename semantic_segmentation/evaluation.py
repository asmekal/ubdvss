#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Классы для вычисления метрик качества для задачи поиска объектов
немного модифицированный код из FindText
- добавлены метрики для классификации объектов по типам
- оценка для произвольных многоугольников вместо четырехугольников
"""

import logging
import math

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from semantic_segmentation import utils


class FtMetrics:
    def __init__(self, all_type_names=None, compute_classification_metrics=False):
        """
        Note: Второй параметр нужен, т.к. в этом месте мы не знаем есть ли классификация на самом деле, несмотря на
        all_object_types, который, если определен, не гарантирует, что классификация на самом деле есть
        :param all_type_names: список всех строк-имен типов/классов/языков объектов, которые детектятся
        :param compute_classification_metrics: нужно ли считать метрики для классификации
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.one_to_one = 0  # те из tp, которые были сматчены идеально
        self.one_to_many = 0  # те из tp, которые были сматчены несколькими областями
        self.many_to_one = 0  # те из tp, которые были сматчены в группах по несколько

        self.average_iou = 0
        self.matched_boxes_count = 0  # число пар сматченных областей, по которым вычисляется average_iou

        self.average_iou_by_area = 0
        self.average_precision_by_area = 0
        self.average_recall_by_area = 0
        self.matched_images_count = 0  # число пар сматченных изображений, по которым вычисляются метрики ...by_area

        self.detection_rate = 0  # количество изображений на которых iou больше порога

        self.all_type_names = all_type_names
        self.confusion_matrix = None
        if all_type_names is not None and compute_classification_metrics:
            self.confusion_matrix = np.zeros((len(all_type_names), len(all_type_names)), dtype=np.float32)

    def append(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

        self.one_to_one += other.one_to_one
        self.one_to_many += other.one_to_many
        self.many_to_one += other.many_to_one

        self.average_iou, self.matched_boxes_count = self.__sum_up(
            self.average_iou,
            self.matched_boxes_count,
            other.average_iou,
            other.matched_boxes_count
        )

        self.average_iou_by_area, matched_images_count = self.__sum_up(
            self.average_iou_by_area,
            self.matched_images_count,
            other.average_iou_by_area,
            other.matched_images_count
        )
        self.average_precision_by_area, matched_images_count = self.__sum_up(
            self.average_precision_by_area,
            self.matched_images_count,
            other.average_precision_by_area,
            other.matched_images_count
        )
        self.average_recall_by_area, matched_images_count = self.__sum_up(
            self.average_recall_by_area,
            self.matched_images_count,
            other.average_recall_by_area,
            other.matched_images_count
        )
        self.detection_rate, matched_images_count = self.__sum_up(
            self.detection_rate,
            self.matched_images_count,
            other.detection_rate,
            other.matched_images_count
        )

        self.matched_images_count = matched_images_count

        if self.confusion_matrix is not None:
            self.confusion_matrix += other.confusion_matrix

    def get_metrics(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    def get_report(self):
        precision, recall, f1 = self.get_metrics()

        report_str = 'pr = {:.4f}, r = {:.4f}, f1 = {:.4f} ' \
                     '[tp = {} = {} (1-1) + {} (1-m) + {} (m-1); fp = {}; fn = {}];' \
                     ' iou boxes = {:.2f}  >>> by area: pr = {:.4f}, r = {:.4f}, iou = {:.4f}, rate = {:.4f}' \
            .format(
                precision, recall, f1,
                self.tp, self.one_to_one, self.one_to_many, self.many_to_one,
                self.fp, self.fn, self.average_iou,
                self.average_precision_by_area, self.average_recall_by_area, self.average_iou_by_area,
                self.detection_rate
            )
        return report_str

    def get_confusion_matrix_report(self):
        assert self.confusion_matrix is not None and self.all_type_names is not None, \
            "confusion matrix undefined in report"

        acc_per_type = np.diag(self.confusion_matrix) / np.maximum(np.sum(self.confusion_matrix, axis=1), 1)
        acc_as_df = pd.DataFrame(acc_per_type.reshape(-1, 1), columns=['Accuracy'], index=self.all_type_names)
        pd.set_option('precision', 3)
        pd.set_option("display.max_columns", self.confusion_matrix.shape[0])
        pd.set_option("display.max_rows", self.confusion_matrix.shape[1])
        report_str = str(acc_as_df)

        report_str += '\nAverage accuracy: {:.3f}\n\n'.format(
            np.diag(self.confusion_matrix).sum() / np.maximum(self.confusion_matrix.sum(), 1))

        report_str += 'Confusion matrix (predicted \\ actual):'
        confusion_as_df = pd.DataFrame(
            # не смотря на то что матрица вещественная можно округлить до интов
            # во имя форматирования, ошибка округления не должна быть большой
            self.confusion_matrix.T.astype(int),
            index=self.all_type_names, columns=self.all_type_names)
        report_str += f'\n{confusion_as_df}'

        return report_str

    def get_types_acc(self):
        assert self.confusion_matrix is not None and self.all_type_names is not None, \
            "confusion matrix undefined in report"
        eps = 1e-5
        accs = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + eps)
        type_to_acc = dict((self.all_type_names[i], accs[i]) for i in range(len(accs)))
        return type_to_acc

    def get_average_acc(self):
        assert self.confusion_matrix is not None and self.all_type_names is not None, \
            "confusion matrix undefined in report"
        return np.diag(self.confusion_matrix).sum() / np.maximum(self.confusion_matrix.sum(), 1)

    @staticmethod
    def __sum_up(average_left, count_left, average_right, count_right):
        """
        Вычисляет сумму двух накопленных статистик
        """
        count = count_left + count_right
        accumulator = average_left * count_left + average_right * count_right
        average = accumulator / count if count > 0 else 0
        return average, count


class FtMetricsCalculator:
    """
    Класс для вычисления метрик качества по целевой разметке gt_boxes и
    найденным многоугольниками found_boxes
    """

    def __init__(self, gt_boxes, found_boxes,
                 gt_object_types=None, found_object_types=None, all_object_types=None,
                 compute_classification_metrics=True):
        """
        если хотя бы один из необязательных параметров None, метрики по типам считаться не будут
        :param gt_boxes: список (выпуклых) многоугольников
        :param found_boxes: список (выпуклых) многоугольников
        :param gt_object_types: список типов(классов/языков) объектов, соответствующих gt_boxes
        :param found_object_types: список типов(классов/языков) объектов, соответствующих found_boxes
        :param all_object_types: список (всех) строк-имен типов/классов/языков объектов, которые детектятся
        :param compute_classification_metrics:
        """
        self.__gt_boxes = np.array(gt_boxes)
        self.__found_boxes = np.array(found_boxes)
        
        # считаем метрики для классификации (confusion matrix) только в этом случае
        # здесь чтобы понять, есть ли классификация, достаточно посмотреть на все эти объекты:
        # действительно, если в разметке и предсказаниях указаны типы - значит, модель с классификацией
        self.__are_classification_metrics_computed = \
            gt_object_types is not None \
            and found_object_types is not None \
            and all_object_types is not None \
            and compute_classification_metrics

        self.__gt_object_types = gt_object_types
        self.__found_object_types = found_object_types
        self.__all_object_types = all_object_types
        self.__gt_boxes_count = len(gt_boxes)
        self.__found_boxes_count = len(found_boxes)

        if self.__are_classification_metrics_computed:
            assert len(gt_object_types) == len(gt_boxes) and len(found_object_types) == len(found_boxes), "uneq len"
            assert all(obj_type in all_object_types for obj_type in np.unique(gt_object_types)), "other types"
            assert all(obj_type in all_object_types for obj_type in np.unique(found_object_types)), "other types"
            self.__typename_id = dict((obj_type, i) for i, obj_type in enumerate(self.__all_object_types))

        # вычисляем площади областей
        self.__gt_boxes_areas = [self.__calc_area(box) for box in gt_boxes]
        self.__found_boxes_areas = [self.__calc_area(box) for box in found_boxes]

        # вычисляем площади пересечений
        # intersection_table: первый индекс относится к GT, второй к тестируемому алгоритму
        self.__intersections_table = np.zeros([self.__gt_boxes_count, self.__found_boxes_count])
        self.__iou_table = np.zeros([self.__gt_boxes_count, self.__found_boxes_count])
        for i, gt_box in enumerate(gt_boxes):
            for j, found_box in enumerate(found_boxes):
                try:
                    intersection = self.__calc_intersection_area(gt_box, found_box)
                except Exception as e:
                    logging.error(f"error while __calc_intersection_area in metrics {str(e)}")
                    intersection = 0
                self.__intersections_table[i, j] = intersection
                self.__iou_table[i, j] = self.__calc_iou(
                    self.__gt_boxes_areas[i], self.__found_boxes_areas[j], intersection)

    def analyze(self, iou_threshold):
        """
        Вычисление метрик качества при заданном пороге на IoU
        если IoU двух областей меньше iou_precision_threshold, то считаем, что они не пересекаются
        """

        iou_precision_threshold = 0.05
        gt_to_found = [np.nonzero(self.__iou_table[i] > iou_precision_threshold)[0] for i in
                       range(self.__gt_boxes_count)]
        found_to_gt = [np.nonzero(self.__iou_table[:, j] > iou_precision_threshold)[0] for j in
                       range(self.__found_boxes_count)]

        one_to_ones = []
        one_to_manys = []  # 1 gt <-> много found
        many_to_ones = []  # много gt <-> 1 found
        for gt_index, indices in enumerate(gt_to_found):
            if len(indices) == 1:
                found_area_index = indices[0]
                inverse_indices = found_to_gt[found_area_index]
                if len(inverse_indices) == 1:
                    # соответствие 1 к 1
                    one_to_ones.append([gt_index, found_area_index])
            elif len(indices) > 1:
                # проверим возможность того, что это соответствие 1 <-> много
                if all(len(found_to_gt[index]) == 1 for index in indices):
                    one_to_manys.append([gt_index, indices])

        for found_area_index, inverse_indices in enumerate(found_to_gt):
            if len(inverse_indices) > 1:
                # проверим возможность того, что это соответствие много <-> 1
                if all(len(gt_to_found[index]) == 1 for index in inverse_indices):
                    many_to_ones.append([inverse_indices, found_area_index])

        metrics = FtMetrics(all_type_names=self.__all_object_types,
                            compute_classification_metrics=self.__are_classification_metrics_computed)
        matched_gt = 0
        matched_found = 0
        iou_summed = 0

        # проверим все соответствия 1 к 1 - это либо tp, либо fn (если пересечение слишком мало)
        one_to_ones_iou = [self.__calc_iou(self.__gt_boxes_areas[gt_index], self.__found_boxes_areas[found_index],
                                           self.__intersections_table[gt_index][found_index])
                           for [gt_index, found_index] in one_to_ones]
        match_iou = [(gt_found, iou) for gt_found, iou in zip(one_to_ones, one_to_ones_iou) if iou >= iou_threshold]
        if match_iou:
            one_to_ones, one_to_ones_iou = list(zip(*match_iou))
        else:
            one_to_ones, one_to_ones_iou = [], []

        one_to_ones_count = len(one_to_ones_iou)
        metrics.one_to_one = one_to_ones_count
        matched_gt += one_to_ones_count
        matched_found += one_to_ones_count
        iou_summed += sum(one_to_ones_iou)
        metrics.matched_boxes_count += one_to_ones_count

        if self.__are_classification_metrics_computed:
            # для соответствий 1 к 1
            confusion_matrix = np.zeros((len(self.__all_object_types), len(self.__all_object_types)), dtype=np.float32)
            for gt_index, found_index in one_to_ones:
                self.__update_confusion_matrix([gt_index], [found_index], confusion_matrix)

        # проверим все соответствия 1 <-> много
        for [gt_index, found_indices] in one_to_manys:
            iou = self.__get_group_to_box_iou(self.__gt_boxes[gt_index], self.__found_boxes[found_indices])
            if iou >= iou_threshold:
                matched_gt += 1
                metrics.one_to_many += 1
                matched_found += len(found_indices)
                iou_summed += iou
                metrics.matched_boxes_count += 1
                if self.__are_classification_metrics_computed:
                    self.__update_confusion_matrix([gt_index], found_indices, confusion_matrix)

        # проверим все соответствия много <-> 1
        for [gt_indices, found_index] in many_to_ones:
            iou = self.__get_group_to_box_iou(self.__found_boxes[found_index], self.__gt_boxes[gt_indices])
            if iou >= iou_threshold:
                matched_gt += len(gt_indices)
                metrics.many_to_one += len(gt_indices)
                matched_found += 1
                iou_summed += iou
                metrics.matched_boxes_count += 1
                if self.__are_classification_metrics_computed:
                    self.__update_confusion_matrix(gt_indices, [found_index], confusion_matrix)

        metrics.tp = matched_gt
        metrics.fn = self.__gt_boxes_count - matched_gt
        metrics.fp = self.__found_boxes_count - matched_found

        metrics.average_iou = iou_summed / metrics.matched_boxes_count if metrics.matched_boxes_count > 0 else 0
        metrics.average_precision_by_area, metrics.average_recall_by_area, metrics.average_iou_by_area = \
            self.__calc_iou_by_area()
        metrics.detection_rate = 1 if metrics.average_iou_by_area > iou_threshold else 0
        metrics.matched_images_count = 1

        if self.__are_classification_metrics_computed:
            metrics.confusion_matrix = confusion_matrix
            metrics.all_type_names = self.__all_object_types
        return metrics

    def __update_confusion_matrix(self, gt_indices, found_indices, confusion_matrix):
        """
        Обноляет значения confusion_matrix, сответствующие переданным наборам индексов
        при len(gt_indices) == len(found_indices) == 1 - one-to-one matching
        len(gt_indices) == 1, len(found_indices) != 1 - one-to-many
        len(gt_indices) != 1,  len(found_indices) == 1 -many-to-one
        :param gt_indices: list индексов
        :param found_indices: list индексов
        :param confusion_matrix:
        :return:
        """
        if len(gt_indices) == 1 == len(found_indices):
            # 1 <-> 1
            actual_type = self.__gt_object_types[gt_indices[0]]
            predicted_type = self.__found_object_types[found_indices[0]]
            confusion_matrix[self.__typename_id[actual_type], self.__typename_id[predicted_type]] += 1
        elif len(gt_indices) == 1:
            # 1 <-> many
            gt_index = gt_indices[0]
            found_intersections = self.__intersections_table[gt_index][found_indices]
            relative_weights = found_intersections / np.sum(found_intersections)
            actual_type = self.__gt_object_types[gt_index]
            predicted_types = [self.__found_object_types[found_index] for found_index in found_indices]
            for predicted_type, relative_weight in zip(predicted_types, relative_weights):
                confusion_matrix[self.__typename_id[actual_type], self.__typename_id[predicted_type]] += relative_weight
        elif len(found_indices) == 1:
            # many <-> 1
            actual_types = [self.__gt_object_types[gt_index] for gt_index in gt_indices]
            predicted_type = self.__found_object_types[found_indices[0]]
            for actual_type in actual_types:
                confusion_matrix[self.__typename_id[actual_type], self.__typename_id[predicted_type]] += 1
        else:
            raise ValueError("At least one of gt/predicted indices must be of length 1")

    @staticmethod
    def __union_polygons(boxes):
        """
        Составляет полигон, являющийся объединением всех поданных полигонов
        :param boxes: массив, в котором каждый элемент это список координат многоугольника [x1, y1, x2, y2, ..., xN, yN]
        """
        poly = Polygon()
        for box_to_add in boxes:
            poly_to_add = Polygon(np.reshape(box_to_add, [-1, 2]))
            try:
                poly = poly.union(poly_to_add)
            except Exception as e:
                logging.error(f"error while union polygons in metrics {str(e)}")
        return poly

    def __get_group_to_box_iou(self, box, boxes_group):
        """
        Объединяет многоугольники из группы в один новый полигон, и вычисляет iou
        """
        poly1 = self.__union_polygons(boxes_group)
        poly2 = Polygon(np.reshape(box, [-1, 2]))

        intersection = poly1.intersection(poly2).area
        return self.__calc_iou(poly1.area, poly2.area, intersection)

    def __calc_iou_by_area(self):
        """
        Объединяет все многоугольники на двух изображениях, и вычисляет
        precision, recall и iou двух полученных областей (на всё изображение)
        """
        gt_poly = self.__union_polygons(self.__gt_boxes)
        found_poly = self.__union_polygons(self.__found_boxes)
        try:
            intersection = gt_poly.intersection(found_poly).area
        except Exception as e:
            logging.error(f"error while poligons intersection in metrics {str(e)}")
            intersection = 0
        precision = intersection / found_poly.area if found_poly.area > 0 else 0
        recall = intersection / gt_poly.area if gt_poly.area > 0 else 0
        iou = self.__calc_iou(gt_poly.area, found_poly.area, intersection)
        return [precision, recall, iou]

    @staticmethod
    def __calc_area(box):
        """
        Вычисление площади многоугольника
        """
        poly = Polygon(np.reshape(box, [-1, 2]))
        return poly.area

    @staticmethod
    def __calc_intersection_area(box1, box2):
        """
        Вычисление площади пересечения для двух многоугольников
        """
        poly1 = Polygon(np.reshape(box1, [-1, 2]))
        poly2 = Polygon(np.reshape(box2, [-1, 2]))
        return poly1.intersection(poly2).area

    @staticmethod
    def __calc_iou(area1, area2, intersection):
        """
        Вычисление IoU
        """
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0


class DatasetMetricCalculator:
    """
    Подсчет метрик по всему датасету
    """
    # пороги iou при которых считаются целевые метрики (precision, recall, f1)
    IOU_THRESHOLDS = np.arange(0.4, 1, 0.05)
    # значение порога, при котором ошибки будут отрисовываться
    ERROR_IOU_THRESHOLD = 0.5

    def __init__(self, net_config):
        """
        :param net_config:
        """
        self._net_config = net_config
        self.__reset_accumulators()

    def __reset_accumulators(self):
        self._n_correct_pixels = 0
        self._n_total_pixels = 0
        self._n_objects = 0
        self._curr_mean_object_acc = 0
        self._per_iou_metrics = dict(
            (k, FtMetrics(all_type_names=self._net_config.get_class_names(),
                          compute_classification_metrics=self._net_config.is_classification_supported()))
            for k in DatasetMetricCalculator.IOU_THRESHOLDS)

    def __update_accumulators(self, n_correct_pixels, n_total_pixels, n_objects, mean_object_acc):
        self._n_correct_pixels += n_correct_pixels
        self._n_total_pixels += n_total_pixels
        self._curr_mean_object_acc = \
            (self._curr_mean_object_acc * self._n_objects + mean_object_acc * n_objects) / (self._n_objects + n_objects)
        self._n_objects += n_objects

    def evaluate_batch(self, gt_objects, found_objects, gt_segmap, classification_logits, meta_infos):
        """
        Подсчет метрик для текущего батча с добавлением их к уже накопленным метрикам
        :param gt_objects:
        :param found_objects:
        :param gt_segmap:
        :param classification_logits:
        :param meta_infos:
        :return: selected_threshold_image_metrics, pixel_classification_mask
            selected_threshold_image_metrics - метрики для каждой картинки отдельно
            pixel_classification_mask - маска маску из -1 (неверный ответ),
                0 (на этом месте нет объекта), 1 (верно предсказанный класс)
        """
        selected_threshold_image_metrics = []
        classification_mode = self._net_config.is_classification_supported()
        # считаем целевые метрики
        for image_idx, (image_gt_objects, image_found_objects) in enumerate(zip(gt_objects, found_objects)):
            assert len(image_gt_objects) > 0, "empty gt bboxes (it should contain at least one bbox)"
            image_gt_bboxes, image_gt_box_types = \
                utils.extract_bboxes_and_object_types(image_gt_objects, self._net_config)
            image_pr_bboxes, image_pr_box_types = \
                utils.extract_bboxes_and_object_types(image_found_objects, self._net_config)
            metrics_calculator = FtMetricsCalculator(
                image_gt_bboxes,
                image_pr_bboxes,
                gt_object_types=image_gt_box_types,
                found_object_types=image_pr_box_types,
                all_object_types=self._net_config.get_class_names(),
                compute_classification_metrics=classification_mode
            )
            for iou_threshold in DatasetMetricCalculator.IOU_THRESHOLDS:
                image_metrics = metrics_calculator.analyze(iou_threshold=iou_threshold)
                self._per_iou_metrics[iou_threshold].append(image_metrics)
                if math.isclose(DatasetMetricCalculator.ERROR_IOU_THRESHOLD, iou_threshold):
                    selected_threshold_image_metrics.append(image_metrics)

        pixel_classification_correctness_mask = None
        if self._net_config.is_classification_supported():
            # pixel classification accuracy считается по gt
            pixel_classification_correctness_mask, k_correct_pixels, k_total_pixels, mean_acc, n_objects = \
                DatasetMetricCalculator._calc_pixel_classification_correctness_mask(classification_logits, gt_segmap)
            self.__update_accumulators(k_correct_pixels, k_total_pixels, n_objects, mean_acc)
        return selected_threshold_image_metrics, pixel_classification_correctness_mask

    def get_metrics(self):
        """
        Возвращает посчитанные метрики по всем прошедшим картинкам
        :return: scalar_logs - dict
        """
        scalar_logs = dict()
        for iou_threshold in DatasetMetricCalculator.IOU_THRESHOLDS:
            save_accs = self._net_config.is_classification_supported() and math.isclose(iou_threshold, 0.5)
            DatasetMetricCalculator._fill_log(scalar_logs, self._per_iou_metrics[iou_threshold],
                                              iou_threshold=iou_threshold, save_accs=save_accs)
        scalar_logs["average_iou_by_area"] = \
            self._per_iou_metrics[DatasetMetricCalculator.IOU_THRESHOLDS[0]].average_iou_by_area
        scalar_logs["average_precision_by_area"] = \
            self._per_iou_metrics[DatasetMetricCalculator.IOU_THRESHOLDS[0]].average_precision_by_area
        scalar_logs["average_recall_by_area"] = \
            self._per_iou_metrics[DatasetMetricCalculator.IOU_THRESHOLDS[0]].average_recall_by_area

        if self._net_config.is_classification_supported():
            # количество правильно угаданных пикселей / количество пикселей всего в ground truth объектах
            scalar_logs["classification_pixel_acc_total"] = self._n_correct_pixels / self._n_total_pixels
            # среднее по всем объектам средних точностей классификации на объекте (инвариантно к размеру)
            scalar_logs["classification_pixel_acc_object"] = self._curr_mean_object_acc
        return scalar_logs

    @staticmethod
    def _fill_log(logs_dict, metrics, iou_threshold, save_accs=False):
        pr, r, f1 = metrics.get_metrics()
        logs_dict["pr_iou{:.2f}".format(iou_threshold)] = pr
        logs_dict["recall_iou{:.2f}".format(iou_threshold)] = r
        logs_dict["f1_iou{:.2f}".format(iou_threshold)] = f1
        logs_dict["detection_rate_iou{:.2f}".format(iou_threshold)] = metrics.detection_rate
        if save_accs:
            logs_dict["types_avg_acc_iou{:.2f}".format(iou_threshold)] = metrics.get_average_acc()
            type_to_acc = metrics.get_types_acc()
            for object_type, acc in type_to_acc.items():
                logs_dict[f"acc_{object_type}_iou{iou_threshold:.2f}"] = acc

    @staticmethod
    def _calc_pixel_classification_correctness_mask(classification_logits, targets):
        """
        Возвращает маску из -1 (неверный ответ), 0 (на этом месте нет объекта), 1 (верно предсказанный класс)
        :param classification_logits: логиты (или вероятности) для каждого пикселя для всех классов
        :param targets: метки классов для каждого пикселя на изображении
        :return:
        """
        mask = (targets > 0).astype(np.float)
        labels_true = (targets - 1) * mask
        labels_pred = np.expand_dims(np.argmax(classification_logits, axis=-1), axis=-1)
        correct = np.equal(labels_true, labels_pred).astype(np.float)
        n_correct_pixels = np.sum(correct * mask)
        n_total_pixels = np.sum(mask)

        object_classification_accs = []
        for i in range(len(targets)):
            contours, _ = utils.get_contours_and_boxes(np.squeeze(mask[i].astype(np.uint8), -1), min_area=-1)
            for cnt in contours:
                object_mask = np.zeros(targets.shape[1:3], dtype=np.uint8)
                cv2.drawContours(object_mask, [cnt], -1, 1, -1)
                acc = correct[i][object_mask.astype(np.bool)].mean()
                object_classification_accs.append(acc)
        mean_acc = np.mean(object_classification_accs)
        n_objects = len(object_classification_accs)

        pixel_correct = correct * 2 - 1  # 0, 1 -> -1, 1
        pixel_correct = np.where(mask, pixel_correct, 0)
        pixel_correct = np.squeeze(pixel_correct, axis=-1)
        return pixel_correct, n_correct_pixels, n_total_pixels, mean_acc, n_objects


# TODO нуждается в рефакторинге
class ImageResultCategories:
    """
    Класс отвечающий за подсчета различных категорий, с точки зрения которых, результат на изображении
    может быть в каком-то смысле интересным
    """
    RECALL_ERROR = 'errors/recall'
    PRECISION_ERROR = 'errors/precision'
    DETECTION_RATE_ERROR = 'errors/detection_rate'
    ALL = 'all'

    # что считается важным в данный момент
    SUITABLE_CATEGORIES = (
        RECALL_ERROR,
        DETECTION_RATE_ERROR,
        ALL
    )

    @staticmethod
    def _is_suitable(category):
        return category in ImageResultCategories.SUITABLE_CATEGORIES

    @staticmethod
    def get_categories(metrics):
        """
        Подсчет различных категорий, с точки зрения которых, результат на изображении
        может быть в каком-то смысле интересным
        :param metrics:
        :return:
        """
        categories = []
        if metrics.detection_rate < 1 and ImageResultCategories._is_suitable(
                ImageResultCategories.DETECTION_RATE_ERROR):
            categories.append(ImageResultCategories.DETECTION_RATE_ERROR)
        precision, recall, f1 = metrics.get_metrics()
        if precision < 1 and ImageResultCategories._is_suitable(ImageResultCategories.PRECISION_ERROR):
            categories.append(ImageResultCategories.PRECISION_ERROR)
        if recall < 1 and ImageResultCategories._is_suitable(ImageResultCategories.RECALL_ERROR):
            categories.append(ImageResultCategories.RECALL_ERROR)
        if ImageResultCategories._is_suitable(ImageResultCategories.ALL):
            categories.append(ImageResultCategories.ALL)
        return categories

    @staticmethod
    def get_errors(visualization_categories):
        return [vis_category for vis_category in visualization_categories
                if vis_category != ImageResultCategories.ALL]

    @staticmethod
    def get_folders():
        return ImageResultCategories.SUITABLE_CATEGORIES
