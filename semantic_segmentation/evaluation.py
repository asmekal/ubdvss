#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2018. All rights reserved.
# author K. Gudkov
"""
Классы для вычисления метрик качества для задачи поиска текста
подробное описание метрик можно прочитать здесь
https://tfs/HQ/_git/OCRT?path=%2FImage%2FUtils%2FDt.FindText%2FReadMe.docx&version=GBmaster&_a=contents
"""

import logging

import numpy as np
import pandas as pd
from shapely.geometry import Polygon


class FtMetrics:
    def __init__(self, all_type_names=None):
        """

        :param all_type_names: список всех строк-имен типов/классов/языков объектов, которые детектятся
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

        self.all_type_names = all_type_names
        self.confusion_matrix = None
        if all_type_names is not None:
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
                     ' iou boxes = {:.2f}  >>> by area: pr = {:.4f}, r = {:.4f}, iou = {:.4f}'\
            .format(
                precision, recall, f1,
                self.tp, self.one_to_one, self.one_to_many, self.many_to_one,
                self.fp, self.fn, self.average_iou,
                self.average_precision_by_area, self.average_recall_by_area, self.average_iou_by_area,
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
        вычисляет сумму двух накопленных статистик
        """
        count = count_left + count_right
        accumulator = average_left * count_left + average_right * count_right
        average = accumulator / count if count > 0 else 0
        return average, count


class FtMetricsCalculator:
    """
    класс для вычисления метрик качества по целевой разметке gt_boxes и
    найденным четырехугольникам found_boxes
    """

    def __init__(self, gt_boxes, found_boxes, gt_object_types=None, found_object_types=None, all_object_types=None):
        """
        если хотя бы один из необязательных параметров None, метрики по типам считаться не будут
        :param gt_boxes: список четырехугольников
        :param found_boxes: список четырехугольников
        :param gt_object_types: список типов(классов/языков) объектов, соответствующих gt_boxes
        :param found_object_types: список типов(классов/языков) объектов, соответствующих found_boxes
        :param all_object_types: список (всех) строк-имен типов/классов/языков объектов, которые детектятся
        """
        self.__gt_boxes = np.array(gt_boxes)
        self.__found_boxes = np.array(found_boxes)
        self.__is_object_type_predicted = \
            gt_object_types is not None \
            and found_object_types is not None \
            and all_object_types is not None

        self.__gt_object_types = gt_object_types
        self.__found_object_types = found_object_types
        self.__all_object_types = all_object_types
        self.__gt_boxes_count = len(gt_boxes)
        self.__found_boxes_count = len(found_boxes)

        if self.__is_object_type_predicted:
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
        вычисление метрик качества при заданном пороге на IoU
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

        metrics = FtMetrics(all_type_names=self.__all_object_types)
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

        if self.__is_object_type_predicted:
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
                if self.__is_object_type_predicted:
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
                if self.__is_object_type_predicted:
                    self.__update_confusion_matrix(gt_indices, [found_index], confusion_matrix)

        metrics.tp = matched_gt
        metrics.fn = self.__gt_boxes_count - matched_gt
        metrics.fp = self.__found_boxes_count - matched_found

        metrics.average_iou = iou_summed / metrics.matched_boxes_count if metrics.matched_boxes_count > 0 else 0
        metrics.average_precision_by_area, metrics.average_recall_by_area, metrics.average_iou_by_area = \
            self.__calc_iou_by_area()
        metrics.matched_images_count = 1

        if self.__is_object_type_predicted:
            metrics.confusion_matrix = confusion_matrix
            metrics.all_type_names = self.__all_object_types
        return metrics

    def __update_confusion_matrix(self, gt_indices, found_indices, confusion_matrix):
        """
        обновлить значения confusion_matrix, сответствующие переданным наборам индексов
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
        boxes - массив, в котором каждый элемент это 8 координат x1, y1, x2, y2, x3, y3, x4, y4
        составляет многоугольник, являющийся объединением четырехугольников
        """
        poly = Polygon()
        for box_to_add in boxes:
            poly_to_add = Polygon(np.reshape(box_to_add, [4, 2]))
            try:
                poly = poly.union(poly_to_add)
            except Exception as e:
                logging.error(f"error while union polygons in metrics {str(e)}")
        return poly

    def __get_group_to_box_iou(self, box, boxes_group):
        """
        объединяет четрыехугольники из группы в один новый прямоугольник, и вычисляет iou
        """
        poly1 = self.__union_polygons(boxes_group)
        poly2 = Polygon(np.reshape(box, [4, 2]))

        intersection = poly1.intersection(poly2).area
        return self.__calc_iou(poly1.area, poly2.area, intersection)

    def __calc_iou_by_area(self):
        """
        объединяет все четырехуголтники на двух изображениях, и вычисляет
        precision, recall и iou двух полученных областей
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
        вычисление площади четырехугольника
        """
        poly = Polygon(np.reshape(box, [4, 2]))
        return poly.area

    @staticmethod
    def __calc_intersection_area(box1, box2):
        """
        вычисление площади пересечения для двух четырехугольников
        """
        poly1 = Polygon(np.reshape(box1, [4, 2]))
        poly2 = Polygon(np.reshape(box2, [4, 2]))
        return poly1.intersection(poly2).area

    @staticmethod
    def __calc_iou(area1, area2, intersection):
        """
        вычисление IoU
        """
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
