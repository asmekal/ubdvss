#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2018. All rights reserved.
"""
Все что связано с callbacks для обучения и валидации
"""
import copy
import logging
import os
import time
import math

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from semantic_segmentation import utils
from semantic_segmentation.data_markup import ClassifiedObjectMarkup
from semantic_segmentation.evaluation import FtMetricsCalculator, FtMetrics
from semantic_segmentation.segmap_manager import SegmapManager


class EvaluationCallback(callbacks.TensorBoard):
    """
    Служит для подсчета целевых метрик и визуализации картинок в тензорборде при обучении
    """
    # пороги iou при которых считаются целевые метрики (precision, recall, f1)
    IOU_THRESHOLDS = np.arange(0.4, 1, 0.05)

    def __init__(self, log_dir, net_config, training_generator, validation_generator, max_evaluated_images=-1):
        """
        :param log_dir: путь до папки с логами (тензорборда)
        :param training_generator: объект типа BatchGenerator с обучающей выборкой
        :param validation_generator: объект типа BatchGenerator с валидационной выборкой
        :param max_evaluated_images: сколько максимум картинок использовать для оценки,
            если -1 все что есть в выборке, иначе min(картинок в выборке, max_evaluated_images)
        """
        super().__init__(log_dir=log_dir)
        self.__net_config = net_config
        self.__training_data_generator = training_generator.generate(add_metainfo=True)
        self.__validation_data_generator = validation_generator.generate(add_metainfo=True)
        self.__evaluated_images_for_train = training_generator.get_images_per_epoch()
        self.__evaluated_images_for_validation = validation_generator.get_images_per_epoch()
        if max_evaluated_images >= 0:
            self.__evaluated_images_for_train = min(self.__evaluated_images_for_train, max_evaluated_images)
            self.__evaluated_images_for_validation = min(self.__evaluated_images_for_validation, max_evaluated_images)
        self.__epochs_count = 0

    def on_epoch_end(self, epoch, logs={}):
        self.__epochs_count += 1
        scalar_logs = {}
        image_logs = {}

        self._evaluate_and_fill_logs(scalar_logs, image_logs, training_phase='train')
        self._evaluate_and_fill_logs(scalar_logs, image_logs, training_phase='valid')

        if epoch < 1:
            for name, value in scalar_logs.items():
                tf.summary.scalar(name, value)
            for name, value in image_logs.items():
                images_placeholder = K.placeholder(shape=(None, None, None, 3), dtype=None, name=name)
                tf.summary.image(name, images_placeholder, max_outputs=10)

            summary_str = tf.summary.merge_all(
                key=tf.GraphKeys.SUMMARIES,
                scope=None
            )
            feed_dict = dict(("{}:0".format(key), value) for key, value in image_logs.items())
            self.writer.add_summary(self.sess.run([summary_str], feed_dict=feed_dict)[0], epoch)
        else:
            for name, value in scalar_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.writer.add_summary(summary, epoch)
            summary_str = tf.summary.merge_all(
                key=tf.GraphKeys.SUMMARIES,
                scope=None
            )
            feed_dict = dict(("{}:0".format(key), value) for key, value in image_logs.items())
            self.writer.add_summary(self.sess.run([summary_str], feed_dict=feed_dict)[0], epoch)
        self.writer.flush()
        super().on_epoch_end(epoch, logs)

    @staticmethod
    def evaluate(model, data_generator, net_config, n_images, pixel_threshold=0.5, save_dir=None,
                 save_visualizations=False, log_time=False):
        """
        Подсчет метрик по различным порогам iou (precision, recall, f1) и отрисовка картинок с разметкой,
        предсказанными картами сегментаций и финальными детекциями после пост-процессинга
        :param model: оцениваемая модель
        :param data_generator: генератор батчей
        :param n_images: сколько изображений использоват для подсчета метрик
        :param net_config: конфигурация сети
        :param pixel_threshold: pixel_probability > pixel_threshold класс считается положительным
        :param save_dir: путь куда сохранять результаты, если None - никуда
        :param save_visualizations: сохранять ли визуализации
        :param log_time: считать ли максимальное время работы на изображениях
        :return: (scalar_logs, visualizations)
        """

        images_processed = 0
        scalar_logs = dict()
        visualizations = dict()
        per_iou_metrics = dict(
            (k, FtMetrics(all_type_names=net_config.get_class_names())) for k in EvaluationCallback.IOU_THRESHOLDS)
        depreprocessing_fn = net_config.get_depreprocessing_fn()

        eps = 1e-9
        logit_threshold = - np.log(1 / np.clip(pixel_threshold, eps, 1 - eps) - 1)

        if save_dir:
            save_gt_dir = os.path.join(save_dir, 'markup')
            save_predictions_dir = os.path.join(save_dir, 'predictions')
            os.makedirs(save_gt_dir, exist_ok=True)
            os.makedirs(save_predictions_dir, exist_ok=True)
            if save_visualizations:
                save_images_dir = os.path.join(save_dir, 'images')
                os.makedirs(save_images_dir, exist_ok=True)

        t_allcores_max = 0
        t_system_max = 0
        n_correct_pixels = 0
        n_total_pixels = 0
        model._make_predict_function()
        while images_processed < n_images:
            # не logging потому что там сложно с тем чтобы не переносить строчку
            # а так не замусоривается лог
            print("\rimages processed: {}/{}".format(images_processed, n_images), end='', flush=True)

            images, targets, meta_infos = next(data_generator)
            t_system = time.time()
            t_allcores = time.clock()
            predicted_targets = model.predict(images)
            t_allcores = time.clock() - t_allcores
            t_system = time.time() - t_system
            detection_logits = predicted_targets[..., :1]
            classification_logits = predicted_targets[..., 1:]

            # второе условие нужно чтобы не учитывать самый первый батч, который почему-то компилит predict_function
            # по идее это должно решаться предварительным вызовом model._make_predict_function()
            # но почему-то не решается, не знаю, если кто-то разбирается, подскажите
            if log_time and images_processed:
                logging.debug(f"shape={images.shape}, time/image={t_system/images.shape[0]}")
                t_allcores_max = max(t_allcores_max, t_allcores / images.shape[0])
                t_system_max = max(t_system_max, t_system / images.shape[0])

            detection_logits = np.where(detection_logits > logit_threshold, 1, 0)

            gt_markup = [meta_info.markup for meta_info in meta_infos]
            # эти отрисовываются (чтобы не перемасштабировать картинки)
            predicted_markup = [
                SegmapManager.postprocess(
                    detection_logits[i],
                    classification_logits[i] if net_config.is_classification_supported() else None,
                    scale=net_config.get_scale(),
                    min_block_threshold=net_config.get_min_pixels_for_detection()
                ) for i in range(predicted_targets.shape[0])
            ]
            # а на этих уже смотрим целевые метрики
            predicted_rescaled_markup = copy.deepcopy(predicted_markup)
            for i in range(len(images)):
                for j in range(len(predicted_rescaled_markup[i])):
                    predicted_rescaled_markup[i][j].bbox = utils.rescale_bbox(
                        predicted_rescaled_markup[i][j].bbox,
                        meta_infos[i].xscale,
                        meta_infos[i].yscale
                    )

            # считаем целевые метрики
            for iou_threshold in EvaluationCallback.IOU_THRESHOLDS:
                for image_idx, (image_gt_markup, image_pr_markup) in enumerate(zip(gt_markup,
                                                                                   predicted_rescaled_markup)):
                    try:
                        assert len(image_gt_markup) > 0, "empty gt bboxes (it should contain at least one bbox)"
                        classification_mode = net_config.is_classification_supported()
                        image_gt_bboxes, image_gt_box_types = \
                            utils.extract_bboxes_and_object_types(image_gt_markup, net_config)
                        image_pr_bboxes, image_pr_box_types = \
                            utils.extract_bboxes_and_object_types(image_pr_markup, net_config)
                        metrics_calculator = FtMetricsCalculator(
                            image_gt_bboxes,
                            image_pr_bboxes,
                            gt_object_types=image_gt_box_types,
                            found_object_types=image_pr_box_types,
                            all_object_types=net_config.get_class_names() if classification_mode else None
                        )
                        image_metrics = metrics_calculator.analyze(iou_threshold=iou_threshold)
                        per_iou_metrics[iou_threshold].append(image_metrics)
                    except AssertionError as e:
                        logging.error("assertion error: {}".format(e))
                    except ZeroDivisionError as e:
                        logging.error("error {} in metric calculation: gt={}, pr={}".format(e,
                                                                                            image_gt_markup,
                                                                                            image_pr_markup))

            if net_config.is_classification_supported():
                ###
                # pixel classification accuracy on gt
                pixel_classification_correctness_mask, k_correct_pixels, k_total_pixels = \
                    EvaluationCallback._calc_pixel_classification_correctness_mask(classification_logits, targets)

                n_correct_pixels += k_correct_pixels
                n_total_pixels += k_total_pixels
                ###

            if images_processed == 0 or len(visualizations["gt"]) <= len(images) or save_dir:
                # не строить визуализации если мы не хотим все сохранять
                # или у нас уже есть достаточно картинок для логов
                # рандом добавляется чтобы не каждый раз одни и те же картинки рисовались
                if not save_dir \
                        and images_processed > 0 \
                        and len(visualizations["gt"]) == len(images) \
                        and np.random.randint(5) != 0:
                    continue

                # чтобы отрисовывать картинки безо всякой нормализации
                unnormalized_images = depreprocessing_fn(images).astype(np.uint8)

                visualizations["gt"] = EvaluationCallback._visualize_segmentation_maps(unnormalized_images,
                                                                                       targets[..., :1])
                visualizations["seg_map"] = EvaluationCallback._visualize_segmentation_maps(unnormalized_images,
                                                                                            detection_logits)
                visualizations["postprocessed"] = EvaluationCallback._draw_bboxes(unnormalized_images,
                                                                                  predicted_markup)
                if net_config.is_classification_supported():
                    visualizations["classification_gt"] = \
                        EvaluationCallback._visualize_classification_masks(unnormalized_images,
                                                                           pixel_classification_correctness_mask)
                if save_dir:
                    for i, meta_info in enumerate(meta_infos):
                        EvaluationCallback._save_markup_csv(
                            filename=os.path.join(save_gt_dir, meta_info.filename + '.txt'),
                            markups=gt_markup[i]
                        )
                        EvaluationCallback._save_markup_csv(
                            filename=os.path.join(save_predictions_dir, meta_info.filename + '.txt'),
                            markups=predicted_rescaled_markup[i]
                        )
                        if save_visualizations:
                            for tag in visualizations:
                                save_fname = os.path.join(save_images_dir, f"{meta_info.filename}.{tag}.png")
                                cv2.imwrite(save_fname, cv2.cvtColor(visualizations[tag][i], cv2.COLOR_RGB2BGR))

            images_processed += len(images)
        # новая строка после images_processed:k/n
        print("\rimages processed: {}/{}\n".format(images_processed, n_images), flush=True)

        for iou_threshold in EvaluationCallback.IOU_THRESHOLDS:
            save_accs = net_config.is_classification_supported() and math.isclose(iou_threshold, 0.5)
            EvaluationCallback._fill_log(scalar_logs, per_iou_metrics[iou_threshold],
                                         iou_threshold=iou_threshold, save_accs=save_accs)
            if save_accs:
                logging.info("For iou threshold {:.2f}:\n".format(iou_threshold))
                logging.info(per_iou_metrics[iou_threshold].get_report())
                logging.info(per_iou_metrics[iou_threshold].get_confusion_matrix_report())
        scalar_logs["average_iou_by_area"] = \
            per_iou_metrics[EvaluationCallback.IOU_THRESHOLDS[0]].average_iou_by_area
        scalar_logs["average_precision_by_area"] = \
            per_iou_metrics[EvaluationCallback.IOU_THRESHOLDS[0]].average_precision_by_area
        scalar_logs["average_recall_by_area"] = \
            per_iou_metrics[EvaluationCallback.IOU_THRESHOLDS[0]].average_recall_by_area

        if net_config.is_classification_supported():
            # честная точность (нормаровано на число пикселей)
            scalar_logs["classification_pixel_accuracy"] = n_correct_pixels / n_total_pixels
        if log_time:
            scalar_logs["max_system_time_per_image"] = t_system_max
            scalar_logs["max_allcores_time_per_image"] = t_allcores_max
        return scalar_logs, visualizations

    def _evaluate_and_fill_logs(self, scalar_logs, image_logs, training_phase='train'):
        """
        запустить модель на данных из training_phase, замерить метрики, получить изображения
        :param scalar_logs: dict куда сохранить метрики
        :param image_logs: dict куда сохранить визуализации
        :param training_phase: 'train' or 'valid'
        :return:
        """
        assert training_phase in ['train', 'valid']
        if training_phase == 'train':
            batch_generator = self.__training_data_generator
            n_images = self.__evaluated_images_for_train
        else:
            batch_generator = self.__validation_data_generator
            n_images = self.__evaluated_images_for_validation

        logging.info(f"evaluating performance on {training_phase} ({n_images} images)...")
        evaluation_metrics, visualizations = self.evaluate(self.model, batch_generator,
                                                           net_config=self.__net_config,
                                                           n_images=n_images)
        for key, value in evaluation_metrics.items():
            scalar_logs[f"{training_phase}_{key}"] = value
        for key, value in visualizations.items():
            image_logs[f"{training_phase}_{key}"] = value

    @staticmethod
    def _fill_log(logs_dict, metrics, iou_threshold, save_accs=False):
        pr, r, f1 = metrics.get_metrics()
        logs_dict["pr_iou{:.2f}".format(iou_threshold)] = pr
        logs_dict["recall_iou{:.2f}".format(iou_threshold)] = r
        logs_dict["f1_iou{:.2f}".format(iou_threshold)] = f1
        if save_accs:
            logs_dict["types_avg_acc_iou{:.2f}".format(iou_threshold)] = metrics.get_average_acc()
            type_to_acc = metrics.get_types_acc()
            for object_type, acc in type_to_acc.items():
                logs_dict[f"acc_{object_type}_iou{iou_threshold:.2f}"] = acc

    @staticmethod
    def _visualize_segmentation_maps(images, targets, threshold=0.5):
        """
        возвращает список изображений с отрисованными картами сегментаций
        :param images: список изображений (np.ndarray)
        :param targets: список соответствующих изображениям карт сегментации (должен быть той же длины что и images)
        :param threshold: порог по которому разделяются положительный и отрицательный классы
        :return:
        """
        assert len(images) == len(targets)
        return [
            EvaluationCallback._visualize_segmentation_map(
                image,
                target,
                threshold=threshold
            )
            for image, target in zip(images, targets)
        ]

    @staticmethod
    def _visualize_classification_masks(images, targets):
        assert len(images) == len(targets)
        return [
            EvaluationCallback._visualize_classification_mask(
                image,
                target
            )
            for image, target in zip(images, targets)
        ]

    @staticmethod
    def _visualize_classification_mask(image, is_pixel_correct):
        pillow_image = utils.pillow_rgb_fromarray(image)
        target_true = np.where(is_pixel_correct == 1, 255, 0).astype(np.uint8)
        target_false = np.where(is_pixel_correct == -1, 255, 0).astype(np.uint8)

        visualized_markup = pillow_image
        for target, color in zip((target_true, target_false), ((0, 255, 0), (255, 0, 0))):
            if target.ndim == 3 and target.shape[2] == 1:
                target = target.squeeze(axis=2)
            target = Image.fromarray(target, 'L')
            if target.size != pillow_image.size:
                target = target.resize(pillow_image.size, resample=Image.NEAREST)
            visualized_markup = SegmapManager.draw_segmentation_map(visualized_markup, target, color=color)
        return np.array(visualized_markup)

    @staticmethod
    def _draw_bboxes(images, image_markups):
        """
        возвращает список изображений с отрисованной разметкой
        :param images: список изображений
        :param image_markups: соответствующие им разметки
        :return:
        """
        assert len(images) == len(image_markups)
        return [
            np.array(SegmapManager.draw_markup(utils.pillow_rgb_fromarray(image), markup))
            for image, markup in zip(images, image_markups)
        ]

    @staticmethod
    def _visualize_segmentation_map(image, target, threshold=0.5):
        """
        накладывает карту сегментации на изображение как маску, то что больше treshold красится в зеленый
        :param image: np.ndarray RGB картинка
        :param target: np.ndarray массив вероятностей (той же размерности что и image но с одним каналом)
        :param threshold: порог по которому разделяются положительный и отрицательный классы
        :return:
        """
        pillow_image = utils.pillow_rgb_fromarray(image)
        target_as_image = np.where(target > threshold, 255, 0).astype(np.uint8)
        if target_as_image.ndim == 3 and target_as_image.shape[2] == 1:
            target_as_image = target_as_image.squeeze(axis=2)
        target_as_image = Image.fromarray(target_as_image, 'L')
        if target_as_image.size != pillow_image.size:
            target_as_image = target_as_image.resize(pillow_image.size, resample=Image.NEAREST)

        visualized_markup = SegmapManager.draw_segmentation_map(pillow_image, target_as_image)
        return np.array(visualized_markup)

    @staticmethod
    def _save_markup_csv(filename, markups):
        """
        запись в CSV массива четырехугольников
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

    @staticmethod
    def _calc_pixel_classification_correctness_mask(classification_logits, targets):
        """
        возвращает маску из -1 (неверный ответ), 0 (на этом месте нет объекта), 1 (верно предсказанный класс)
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

        pixel_correct = correct * 2 - 1  # 0, 1 -> -1, 1
        pixel_correct = np.where(mask, pixel_correct, 0)
        pixel_correct = np.squeeze(pixel_correct, axis=-1)
        return pixel_correct, n_correct_pixels, n_total_pixels


def build_callbacks_list(log_dir, net_config, training_generator, validation_generator, max_evaluated_images=-1):
    backup_dir = os.path.join(log_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(backup_dir, "model_{epoch:03d}.h5"))

    last_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"))
    best_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "model_best.h5"), save_best_only=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, min_lr=1e-5)

    eval_callback = EvaluationCallback(
        log_dir, net_config, training_generator, validation_generator, max_evaluated_images=max_evaluated_images)

    return [
        eval_callback,
        last_checkpoint_callback,
        best_checkpoint_callback,
        backup_checkpoint_callback,
        reduce_lr_callback
    ]
