#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2017. All rights reserved.
"""
Класс для аугментации данных (кропы и повороты изображений и разметки)
Основа заимствована из проекта FindText
Зафиксировав seeds в этом файле (раскомментировав соответствующие строчки), можно добиться
воспроизводимости порядка батчей и аугментации при обучении
"""

import logging
import random

import PIL
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from imgaug import seed as imgaug_seed
from shapely import affinity
from shapely.geometry import Point, Polygon

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# раскомментировав установки сидов ниже можно получить воспроизводимую аугментацию и порядок батчей и картинок в них
# НО! если запускать многопроцессорно этот файл каждый раз будет переимпортится и сиды заново устанавливаться
# так что мы получим фиксированный (порядка prepare_batch_size // n_processes) набор трансформаций
# что все же не очень хорошо
# random.seed(42)
# np.random.seed(42)
# imgaug_seed(42)


class SegLinksImageAugmentation:
    def __init__(self, image, markup, net_config):
        """
        Класс для аугментации данных кропами и поворотами. Каждый элемент списка разметки имеет формат
        x1, y1, x2, y2, ..., xN, yN
        """
        self.__aug_image, self.__aug_markup = (image, markup)
        self.__net_config = net_config
        if len(markup) == 0:
            # если разметки нет, то не будем делать никаких модификаций изображения
            return
        try:
            self.__aug_image, self.__aug_markup = self.__augment_image(image, markup)
        except Exception as e:
            logging.error(str(e) + " error while augmenting")

    def __augment_image(self, image, markup):
        feed_original_probability = 0.1
        crop_probability = 0.5
        rotate_probability = 0.5
        rotate_90_probability = 0.5
        # углы, на которые разрешено поворачивать изображения при rotate_90
        rotation_90_angles = [90, -90, 180]
        # формально флипы нельзя баркодам (по крайней мере не всем),
        # но для детекции на самом деле не такая большая разница
        # TODO: можно включить
        horizontal_flip_probability = 0
        vertical_flip_probability = 0

        perspective_distortion_probability = 0.5
        image_aug_probability = 0.7

        self.__aug_image = image.copy()
        self.__aug_markup = markup.copy()
        if random.random() < feed_original_probability:
            return self.__aug_image, self.__aug_markup
        if random.random() < rotate_probability:
            self.__aug_image, self.__aug_markup = self.__rotate(self.__aug_image, self.__aug_markup, -45, 45)
        if random.random() < crop_probability:
            self.__aug_image, self.__aug_markup = self.__crop(self.__aug_image, self.__aug_markup)
        if random.random() < horizontal_flip_probability:
            self.__aug_image, self.__aug_markup = self.__horizontal_flip(self.__aug_image, self.__aug_markup)
        if random.random() < vertical_flip_probability:
            self.__aug_image, self.__aug_markup = self.__vertical_flip(self.__aug_image, self.__aug_markup)
        if random.random() < rotate_90_probability:
            angle = random.choice(rotation_90_angles)
            self.__aug_image, self.__aug_markup = self.__rotate(self.__aug_image, self.__aug_markup, angle, angle)
        if random.random() < perspective_distortion_probability:
            self.__aug_image, self.__aug_markup = self.__perspective_distortion(self.__aug_image, self.__aug_markup)
        if random.random() < image_aug_probability:
            self.__aug_image, self.__aug_markup = self.__image_augmentation(self.__aug_image, self.__aug_markup)
        return self.__aug_image, self.__aug_markup

    def get_modified_image(self):
        return self.__aug_image

    def get_modified_markup(self):
        return self.__aug_markup

    @staticmethod
    def __crop(image, markup):
        # определяем границы разметки
        all_poly = Polygon()
        for box in markup:
            word_poly = Polygon(np.reshape(box.bbox, [-1, 2]))
            all_poly = all_poly.union(word_poly)

        all_markup_rect = list(all_poly.bounds)
        # если прямоугольник получается с плохими пропорциями, то немного расширим его по одному из направлений
        all_markup_rect = SegLinksImageAugmentation.__normalize_rect(all_markup_rect, image.size)

        # определяем границы вырезаемой области
        max_margin = 0.4  # максимальный размер отрезаемых полей в единицах размера изображения, < 0.5
        max_left = min(max_margin * image.size[0], all_markup_rect[0])
        max_right = min(max_margin * image.size[0], image.size[0] - all_markup_rect[2])
        max_top = min(max_margin * image.size[1], all_markup_rect[1])
        max_bottom = min(max_margin * image.size[1], image.size[1] - all_markup_rect[3])

        left = random.uniform(0, max_left)
        top = random.uniform(0, max_top)
        right = image.size[0] - random.uniform(0, max_right)
        bottom = image.size[1] - random.uniform(0, max_bottom)

        cropped_image = image.crop((left, top, right, bottom))
        cropped_markup = []
        for box in markup:
            box.bbox = SegLinksImageAugmentation.__shift_box(box.bbox, -left, -top)
            cropped_markup.append(box)

        return cropped_image, cropped_markup

    @staticmethod
    def __normalize_rect(rect, image_size):
        """
        Проверяет, что пропорции прямоугольника не слишком сильно отличаются от пропорций изображения,
        и при необходимости увеличивает прямоугольник вдоль одной из сторон
        """
        new_rect = rect
        new_rect[0] = max(0, new_rect[0])
        new_rect[1] = max(0, new_rect[1])
        new_rect[2] = min(image_size[0], new_rect[2])
        new_rect[3] = min(image_size[1], new_rect[3])

        rect_width = new_rect[2] - new_rect[0]
        rect_height = new_rect[3] - new_rect[1]

        ratio = rect_width / rect_height
        image_ratio = image_size[0] / image_size[1]

        if ratio < image_ratio * 0.7:
            new_width = image_ratio * rect_height
            if new_rect[0] < (image_size[0] - new_width) / 2:
                new_rect[2] = new_rect[0] + new_width
            elif new_rect[2] > (image_size[0] + new_width) / 2:
                new_rect[0] = new_rect[2] - new_width
            else:
                new_rect[0] = (image_size[0] - new_width) / 2
                new_rect[2] = new_rect[0] + new_width
        elif ratio > image_ratio * 1.3:
            new_height = rect_width / image_ratio
            if new_rect[1] < (image_size[1] - new_height) / 2:
                new_rect[3] = new_rect[1] + new_height
            elif new_rect[3] > (image_size[1] + new_height) / 2:
                new_rect[1] = new_rect[3] - new_height
            else:
                new_rect[1] = (image_size[1] - new_height) / 2
                new_rect[3] = new_rect[1] + new_height

        return new_rect

    @staticmethod
    def __rotate(image, markup, min_angle, max_angle):
        angle = random.uniform(min_angle, max_angle)
        rotated_image = image.rotate(angle, PIL.Image.BILINEAR, expand=True)

        # меняем знак угла, т.к. у изображений координата y направлена сверху вниз,
        # и угол поворота в координатах имеет другой смысл, чем для изображения
        rotated_markup = []
        for box in markup:
            box.bbox = SegLinksImageAugmentation.__rotate_box(box.bbox, -angle, image.size, rotated_image.size)
            rotated_markup.append(box)

        return rotated_image, rotated_markup

    @staticmethod
    def __perspective_distortion(image, markup):
        """
        Случайное переспективное преобразование изображения и разметки
        """

        # половина разброса случайных чисел
        random_half_size = np.array([
            0.1, 0.1, 50,
            0.1, 0.1, 50,
            0.0002, 0.0002
        ])

        # средние значения случайных чисел
        random_mean = np.array([
            1, 0, 0,
            0, 1, 0,
            0, 0
        ])

        # генерируем коэффициенты трансформации
        coeffs = np.random.uniform(random_mean - random_half_size, random_mean + random_half_size).tolist()

        rotated_image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)

        # дополеним до матрицы 3*3
        coeffs.append(1)
        coeffs = np.reshape(coeffs, (3, 3))

        # разработчики PIL почему-то решили делать обратное преобразование
        coeffs = np.linalg.inv(coeffs)

        # преобразуем разметку с помощью мтарицы преобразования
        rotated_markup = []
        for box in markup:
            box.bbox = SegLinksImageAugmentation.__transform_box(box.bbox, coeffs)
            rotated_markup.append(box)

        return rotated_image, rotated_markup

    @staticmethod
    def __transform_box(box, transform_matrix):
        """
        Прербразование многоугольники с помощью матрицы преобразования 3*3
        """
        res = np.array(box).reshape((-1, 2)).transpose()
        res = np.vstack((res, np.ones((1, res.shape[1]))))
        res = np.dot(transform_matrix, res)
        res /= res[-1, :]
        return res[:-1].transpose().flatten()

    @staticmethod
    def __horizontal_flip(image, markup):
        image = Image.fromarray(np.fliplr(image))
        for i, box in enumerate(markup):
            for j in range(len(box) // 2):
                box.bbox[j * 2] = image.width - box.bbox[j * 2]
            markup[i] = box
        return image, markup

    @staticmethod
    def __vertical_flip(image, markup):
        image = Image.fromarray(np.flipud(image))
        for i, box in enumerate(markup):
            for j in range(len(box) // 2):
                box.bbox[j * 2 + 1] = image.height - box.bbox[j * 2 + 1]
            markup[i] = box
        return image, markup

    @staticmethod
    def __rotate_box(box, angle, image_size, rotated_image_size):
        """
        Поворот многоугольника
        """
        rotated_box = []
        for i in range(len(box) // 2):
            pt = Point(box[2 * i], box[2 * i + 1])
            pt = affinity.translate(pt, -image_size[0] / 2, -image_size[1] / 2)
            rotated_pt = affinity.rotate(pt, angle, origin=[0, 0], use_radians=False)
            rotated_pt = affinity.translate(rotated_pt, rotated_image_size[0] / 2, rotated_image_size[1] / 2)
            rotated_box.extend([rotated_pt.x, rotated_pt.y])

        return rotated_box

    @staticmethod
    def __shift_box(box, dx, dy):
        """
        Сдвиг всех координат многоугольника
        """
        shifted_box = []
        for i in range(len(box) // 2):
            pt = Point(box[2 * i], box[2 * i + 1])
            pt = affinity.translate(pt, dx, dy)
            shifted_box.extend([pt.x, pt.y])

        return shifted_box

    @staticmethod
    def __image_augmentation(image, markup):
        """
        Искажения изображения без разметки
        """
        seq = iaa.SomeOf((0, 5),
                         [
                             # (может очень сильно испортить текст на изображении, поэтому лучше не использовать)
                             # была гипотеза что включением следующей строчки увеличивается recall ценой precision
                             # но НЕТ - на последнем тесте это не подтвердилось, впрочем, окончательно сказать нельзя
                             # так что строчка оставлена, если recall низкий - можно попробовать раскомментить
                             # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                             # # convert images into their superpixel representation
                             # # p_replace-вероятность объединения соседних superpixel
                             # # n_segments-количество superpixel
                             iaa.OneOf([
                                 iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                 iaa.AverageBlur(k=(2, 7)),
                                 # blur image using local means with kernel sizes between 2 and 7
                                 iaa.MedianBlur(k=(3, 11)),
                                 # blur image using local medians with kernel sizes between 2 and 7
                             ]),
                             iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                             iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                             # search either for all edges or for directed edges,
                             # blend the result with the original image using a blobby mask
                             iaa.SimplexNoiseAlpha(iaa.OneOf([
                                 iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                 iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                             ])),
                             # добавляет черные капли на картинку
                             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                             # add gaussian noise to images
                             iaa.OneOf([
                                 iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                             ]),
                             iaa.Invert(0.05, per_channel=True),  # invert color channels
                             iaa.Add((-10, 10), per_channel=0.5),
                             # change brightness of images (by -10 to 10 of original value)
                             iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                             # either change the brightness of the whole image (sometimes
                             # per channel) or change the brightness of subareas
                             iaa.OneOf([
                                 iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                 iaa.FrequencyNoiseAlpha(
                                     exponent=(-4, 0),
                                     first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                     second=iaa.ContrastNormalization((0.5, 2.0))
                                 )
                             ]),
                             iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                             iaa.Grayscale(alpha=(0.0, 1.0)),
                             sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
                         ],
                         random_order=True
                         )
        images_aug = Image.fromarray(seq.augment_image(np.array(image, dtype=np.uint8)))
        return images_aug, markup
