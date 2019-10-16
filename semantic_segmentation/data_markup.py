#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Классы для хранения разметки одного объекта на изображении
"""


class ObjectMarkup:
    """
    Класс для храниния информации об одном объекте на изображении
    Содержит координаты многоугольника (контура вокруг объекта)
    """
    __slots__ = ['bbox']

    def __init__(self, bbox):
        self.bbox = bbox

    def create_same_markup(self, new_bbox):
        return ObjectMarkup(new_bbox)


class ClassifiedObjectMarkup(ObjectMarkup):
    """
    Класс для храниния информации об одном объекте на изображении
    Содержит координаты многоугольника (контура вокруг объекта) и тип объекта (представляемый целым числом)
    """
    __slots__ = ['object_type']

    def __init__(self, bbox, object_type):
        super().__init__(bbox)
        # int() чтобы не было всяких коварних эффектов, например, при отрисовке с np.int64 не прокатывает
        self.object_type = int(object_type)

    def create_same_markup(self, new_bbox):
        return ClassifiedObjectMarkup(new_bbox, self.object_type)
