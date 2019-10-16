#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2018. All rights reserved.
"""
Скрипт для обучения сети
"""
import argparse
import logging
import os
import time

from keras.optimizers import Adam

from semantic_segmentation import losses
from semantic_segmentation.data_generators import BatchGenerator
from semantic_segmentation.keras_callbacks import build_callbacks_list
from semantic_segmentation.keras_metrics import get_all_metrics
from semantic_segmentation.net import NetConfig, NetManager, supported_preprocessing_types

argparser = argparse.ArgumentParser()
argparser.add_argument('--train_markup_path', '-t', type=str,
                       help="path to training markup dir (contains Image and Markup subfolders)")
argparser.add_argument('--valid_markup_path', '-v', type=str,
                       help="path to validation markup dir (contains Image and Markup subfolders)")
argparser.add_argument('--markup_type', '-mt', type=str, default="Barcode",
                       help="markup type for train and test")
argparser.add_argument('--log_dir', '-l', type=str, default=os.path.join('logs', time.strftime('%Y-%m-%d_%H.%M.%S')),
                       help="path to logging dir with tensorboard events, backup models, etc")
argparser.add_argument('--batch_size', '-b', type=int, default=8,
                       help="batch size for train, test and evaluation")
argparser.add_argument('--epochs', '-e', type=int, default=50,
                       help="number of training epochs")
argparser.add_argument('-lr', type=float, default=1e-3,
                       help="initial learning rate")
argparser.add_argument('--n_workers', '-n', type=int, default=4,
                       help="number of preprocessing threads")
argparser.add_argument('--prepare_batch_size', '-pbs', type=int, default=3000,
                       help="number of preprocessed images before groupby")
argparser.add_argument('--max_evaluated_images', type=int, default=200,
                       help="number of images used in evaluation callback")
argparser.add_argument('-cfm', default=None,
                       help="path to model from which to initialize weights")
argparser.add_argument('--custom_description', '-d',
                       help="custom desctiption of the model and how it will be trained")
argparser.add_argument('--preprocessing', default='mobilenet_like',
                       help=f"preprocessing type, one of {supported_preprocessing_types.keys()}")
argparser.add_argument('--fml_incompatible', action='store_true',
                       help="model will NOT be FML-compatible, "
                            "but implementation will be simpler (no additional paddings)")
argparser.add_argument('--max_image_side', type=int, default=512,
                       help="max size for image height and width "
                            "(if it is larger image will be downsized maintaining aspect ratio)")
argparser.add_argument('--object_types_fname', default=None,
                       help="path to file containing types (separated by end of line); "
                            "if not stated - no type classification")


def save_desctiption(args):
    with open(os.path.join(args.log_dir, 'description.txt'), 'w') as f:
        if args.custom_description:
            f.write("{}\n\n\nargs:\n".format(args.custom_description))
        args_dict = vars(args)
        for key in args_dict:
            f.write("\t{}: {}\n".format(key, args_dict[key]))


def main():
    args = argparser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    save_desctiption(args)
    main_log_filename = os.path.join(args.log_dir, 'log.txt')
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(main_log_filename, 'w'))

    assert args.preprocessing in supported_preprocessing_types, f"not supported preprocessing: {args.preprocessing}"

    net_config = NetConfig(object_types_fname=args.object_types_fname,
                           max_side=args.max_image_side,
                           preprocessing=supported_preprocessing_types[args.preprocessing],
                           fml_compatible=not args.fml_incompatible)

    net_manager = NetManager(log_dir=args.log_dir, net_config=net_config)
    if args.cfm:
        net_config = net_manager.load_another_model(another_log_dir=args.cfm)
    else:
        net_config = net_manager.build_model()
    net_manager.save_config()
    model = net_manager.get_keras_model()
    model.summary()

    model.compile(optimizer=Adam(args.lr),
                  loss=losses.get_loss(classification_mode=net_config.is_classification_supported()),
                  metrics=get_all_metrics(classification_mode=net_config.is_classification_supported()))

    train_generator = BatchGenerator(
        args.train_markup_path,
        batch_size=args.batch_size,
        markup_type=args.markup_type,
        net_config=net_config,
        validation_mode=False,
        n_workers=args.n_workers,
        prepare_batch_size=args.prepare_batch_size
    )
    # этот генератор используется на валидации для обучающей выборки
    # отдельно чтобы 1)отрисовывать больше картинок 2)смотреть на неаугментированную выборку
    # в принципе он не то чтобы сильно нужен, но пусть будет пока
    train_eval_generator = BatchGenerator(
        args.train_markup_path,
        batch_size=args.batch_size,
        markup_type=args.markup_type,
        net_config=net_config,
        validation_mode=True,
        prepare_batch_size=200,  # тут стоит что-то не очень большое, дабы память не жрать сильно
        yield_incomplete_batches=True,  # дабы сохранять больше картинок
        n_workers=args.n_workers,
    )
    val_generator = BatchGenerator(
        args.valid_markup_path,
        batch_size=args.batch_size,
        markup_type=args.markup_type,
        net_config=net_config,
        validation_mode=True,
        prepare_batch_size=1000,
        yield_incomplete_batches=True,
        n_workers=args.n_workers,
    )

    callbacks = build_callbacks_list(args.log_dir, net_config, train_eval_generator, val_generator,
                                     max_evaluated_images=args.max_evaluated_images)

    # бывает так что валидация сильно большая, чтобы всю не проходить, все равно это примерная оценка
    validation_steps = max(min(train_generator.get_epoch_size() // 20, val_generator.get_epoch_size()), 1)
    model.fit_generator(generator=train_generator.generate(),
                        steps_per_epoch=train_generator.get_epoch_size(),
                        validation_data=val_generator.generate(),
                        validation_steps=validation_steps,
                        epochs=args.epochs,
                        max_queue_size=args.prepare_batch_size // args.batch_size,
                        verbose=1,
                        callbacks=callbacks,
                        # если ставить workers=1 возникает необъяснимая ошибка зависания в случайный момент
                        # обучения, например, дня через полтора, она вытекает из многопоточности внутри самого
                        # генератора (где-то происходит seg fault судя по всему,
                        # а multiprocessing с таким не справляется)
                        workers=0)

    net_manager.save_inference()


if __name__ == '__main__':
    main()
