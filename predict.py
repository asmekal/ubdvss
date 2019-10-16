#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2019. All rights reserved.
"""
Скрипт для оценки результата работы сети (численно) и получения визуализаций
"""
import argparse
import logging
import os
import time

import numpy as np

from semantic_segmentation.data_generators import BatchGenerator
from semantic_segmentation.model_runner import ModelRunner
from semantic_segmentation.net import NetManager, NetConfig

argparser = argparse.ArgumentParser()
argparser.add_argument('--source', '-s', type=str, required=True,
                       help="path to data markup dir (contains Image and Markup subfolders)")
argparser.add_argument('--dest', '-d', type=str, required=False,
                       help="path to dir with results")
argparser.add_argument('--log_dir', '-l', type=str, required=True,
                       help="path to training logging dir with config and saved models")
argparser.add_argument('--model_path', type=str, default=None,
                       help="path to trained model (.h5) - either local, global or from log_dir; "
                            "if None, load last model from log_dir")
argparser.add_argument('--markup_type', '-mt', type=str, default="Barcode",
                       help="markup type for train and test")
argparser.add_argument('--batch_size', '-b', type=int, default=8,
                       help="batch size for train, test and evaluation")
argparser.add_argument('--n_workers', '-n', type=int, default=4,
                       help="number of preprocessing threads")
argparser.add_argument('--prepare_batch_size', '-pbs', type=int, default=3000,
                       help="number of preprocessed images before groupby")
argparser.add_argument('--max_image_side', type=int, default=None,
                       help="max size for image height and width "
                            "(if it is larger image will be downsized maintaining aspect ratio)")
argparser.add_argument('--visualize', '-viz', action='store_true',
                       help="if True draws images with gt, results, segmentation (and classification if enabled)")
argparser.add_argument('--min_detection_area', '-min_area', type=int, default=None,
                       help="found connected components with area less than this value will be filtered out")


def probability(string):
    value = float(string)
    if not (0 <= value <= 1):
        raise argparse.ArgumentTypeError(f"{value} is not a probability")
    return value


argparser.add_argument('--pixel_threshold', '-pixel_threshold', type=probability, default=0.5,
                       help="threshold to consider pixel as detected, should be between 0 and 1")


def main():
    args = argparser.parse_args()
    if not args.dest:
        args.dest = os.path.join(args.log_dir, "results", "last_result")
    else:
        args.dest = os.path.join(args.log_dir, "results", args.dest)
    os.makedirs(args.dest, exist_ok=True)

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(args.dest, 'log.txt'), 'w'))

    net_manager = NetManager(args.log_dir)
    net_config = net_manager.load_model(args.model_path)
    net_config = NetConfig.from_others(net_config,
                                       max_image_side=args.max_image_side,
                                       min_pixels_for_detection=args.min_detection_area)
    model = net_manager.get_keras_model()
    input_image_shape = (1, net_config.get_max_side(), net_config.get_max_side(), 1)
    model.predict(np.zeros(input_image_shape))  # чтобы точно скомпилилось
    t = time.time()
    model.predict(np.zeros(input_image_shape))
    t = time.time() - t
    logging.info(f"INFERENCE TIME: {t}, shape={input_image_shape}")

    test_generator = BatchGenerator(
        args.source,
        batch_size=args.batch_size,
        markup_type=args.markup_type,
        net_config=net_config,
        use_augmentation=False,
        prepare_batch_size=args.prepare_batch_size,
        yield_incomplete_batches=True,
        n_workers=args.n_workers
    )

    logging.info(f"Config: {net_config}")

    logging.info(f"Predicting {args.source} --> {args.dest}")
    model_runner = ModelRunner(net_config,
                               pixel_threshold=args.pixel_threshold)
    metrics, _ = model_runner.run(model=model,
                                  data_generator=test_generator.generate(add_metainfo=True),
                                  n_images=test_generator.get_images_per_epoch(),
                                  save_dir=args.dest,
                                  save_visualizations=args.visualize)

    with open(os.path.join(args.dest, "result.txt"), 'w') as f:
        f.write("Evaluation {} images from {}\n\n".format(test_generator.get_images_per_epoch(), args.source))
        for metric_name, metric_value in metrics.items():
            f.write("{}: {}\n".format(metric_name, metric_value))


if __name__ == '__main__':
    main()
