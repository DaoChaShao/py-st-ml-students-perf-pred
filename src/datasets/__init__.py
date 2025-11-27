#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Datasets Module - PyTorch Dataset Implementations
----------------------------------------------------------------
This module provides specialized PyTorch Dataset classes for various
machine learning tasks including segmentation, classification, and
sequence prediction.

Available Datasets:
- SemSegDataset: Semantic segmentation dataset for image-mask pairs
- LabelTorchDatasetForClassification: Image classification dataset
  with label support
- SeqTorchDatasetForClassification: Sequence data for classification tasks
- SeqTorchDatasetForPrediction: Sequence data for prediction/forecasting tasks

Utilities:
- mask_map_class_id: Utility function for converting segmentation masks
  to class ID mappings and handling multi-class segmentation formats
- reshape_to_grayscale: Reshape tensor to grayscale image format for CNN input
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .label_classes import LabelTorchDatasetForClassification
from .mask_map import mask_map_class_id
from .reshaper import reshape_to_grayscale
from .sem_segmentation import SemSegDataset
from .seq_classes import SeqTorchDatasetForClassification
from .seq_prediction import SeqTorchDatasetForPrediction

__all__ = [
    "LabelTorchDatasetForClassification",
    "mask_map_class_id",
    "reshape_to_grayscale",
    "SemSegDataset",
    "SeqTorchDatasetForClassification",
    "SeqTorchDatasetForPrediction",
]
