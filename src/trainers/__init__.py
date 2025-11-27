#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Neural Nets Trainer Module - Model Training Utilities
----------------------------------------------------------------
This module provides specialized trainer classes for training and
evaluating neural network models with comprehensive training loops,
metrics tracking, and model management.

Available Trainers:
- UNetSemSegTorchTrainer: Trainer for UNet-based semantic segmentation
  tasks with support for image-mask pairs and segmentation metrics

- RNNClassificationTorchTrainer: Trainer for RNN-based sequence
  classification tasks with sequence data processing and classification
  metrics
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .sem_classes import UNetSemSegTorchTrainer
from .seq_classes import RNNClassificationTorchTrainer

__all__ = [
    "UNetSemSegTorchTrainer",
    "RNNClassificationTorchTrainer",
]
