#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Neural Nets Module - Neural Network Architectures
----------------------------------------------------------------
This module provides various neural network architectures for different
machine learning tasks including segmentation, classification, and
sequence modeling.

Available Models:
- Standard4LayersUNet: 4-layer UNet variant for semantic segmentation
- Standard5LayersUNet: 5-layer UNet variant for semantic segmentation
- RNNModelForClassification: Recurrent Neural Network for sequence
  classification tasks

Architecture Features:
- UNet variants with encoder-decoder structure and skip connections
- Support for binary and multi-class segmentation
- Configurable channel dimensions and input sizes
- RNN models with flexible hidden layers and activation functions
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .sem_standard4 import Standard4LayersUNet
from .sem_standard5 import Standard5LayersUNet
from .seq_classes import RNNModelForClassification

__all__ = [
    "Standard4LayersUNet",
    "Standard5LayersUNet",
    "RNNModelForClassification",
]
