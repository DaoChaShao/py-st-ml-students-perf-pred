#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Machine Learning Package - Comprehensive ML Framework
----------------------------------------------------------------
A comprehensive machine learning package providing end-to-end solutions
for various ML tasks including computer vision, natural language processing,
and sequential data analysis.

Core Modules:
- criteria: Specialized loss functions and evaluation metrics for
  segmentation, classification, and regression tasks
- dataloader: Custom data loading utilities with enhanced batching
  and sampling strategies
- datasets: PyTorch Dataset implementations for image segmentation,
  classification, and sequence prediction
- nets: Neural network architectures including UNet variants for
  segmentation and RNN models for sequence tasks
- trainers: Complete training frameworks with automated loops,
  metric tracking, and model management
- utils: Essential utilities for configuration, timing, text processing,
  tensor operations, and statistical analysis

Key Features:
- Modular design for easy experimentation
- Support for both computer vision and NLP tasks
- Production-ready training pipelines
- Comprehensive evaluation metrics
- Flexible configuration system
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from . import criteria
from . import dataloader
from . import datasets
from . import nets
from . import trainers
from . import utils

__all__ = [
    "criteria",
    "dataloader",
    "datasets",
    "nets",
    "trainers",
    "utils",
]
