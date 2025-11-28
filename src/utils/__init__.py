#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :

"""
****************************************************************
Utility Module - Comprehensive Toolkit for ML/Data Processing
----------------------------------------------------------------
This module provides a comprehensive collection of utility functions
and classes for machine learning, natural language processing,
computer vision, and general data processing tasks.

Main Categories:
- Configuration management (CONFIG)
- Performance measurement and timing (timer, Timer)
- Text formatting and highlighting
- Chinese/English text processing and tokenization
- PyTorch tensor operations and data loading
- Statistical data processing and analysis
- File I/O operations (CSV, JSON, text)
- Data standardization and dimensionality reduction
- Multiple Chinese word segmentation implementations (JIEBA, THULAC)
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .config import CONFIG
from .decorator import timer, beautifier
from .helper import Timer, Beautifier, RandomSeed, read_file
from .highlighter import (black, red, green, yellow, blue, purple, cyan, white,
                          bold, underline, invert, strikethrough,
                          starts, lines, sharps)
from .PT import (TorchRandomSeed, TorchDataLoader, GrayTensorReshaper,
                 check_device, get_device, arr2tensor, df2tensor)
from .stats import (NumpyRandomSeed,
                    load_csv, load_text, summary_dataframe,
                    load_paths, split_paths, split_train, split_features, split_data,
                    save_json, load_json,
                    create_data_transformer, transform_data,
                    pca_importance,
                    get_correlation_btw_features, get_cat_correlation, get_correlation_btw_Xy)

__all__ = [
    "CONFIG",

    "timer", "beautifier",

    "Timer", "Beautifier", "RandomSeed", "read_file",

    "black", "red", "green", "yellow", "blue", "purple", "cyan", "white",
    "bold", "underline", "invert", "strikethrough",
    "starts", "lines", "sharps",

    "TorchRandomSeed", "TorchDataLoader", "GrayTensorReshaper",
    "check_device", "get_device", "arr2tensor", "df2tensor",

    "NumpyRandomSeed",
    "load_csv", "load_text", "summary_dataframe",
    "load_paths", "split_paths", "split_train", "split_data",
    "save_json", "load_json",
    "create_data_transformer", "transform_data",
    "pca_importance",
    "get_correlation_btw_features", "get_cat_correlation", "get_correlation_btw_Xy",
]
