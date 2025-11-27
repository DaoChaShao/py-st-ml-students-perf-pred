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
from .JB import (cut_accuracy, cut_full, cut_search,
                 cut_pos as jb_cut_pos,
                 extract_tfidf_weights, extract_textrank_weights)
from .nlp import (regular_chinese, regular_english,
                  count_frequency, unique_characters, extract_zh_chars,
                  spacy_single_tokeniser, stanza_tokeniser)
from .PT import (TorchRandomSeed, TorchDataLoader, GrayTensorReshaper,
                 check_device, get_device, arr2tensor, df2tensor)
from .stats import (NumpyRandomSeed,
                    load_csv, load_text, summary_dataframe,
                    load_paths, split_data, save_json, load_json,
                    standardise_data,
                    split_array,
                    select_pca_importance)
from .THU import cut_pos as thu_cut_pos, cut_only

__all__ = [
    "CONFIG",

    "timer", "beautifier",

    "Timer", "Beautifier", "RandomSeed", "read_file",

    "black", "red", "green", "yellow", "blue", "purple", "cyan", "white",
    "bold", "underline", "invert", "strikethrough",
    "starts", "lines", "sharps",

    "cut_accuracy", "cut_full", "cut_search",
    "jb_cut_pos",
    "extract_tfidf_weights", "extract_textrank_weights",

    "regular_chinese", "regular_english",
    "count_frequency", "unique_characters", "extract_zh_chars",
    "spacy_single_tokeniser", "stanza_tokeniser",

    "TorchRandomSeed", "TorchDataLoader", "GrayTensorReshaper",
    "check_device", "get_device", "arr2tensor", "df2tensor",

    "NumpyRandomSeed",
    "load_csv", "load_text", "summary_dataframe",
    "load_paths", "split_data", "save_json", "load_json",
    "standardise_data",
    "split_array",
    "select_pca_importance",

    "thu_cut_pos", "cut_only",
]
