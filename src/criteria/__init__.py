#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :   

"""
****************************************************************
Criterion Module - Loss Functions for Neural Network Training
----------------------------------------------------------------
This module provides specialized loss functions for various machine
learning tasks, particularly focused on segmentation and regression
problems with class imbalance and boundary sensitivity.

Available Loss Functions:
- DiceBCELoss: Combined Dice coefficient and Binary Cross-Entropy loss
  for segmentation tasks, balancing region overlap and pixel-wise accuracy

- ComprehensiveLossWithDiceAndFocal: Multi-component loss combining
  Dice, Focal, and boundary-aware terms for robust segmentation

- EdgeAwareLoss: Specialized segmentation loss with emphasis on
  preserving object boundaries and edge details

- FocalLoss: Adaptive cross-entropy variant that addresses class
  imbalance by focusing on hard-to-classify examples

- LogMSELoss: Mean Squared Error with logarithmic scaling for
  regression tasks with large value ranges
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .mse_log import log_mse_loss
from .sem_dice import DiceBCELoss
from .sem_dnf import ComprehensiveLossWithDiceAndFocal
from .sem_edge import EdgeAwareLoss
from .sem_focal import FocalLoss

__all__ = [
    "log_mse_loss",
    "DiceBCELoss",
    "ComprehensiveLossWithDiceAndFocal",
    "EdgeAwareLoss",
    "FocalLoss",
]
