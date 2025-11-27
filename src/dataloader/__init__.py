#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :   

"""
****************************************************************
Dataloader Module
----------------------------------------------------------------
This module provides a custom PyTorch DataLoader wrapper with
simplified interface and enhanced accessibility.

Features:
- Simplified constructor with intuitive parameter names
- Direct dataset indexing support via __getitem__
- Property access to underlying dataset
- Clean repr for debugging
- Maintains full compatibility with standard DataLoader iteration

Note: This is a wrapper around torch.utils.data.DataLoader
that provides a more Pythonic interface rather than extending
functionality.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .general import TorchDataLoader

__all__ = [
    "TorchDataLoader",
]
