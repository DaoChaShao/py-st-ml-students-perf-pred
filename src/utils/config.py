#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 23:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :

from dataclasses import dataclass, field
from pathlib import Path
from torch import cuda

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class FilePaths:
    SEM_SEG_MODEL: Path = BASE_DIR / "models/sem_seg_model.pth"
    SEQ_CLASSES_MODEL: Path = BASE_DIR / "models/seq_classes_model.pth"
    SPACY_MODEL_EN: Path = BASE_DIR / "models/spacy/en_core_web_md"
    SPACY_MODEL_ZH: Path = BASE_DIR / "models/spacy/zh_core_web_md"
    STANZA_MODEL: Path = BASE_DIR / "models/stanza"
    TRAIN: Path = BASE_DIR / "data/train/"
    TEST: Path = BASE_DIR / "data/test/"
    DICTIONARY: Path = BASE_DIR / "data/dictionary.json"


@dataclass
class DataPreprocessor:
    PCA_VARIANCE_THRESHOLD: float = 0.95

    RANDOM_STATE: int = 27
    TEST_SIZE: float = 0.2
    SHUFFLE_STATUS: bool = True
    BATCHES: int = 16

    IMAGE_HEIGHT: int = 320
    IMAGE_WIDTH: int = 384


@dataclass
class ModelParams:
    DROPOUT_RATE: float = 0.3
    FC_HIDDEN_UNITS: int = 128


@dataclass
class CNNParams:
    CNN_OUT_CHANNELS: int = 64
    CNN_KERNEL_SIZE: int = 3
    CNN_STRIDE: int = 1
    CNN_PADDING: int = 1


@dataclass
class RNNParams:
    EMBEDDING_DIM: int = 256
    HIDDEN_SIZE: int = 128
    LAYERS: int = 2
    TEMPERATURE: float = 1.0
    CLASSES: int = 2  # Binary classification


@dataclass
class UNetParams:
    INITIAL_FILTERS: int = 64
    SEG_CLASSES: int = 1  # Binary segmentation: 1 channel output


@dataclass
class Hyperparameters:
    ALPHA: float = 1e-4
    EPOCHS: int = 100
    DECAY: float = 1e-4
    ACCELERATOR: str = "cuda" if cuda.is_available() else "cpu"


@dataclass
class Configuration:
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    MODEL_PARAMS: ModelParams = field(default_factory=ModelParams)
    CNN_PARAMS: CNNParams = field(default_factory=CNNParams)
    RNN_PARAMS: RNNParams = field(default_factory=RNNParams)
    UNET_PARAMS: UNetParams = field(default_factory=UNetParams)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)


CONFIG = Configuration()
