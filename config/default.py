import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.WORD_DIM = 300
_C.INPUT.FRAME_DIM = 500
_C.INPUT.MAX_WORD_NUM = 30
_C.INPUT.MAX_FRAME_NUM = 256

# -----------------------------------------------------------------------------
# Path
# -----------------------------------------------------------------------------
_C.PATH = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAME = ""
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "PYRAMID"

# -----------------------------------------------------------------------------
# Pyramid
# -----------------------------------------------------------------------------
_C.MODEL.PYRAMID = CN()
_C.MODEL.PYRAMID.MODEL_SIZE = 256
_C.MODEL.PYRAMID.DROPOUT = 0.1

_C.MODEL.SWITCH = CN()
_C.MODEL.SWITCH.USE_FILTER = False
_C.MODEL.SWITCH.SALIENT_SEGMENT = True

_C.MODEL.PYRAMID.FILTER = CN()
_C.MODEL.PYRAMID.FILTER.HIDDEN_SIZE = _C.MODEL.PYRAMID.MODEL_SIZE

_C.MODEL.PYRAMID.SCORER = CN()
_C.MODEL.PYRAMID.SCORER.INPUT_SIZE = _C.MODEL.PYRAMID.MODEL_SIZE
_C.MODEL.PYRAMID.SCORER.HIDDEN_SIZE = _C.MODEL.PYRAMID.MODEL_SIZE
_C.MODEL.PYRAMID.SCORER.KERNEL_SIZE = (3, 9, 9)
_C.MODEL.PYRAMID.SCORER.STRIDE = 1
_C.MODEL.PYRAMID.SCORER.PADDING = 2
_C.MODEL.PYRAMID.SCORER.DILATION = 1

_C.MODEL.PYRAMID.VIDEO_ENCODER = CN()
_C.MODEL.PYRAMID.VIDEO_ENCODER.INPUT_SIZE = _C.INPUT.FRAME_DIM
_C.MODEL.PYRAMID.VIDEO_ENCODER.HIDDEN_SIZE = _C.MODEL.PYRAMID.MODEL_SIZE
_C.MODEL.PYRAMID.VIDEO_ENCODER.MAX_FRAME_NUM = _C.INPUT.MAX_FRAME_NUM
_C.MODEL.PYRAMID.VIDEO_ENCODER.SALIENT_SEGMENT = _C.MODEL.SWITCH.SALIENT_SEGMENT

_C.MODEL.PYRAMID.QUERY_ENCODER = CN()
_C.MODEL.PYRAMID.QUERY_ENCODER.INPUT_SIZE = _C.INPUT.WORD_DIM
_C.MODEL.PYRAMID.QUERY_ENCODER.HIDDEN_SIZE = _C.MODEL.PYRAMID.MODEL_SIZE

_C.MODEL.PYRAMID.FUSION = CN()
_C.MODEL.PYRAMID.FUSION.HIDDEN_SIZE = _C.MODEL.PYRAMID.MODEL_SIZE
_C.MODEL.PYRAMID.FUSION.ENCODER_TYPE = "GRU"

_C.MODEL.PYRAMID.SPARSE = CN()
_C.MODEL.PYRAMID.SPARSE.STEP = 3
_C.MODEL.PYRAMID.SPARSE.STRIDES = [1, 2, 4, 8]
_C.MODEL.PYRAMID.SPARSE.BASES = [1, 4, 16, 64]


_C.MODEL.PYRAMID.LOSS = CN()
_C.MODEL.PYRAMID.LOSS.NORM1 = 0.1
_C.MODEL.PYRAMID.LOSS.NORM2 = 0.01
_C.MODEL.PYRAMID.LOSS.INTRA = 0.1

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCH = 20
_C.TRAIN.CHECKPOINT_PERIOD = 1
_C.TRAIN.TEST_PERIOD = 1
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.TOPK = 32
_C.TRAIN.NUM_WORKERS = 4


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.0008
_C.OPTIMIZER.WEIGHT_DECAY = 1.0e-7
_C.OPTIMIZER.WARMUP_UPDATES = 200
_C.OPTIMIZER.WARMUP_INIT_LR = 1.0e-7

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NUM_WORKERS = 4

