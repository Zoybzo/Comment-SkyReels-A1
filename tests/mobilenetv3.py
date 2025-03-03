import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
import timm
from loguru import logger as loguru_logger

from inference import customize_loguru_logger
from skyreels_a1.src.smirk_encoder import SmirkEncoder


def test_smirk_encoder():
    smirk_encoder = SmirkEncoder()
    loguru_logger.log('UNIT_DEBUG', "Load Pose Encoder")
    pose_encoder = smirk_encoder.pose_encoder
    summary(pose_encoder, (3, 224, 224))
    loguru_logger.log('UNIT_DEBUG', "Load Pose Encoder")
    expression_encoder = smirk_encoder.expression_encoder
    summary(expression_encoder, (3, 224, 224))
    loguru_logger.log('UNIT_DEBUG', "Load Pose Encoder")
    shape_encoder = smirk_encoder.shape_encoder
    summary(shape_encoder, (3, 224, 224))


if __name__ == '__main__':
    customize_loguru_logger()
    test_smirk_encoder()
