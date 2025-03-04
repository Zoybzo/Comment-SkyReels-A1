from insightface.app import FaceAnalysis
from decord import VideoReader
from PIL import Image
import numpy as np

from inference import parse_video, customize_loguru_logger
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor


def test_no_face(image_path='assets/test_images/1.jpg'):
    processor = FaceAnimationProcessor(
        checkpoint='pretrained_models/smirk/SMIRK_em1.pt')
    img = Image.open(image_path).convert('RGB')
    frame, _, _ = processor.face_crop(np.array(img))


if __name__ == '__main__':
    customize_loguru_logger()
    test_no_face()
