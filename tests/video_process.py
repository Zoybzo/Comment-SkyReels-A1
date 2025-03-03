from insightface.app import FaceAnalysis
from decord import VideoReader
from inference import parse_video, customize_loguru_logger
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor


def test_video(video_path='assets/driving_video/1.mp4', max_frame_num=49):
    processor = FaceAnimationProcessor(
        checkpoint='pretrained_models/smirk/SMIRK_em1.pt')
    control_frames = parse_video(video_path, max_frame_num)
    # driving video crop face
    driving_video_crop = []
    for control_frame in control_frames:
        frame, _, _ = processor.face_crop(control_frame)
        driving_video_crop.append(frame)

if __name__ == '__main__':
    customize_loguru_logger()
    test_video()
