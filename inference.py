import torch
import os
import numpy as np
from PIL import Image
import glob
import insightface
import cv2
import subprocess
import argparse
from decord import VideoReader
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
from loguru import logger as loguru_logger

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video, load_image
from transformers import AutoModelForDepthEstimation, AutoProcessor, \
    SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from skyreels_a1.skyreels_a1_i2v_pipeline import \
    SkyReelsA1ImagePoseToVideoPipeline
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d


def customize_loguru_logger():
    # UNIT_DEBUG
    loguru_logger.level("UNIT_DEBUG", no=16, color="<green>",
                        icon="🧪")  # 单元测试的调试信息
    loguru_logger.add("logs/unit_debug.log",
                      filter=lambda record: record[
                                                "level"].name == "UNIT_DEBUG",
                      level="UNIT_DEBUG")
    loguru_logger.log("UNIT_DEBUG", "This is a unit test debug message.")
    # MODEL DEBUG
    loguru_logger.level("MODEL_DEBUG", no=15, color="<blue>",
                        icon="🤖")  # 测试模型的调试信息
    loguru_logger.add("logs/model_debug.log",
                      filter=lambda record: record[
                                                "level"].name == "MODEL_DEBUG",
                      level="MODEL_DEBUG")
    loguru_logger.log("MODEL_DEBUG", "This is a model debug message.")


def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left + croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.zeros((image_height, image_width + pad * 2, 3),
                                dtype=np.uint8)
        padded_image[:, pad:pad + image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    return image


def write_mp4(video_path, samples, fps=12, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac",
                         audio_bitrate=audio_bitrate,
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])


def parse_video(driving_video_path, max_frame_num):
    """
    处理视频帧数，最终返回 max-1 帧；
    1. 复制第一帧；
    2. 帧数不足时复制最后一帧，过长则截断
    """
    vr = VideoReader(driving_video_path)  # decord.NDarray，类似于numpy
    fps = vr.get_avg_fps()  # 获取视频帧率
    video_length = len(vr)  # 视频的总帧数

    duration = video_length / fps  # 时长:秒
    target_times = np.arange(0, duration, 1 / 12)  # 每秒提取12帧，获取每一帧的时间点
    frame_indices = (target_times * fps).astype(np.int32)  # 根据时间点获取帧的ID

    frame_indices = frame_indices[frame_indices < video_length]
    control_frames = vr.get_batch(frame_indices).asnumpy()[
                     :(max_frame_num - 1)]  # 根据帧索引提取帧,最多max-1个帧
    loguru_logger.log('MODEL_DEBUG', f"Control Frames: {len(control_frames)}")
    # 为什么max-1？
    # max-1是为了之后可以复制一次第一帧

    out_frames = len(control_frames) - 1
    if len(control_frames) < max_frame_num - 1:  # 帧数不足
        video_lenght_add = max_frame_num - len(control_frames) - 1
        control_frames = np.concatenate(([control_frames[0]] * 2,
                                         control_frames[
                                         1:len(control_frames) - 1], [
                                             control_frames[
                                                 -1]] * video_lenght_add),
                                        axis=0)  # 帧数不足时，复制最后一帧补全
    else:
        control_frames = np.concatenate(([control_frames[0]] * 2,
                                         control_frames[
                                         1:len(control_frames) - 1]), axis=0)

    return control_frames


def exec_cmd(cmd):
    return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


def add_audio_to_video(silent_video_path: str, audio_video_path: str,
                       output_video_path: str):
    cmd = [
        'ffmpeg',
        '-y',
        '-i', f'"{silent_video_path}"',
        '-i', f'"{audio_video_path}"',
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        f'"{output_video_path}"'
    ]

    try:
        exec_cmd(' '.join(cmd))
        print(f"Video with audio generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video and image for face animation.")
    parser.add_argument('--image_path', type=str,
                        default="assets/ref_images/21.JPG",
                        help='Path to the source image.')
    parser.add_argument('--driving_video_path', type=str,
                        default="assets/driving_video/1.mp4",
                        help='Path to the driving video.')
    parser.add_argument('--output_path', type=str, default="outputs",
                        help='Path to save the output video.')
    parser.add_argument('--prefix', type=str, default="./../Models/",
                        help='The path to load the model.')
    parser.add_argument('--infer', type=bool, default=True,
                        help='Inference or not.')
    parser.add_argument('--cpu', type=bool, default=True, )
    args = parser.parse_args()
    infer = args.infer
    use_cpu = args.cpu
    # Add logger
    customize_loguru_logger()

    guidance_scale = 3.0
    seed = 43
    num_inference_steps = 10
    sample_size = [480, 720]
    max_frame_num = 49
    weight_dtype = torch.bfloat16
    save_path = args.output_path
    generator = torch.Generator(device='cpu').manual_seed(
        seed) if use_cpu else torch.Generator(device="cuda").manual_seed(seed)
    prefix = args.prefix  # Need change many path; stay here and don't use it
    model_name = "pretrained_models/SkyReels-A1-5B/"
    siglip_name = "pretrained_models/SkyReels-A1-5B/siglip-so400m-patch14-384"

    lmk_extractor = LMKExtractor()
    processor = FaceAnimationProcessor(
        checkpoint='pretrained_models/smirk/SMIRK_em1.pt')
    vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False,
                               draw_iris=False, )
    face_helper = FaceRestoreHelper(upscale_factor=1, face_size=512,
                                    crop_ratio=(1, 1),
                                    det_model='retinaface_resnet50',
                                    save_ext='png', device="cuda", )

    # siglip visual encoder
    siglip = SiglipVisionModel.from_pretrained(siglip_name)
    siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_name)

    # 处理视频帧数
    loguru_logger.info("Parse Video...")
    control_frames = parse_video(args.driving_video_path, max_frame_num)
    loguru_logger.log('MODEL_DEBUG', f"Frames: {len(control_frames)}")
    loguru_logger.log('MODEL_DEBUG', f"Shape 0: {control_frames[0].shape}")
    loguru_logger.log('MODEL_DEBUG', f'Shape 2: {control_frames[2].shape}')
    assert control_frames[0].shape == control_frames[2].shape

    loguru_logger.info('Crop Driving video...')
    # driving video crop face
    driving_video_crop = []
    for control_frame in control_frames:
        frame, _, _ = processor.face_crop(control_frame)  # 这里得到的每一帧的大小并不相同
        driving_video_crop.append(frame)
    Image.fromarray(driving_video_crop[0]).save(
        "assets/tmp/crop_driving_video_0.jpg")
    Image.fromarray(driving_video_crop[2]).save(
        "assets/tmp/crop_driving_video_2.jpg")
    Image.fromarray(driving_video_crop[-1]).save(
        "assets/tmp/crop_driving_video_-1.jpg")
    loguru_logger.log('MODEL_DEBUG', f"Frames: {len(driving_video_crop)}")
    # 一般情况下 大小不同
    loguru_logger.log('MODEL_DEBUG', f'Shape 0: {driving_video_crop[0].shape}')
    loguru_logger.log('MODEL_DEBUG', f'Shape 2: {driving_video_crop[2].shape}')
    loguru_logger.log('MODEL_DEBUG',
                      f'Shape -1: {driving_video_crop[-1].shape}')

    # Check the crop driving image lmk
    crop_driving_frame_0 = cv2.resize(driving_video_crop[0], (512, 512))
    crop_driving_frame_0_lmk = lmk_extractor(
        driving_video_crop[0][:, :, ::-1])  # RGB -> BGR
    crop_driving_frame_0 = vis.draw_landmarks_v3((512, 512), (
        driving_video_crop[0].shape[0], driving_video_crop[0].shape[1]),
                                                 crop_driving_frame_0_lmk[
                                                     'lmks'].astype(np.float32),
                                                 normed=True)
    Image.fromarray(crop_driving_frame_0).save(
        'assets/tmp/lmk_crop_driving_video_0.jpg')

    image = load_image(image=args.image_path)
    loguru_logger.log('MODEL_DEBUG', f"Image: {np.array(image).shape}")
    image = processor.crop_and_resize(image, sample_size[0], sample_size[1])
    loguru_logger.log('MODEL_DEBUG',
                      f"Crop and Resize Image: {np.array(image).shape}")
    assert np.array(image).shape == (sample_size[0], sample_size[1], 3)
    # Shape: [sp0, sp1, 3] / [480,720,3]

    # ref image crop face
    ref_image, x1, y1 = processor.face_crop(np.array(image))
    loguru_logger.log('MODEL_DEBUG', f'Face Crop Image: {ref_image.shape}')
    Image.fromarray(ref_image).save(f"assets/tmp/crop_ref_image.jpg")
    face_h, face_w, _, = ref_image.shape
    source_image = ref_image  # Shape 不固定，根据图片中人脸的大小而定
    driving_video = driving_video_crop
    # INFO: 使用 FLAME+mediapipe 预处理视频帧，得到关键点；处理后的 frame 的大小与 ref_image 一致
    out_frames = processor.preprocess_lmk3d(source_image,
                                            driving_video)
    loguru_logger.log('MODEL_DEBUG', f"Out Frames: {len(out_frames)}")
    loguru_logger.log('MODEL_DEBUG', f"Out Frames 0: {out_frames[0].shape}")
    loguru_logger.log('MODEL_DEBUG', f"Out Frames 2: {out_frames[2].shape}")
    loguru_logger.log('MODEL_DEBUG', f"Out Frames -1: {out_frames[-1].shape}")
    assert out_frames[0].shape == ref_image.shape
    assert out_frames[2].shape == ref_image.shape
    # save out_frames[0] as a image
    Image.fromarray(out_frames[0]).save('assets/tmp/out_frames_0.jpg')
    Image.fromarray(out_frames[2]).save('assets/tmp/out_frames_2.jpg')

    # INFO: 生成 48 帧的运动，将处理后的 关键点 信息放入到 48 帧中; Rescale to Sample Size
    loguru_logger.info("Rescale Motions to Sample Size...")
    rescale_motions = np.zeros_like(image)[np.newaxis, :].repeat(48,
                                                                 axis=0)
    # Shape: [new(48), h, w, ch]
    loguru_logger.log('MODEL_DEBUG',
                      f"Rescale Motions: {rescale_motions.shape}")
    for ii in range(rescale_motions.shape[0]):
        rescale_motions[ii][y1:y1 + face_h, x1:x1 + face_w] = out_frames[ii]
    loguru_logger.log('MODEL_DEBUG',
                      f"Rescale Motions 0: {rescale_motions[0].shape}")
    loguru_logger.log('MODEL_DEBUG',
                      f"Rescale Motions 2: {rescale_motions[2].shape}")
    # save rescale_motions[0] as a image
    Image.fromarray(rescale_motions[0]).save("assets/tmp/rescale_motions_0.jpg")
    Image.fromarray(rescale_motions[2]).save("assets/tmp/rescale_motions_2.jpg")

    # 3D 信息 ? 得到的图片还是关键点
    ref_image = cv2.resize(ref_image, (512, 512))
    ref_lmk = lmk_extractor(ref_image[:, :, ::-1])  # RGB -> BGR
    """
    {
        "lmks": lmks, # [[x,y,z], [x,y,z]]
        'lmks3d': lmks3d,
        "trans_mat": trans_mat,
        'faces': mp_tris,
        "bs": bs_values
    }
    """
    # 3D 信息似乎并没有被下方的函数处理
    ref_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h),
                                    ref_lmk['lmks'].astype(np.float32),
                                    normed=True)
    # show ref_img
    loguru_logger.log('MODEL_DEBUG', f"Dtype of ref_img: {ref_image.dtype}")
    Image.fromarray(ref_img).save("assets/tmp/ref_img.jpg")

    # 加上第一帧
    loguru_logger.info('Add the reference image as the first frame')
    first_motion = np.zeros_like(np.array(image))
    first_motion[y1:y1 + face_h, x1:x1 + face_w] = ref_img
    first_motion = first_motion[np.newaxis, :]  # Shape: [h, w, ch]
    # Ref 作为第一帧在最前面，现在是 49 帧
    motions = np.concatenate([first_motion, rescale_motions])
    input_video = motions[:max_frame_num]
    loguru_logger.log('MODEL_DEBUG', f"len(input_video): {len(input_video)}")
    loguru_logger.log('MODEL_DEBUG', f"Shape input_video: {input_video.shape}")

    # 读取图像并检测人脸
    loguru_logger.info("Check the face in reference image")
    face_helper.clean_all()
    face_helper.read_image(np.array(image)[:, :, ::-1])  # 反转后3个通道 RGB->BGR
    # image 为原始图片 crop 和 resize 之后的图片
    face_helper.get_face_landmarks_5(only_center_face=True)  # 人脸的检测
    face_helper.align_warp_face()  # 人脸的对齐与变形
    align_face = face_helper.cropped_faces[0]  # 人脸裁剪
    image_face = align_face[:, :, ::-1]  # 拿到人脸，反转通道
    # save image_face
    loguru_logger.log('MODEL_DEBUG', f"Shape image_face: {image_face.shape}")
    Image.fromarray(image_face).save("assets/tmp/image_face.jpg")

    input_video = input_video[:max_frame_num]
    motions = np.array(input_video)  # Shape: [H, W, C, F] # 之后就没有用了

    # [F, H, W, C] -> [1, C, F, H, W]
    input_video = torch.from_numpy(np.array(input_video)).permute(
        [3, 0, 1, 2]).unsqueeze(0)
    loguru_logger.log('MODEL_DEBUG',
                      f"Permute Shape input_video: {input_video.shape}")
    input_video = input_video / 255  # norm

    out_samples = []

    # Load Model
    # skyreels a1 model
    if not infer:
        exit(0)

    loguru_logger.info("Loading CogVideoX...")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        load_in_8bit=True,
        # device_map="auto" # CogVideoX 不支持 auto
    ).to(weight_dtype)

    loguru_logger.info("Loading VAE...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder="vae",
        # device_map="balanced",
        load_in_8bit=True,
    ).to(weight_dtype)

    loguru_logger.info("Loading lmk_encoder...")
    lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder="pose_guider",
        # device_map="balanced",
        load_in_8bit=True,
    ).to(weight_dtype)

    loguru_logger.info("Loading SkyReels...")
    # device_map = {
    #     "feature_extractor": "cuda:0",
    #     "image_encoder": "cuda:0",
    #     "scheduler": "cuda:0",
    #     "lmk_encoder": "cuda:1",
    #     "transformer": "cuda:1",
    #     "vae": "cuda:1",
    #     "face_helper": "cuda:2",
    #     "text_encoder": "cuda:3",
    #     "tokenizer": "cuda:3",
    # } # not suppored this one
    pipe = SkyReelsA1ImagePoseToVideoPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        vae=vae,
        lmk_encoder=lmk_encoder,
        image_encoder=siglip,
        feature_extractor=siglip_normalize,
        # face_helper=face_helper,
        torch_dtype=torch.bfloat16,
        # device_map="auto", # not supported
        # device_map="balanced_low_0", # not supported
        # device_map="balanced",
        device="cpu",
        # device_map=device_map,
        # load_in_8bit=True, # not supported
    )
    # 确保模型组件被正确加载到指定设备
    # for name, module in pipe.named_modules():
    #     if name in device_map:
    #         module.to(device_map[name])
    loguru_logger.info("Loaded All !!!")

    # pipe.to("cuda")
    # pipe.enable_model_cpu_offload()
    # enable_model_cpu_offload 不可以和 device_map 一起使用
    pipe.vae.enable_tiling()

    with torch.no_grad():
        sample = pipe(
            image=image,  # 裁剪图片 [sp0, sp1, C]
            image_face=image_face,  # 修复后的人脸图片 # [W, H, C]: W=H=512
            control_video=input_video,
            # 关键点处理后的视频信息 # [1, C, F, H(sp0), W(sp1)]
            prompt="",
            negative_prompt="",
            height=sample_size[0],
            width=sample_size[1],
            num_frames=49,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        out_samples.extend(sample.frames[0])  # 复制了第一帧 # 为什么？
    out_samples = out_samples[2:]  # 抛弃前两帧 # 为什么？

    save_path_name = os.path.basename(args.image_path).split(".")[0] + "-" + \
                     os.path.basename(args.driving_video_path).split(".")[
                         0] + ".mp4"

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, save_path_name + "-output.mp4")
    export_to_video(out_samples, video_path, fps=12)

    target_h, target_w = sample_size[0], sample_size[1]
    final_images = []
    final_images2 = []
    rescale_motions = rescale_motions[1:]
    control_frames = control_frames[1:]
    for q in range(len(out_samples)):
        frame1 = image  # 参考图片
        frame2 = crop_and_resize(
            Image.fromarray(np.array(control_frames[q])).convert("RGB"),
            target_h, target_w)
        frame3 = Image.fromarray(np.array(out_samples[q])).convert("RGB")

        result = Image.new('RGB', (target_w * 3, target_h))
        result.paste(frame1, (0, 0))
        result.paste(frame2, (target_w, 0))
        result.paste(frame3, (target_w * 2, 0))
        final_images.append(np.array(result))

    video_out_path = os.path.join(save_path, save_path_name)
    write_mp4(video_out_path, final_images, fps=12)

    add_audio_to_video(video_out_path, args.driving_video_path,
                       video_out_path + ".audio.mp4")
    add_audio_to_video(video_path, args.driving_video_path,
                       video_path + ".audio.mp4")
