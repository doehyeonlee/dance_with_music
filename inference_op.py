import argparse
import copy
import os
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import imageio
from animation.helper.backbones import face_embedding_extractor_get_model
from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.refined_vae import RefinedAutoencoderKLTemporalDecoder
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.pipelines.euler_discrete_pro import EulerDiscreteSchedulerPro
import random
from einops import rearrange
from animation.pipelines.inference_pipeline_animation_pro import InferenceAnimationProPipeline
import torch.nn.functional as F

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)

    return images

def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=125,
                       loop=0)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_image_mask",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_mask_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--height",
        type=int,
        default=768,
        required=False
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        required=False
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False
    )

    parser.add_argument(
        "--posenet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained posenet model",
    )
    parser.add_argument(
        "--face_encoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained face encoder model",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model",
    )

    parser.add_argument(
        "--max_frame_num",
        type=int,
        default=50,
        required=False
    )

    parser.add_argument(
        "--batch_frames",
        type=int,
        default=14,
        required=False
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        required=False
    )

    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,  # or set to 0.02
        required=False
    )
    parser.add_argument(
        "--frames_overlap",
        type=int,
        default=4,
        required=False
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--face_embedding_extractor_weight_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_optimization_iter",
        type=int,
        default=3,
        required=True
    )
    parser.add_argument(
        "--start_refine_step",
        type=int,
        default=0,
        required=True
    )
    parser.add_argument(
        "--end_refine_step",
        type=int,
        default=20,
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda:0'
    seed = 23123134
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteSchedulerPro.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = RefinedAutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = FaceModel()

    # weight_dtype = torch.float16
    weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16

    face_embedding_extractor_weight_path = args.face_embedding_extractor_weight_path
    if weight_dtype == torch.float32:
        face_embedding_extractor = face_embedding_extractor_get_model(name='r100', fp16=False)
    else:
        face_embedding_extractor = face_embedding_extractor_get_model(name='r100', fp16=True)
    face_embedding_extractor_state_dict = torch.load(face_embedding_extractor_weight_path, map_location="cpu")
    face_embedding_extractor.load_state_dict(face_embedding_extractor_state_dict, strict=True)

    noise_scheduler.init_face_model(app=face_model.app, face_helper=face_model.face_helper, handler_ante=face_model.handler_ante, face_embedding_extractor=face_embedding_extractor)

    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()
    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"],
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = XFormersAttnProcessor()
            else:
                attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    # resume the previous checkpoint
    if args.posenet_model_name_or_path is not None and args.face_encoder_model_name_or_path is not None and args.unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
        if args.posenet_model_name_or_path.endswith(".pth"):
            pose_net_state_dict = torch.load(args.posenet_model_name_or_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)
            print("successfully loading the weights of pose_net")
        else:
            print("posenet weights loading fail")
            print(1/0)
        if args.face_encoder_model_name_or_path.endswith(".pth"):
            face_encoder_state_dict = torch.load(args.face_encoder_model_name_or_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
            print("successfully loading the weights of face encoder")
        else:
            print("face_encoder weights loading fail")
            print(1/0)
        if args.unet_model_name_or_path.endswith(".pth"):
            unet_state_dict = torch.load(args.unet_model_name_or_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
            print("successfully loading the weights of unet")
        else:
            print("unet weights loading fail")
            print(1/0)

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        print("gradient_checkpointing is activated!")
        unet.enable_gradient_checkpointing()
        vae._set_gradient_checkpointing(module=vae.decoder, value=True)

    pipeline = InferenceAnimationProPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device=device, dtype=weight_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    validation_image_path = args.validation_image
    validation_image = Image.open(args.validation_image).convert('RGB')
    validation_control_images = load_images_from_folder(args.validation_control_folder, width=args.width, height=args.height)

    num_frames = len(validation_control_images)
    face_model.face_helper.clean_all()
    validation_face = cv2.imread(validation_image_path)
    validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR)
    validation_image_face_info = face_model.app.get(validation_image_bgr)
    if len(validation_image_face_info) > 0:
        validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        validation_image_id_ante_embedding = validation_image_face_info['embedding']
    else:
        validation_image_id_ante_embedding = None

    if validation_image_id_ante_embedding is None:
        face_model.face_helper.read_image(validation_image_bgr)
        face_model.face_helper.get_face_landmarks_5(only_center_face=True)
        face_model.face_helper.align_warp_face()

        if len(face_model.face_helper.cropped_faces) == 0:
            validation_image_id_ante_embedding = np.zeros((512,))
        else:
            validation_image_align_face = face_model.face_helper.cropped_faces[0]
            print('fail to detect face using insightface, extract embedding on align face')
            validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)

    # refined reference face embedding extraction: [face loss], [background content loss], [face loss, background content loss]
    # refine_reference_face_embeddings = copy.deepcopy(validation_image_id_ante_embedding)

    reference_face_mask = imageio.imread(args.validation_image_mask).astype(np.float32)
    reference_face_mask = reference_face_mask / 255
    reference_face_mask = torch.from_numpy(reference_face_mask).float()
    reference_face_mask = rearrange(reference_face_mask[:, :, None], 'h w c -> c h w')
    reference_y_indices, reference_x_indices = torch.where(reference_face_mask[0] == 1)
    if len(reference_y_indices) > 0 and len(reference_x_indices) > 0:
        reference_y_min = reference_y_indices.min().item()
        reference_y_max = reference_y_indices.max().item()
        reference_x_min = reference_x_indices.min().item()
        reference_x_max = reference_x_indices.max().item()
    else:
        print(1/0)

    target_face_mask_list = []
    target_face_mask_files = os.listdir(args.validation_control_mask_folder)
    mask_png_files = [f for f in target_face_mask_files if f.endswith('.png')]
    mask_png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in mask_png_files:
        target_mask_position = []
        target_mask = imageio.imread(os.path.join(args.validation_control_mask_folder, filename)).astype(np.float32)
        target_mask = target_mask / 255
        target_mask = torch.from_numpy(target_mask).float()
        target_mask = rearrange(target_mask[:, :, None], 'h w c -> c h w')
        target_y_indices, target_x_indices = torch.where(target_mask[0] == 1)
        if len(target_y_indices) > 0 and len(target_x_indices) > 0:
            target_y_min = target_y_indices.min().item()
            target_y_max = target_y_indices.max().item()
            target_x_min = target_x_indices.min().item()
            target_x_max = target_x_indices.max().item()
            target_mask_position.append(target_y_min)
            target_mask_position.append(target_y_max)
            target_mask_position.append(target_x_min)
            target_mask_position.append(target_x_max)
        else:
            print(1/0)
        target_face_mask_list.append(target_mask_position)

    refine_reference_face = cv2.imread(validation_image_path)
    refine_reference_face = cv2.cvtColor(refine_reference_face, cv2.COLOR_BGR2RGB)
    refine_reference_face = np.transpose(refine_reference_face, (2, 0, 1))
    refine_reference_face = torch.from_numpy(refine_reference_face).unsqueeze(0).float()
    refine_reference_face.div_(255).sub_(0.5).div_(0.5)
    face_embedding_extractor.eval()
    face_embedding_extractor.to(device='cuda')
    refine_reference_face = refine_reference_face[:, :, reference_y_min:reference_y_max + 1, reference_x_min:reference_x_max + 1]
    refine_reference_face = F.interpolate(refine_reference_face, size=(112, 112), mode='bilinear', align_corners=False).to(device='cuda')
    refine_reference_face_embeddings = face_embedding_extractor(refine_reference_face)


    video_frames = pipeline(
        image=validation_image,
        image_pose=validation_control_images,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        tile_size=num_frames,
        tile_overlap=args.frames_overlap,
        decode_chunk_size=4,
        motion_bucket_id=127.,
        fps=7,
        min_guidance_scale=args.guidance_scale,
        max_guidance_scale=args.guidance_scale,
        noise_aug_strength=args.noise_aug_strength,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pil",
        validation_image_id_ante_embedding=validation_image_id_ante_embedding,
        refine_reference_face_embeddings=refine_reference_face_embeddings,
        target_face_mask_list=target_face_mask_list,
        reference_face_mask=reference_face_mask,
        start_refine_step=args.start_refine_step,
        end_refine_step=args.end_refine_step,
        num_optimization_iter=args.num_optimization_iter,
    ).frames[0]

    out_file = os.path.join(
        args.output_dir,
        f"animation_video.mp4",
    )
    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)
    export_to_gif(video_frames, out_file, 8)



