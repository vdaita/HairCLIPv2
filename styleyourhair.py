#!/usr/bin/env python
# coding: utf-8

# Adapted from https://www.kaggle.com/code/taylorsamarel/change-python-version-on-kaggle-taylor-amarel

def setup():
    import gdown

    files = [
        "https://drive.google.com/file/d/1g8S81ZybmrF86OjvjLYJzx-wx83ZOiIw/view?usp=drive_link",
        "https://drive.google.com/file/d/1OG6t7q4PpHOoYNdP-ipoxuqYbfMSgPta/view?usp=drive_link",
        "https://drive.google.com/file/d/1c-SgUUQj0X1mIl-W-_2sMboI2QS7GzfK/view?usp=drive_link",
        "https://drive.google.com/file/d/1sa732uBfX1739MFsvtRCKWCN54zYyltC/view?usp=drive_link",
        "https://drive.google.com/file/d/1qk0ZIfA1VmrFUzDJ0g8mK8nx0WtF-5sY/view?usp=drive_link"
    ]

    filenames = [
        "ffhq.pt",
        "seg.pth",
        "shape_predictor_68_face_landmarks.dat",
        "bald_proxy.pt",
        "sketch_proxy.pt"
    ]

    for url, filename in zip(files, filenames):
        gdown.download(url, fuzzy=True, output=f"HairCLIPv2/pretrained_models/{filename}")



import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import torch
import numpy as np
import tempfile
from PIL import Image
from torchvision import transforms
from scripts.Embedding import Embedding
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.sketch_proxy import SketchProxy
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_base_models
from utils.options import Options
import PIL.Image
import scipy
import scipy.ndimage
from scipy.ndimage import correlate
import dlib
from pathlib import Path
from matplotlib.pyplot import imshow
import cv2
from matplotlib import pyplot as plt

"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""


def find_bounding_box(large_image, small_image, threshold):
    large_image = np.array(large_image)
    small_image = np.array(small_image)
    
    fig = plt.figure(figsize=(10, 7)) 
    fig.add_subplot(2, 1, 1) 
    
    plt.imshow(large_image)
    plt.title("Large")
    
    fig.add_subplot(2, 1, 2)
    
    plt.imshow(small_image)
    plt.title("Small")

    # Use template matching to find the location of the small image within the large image
    res = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    print("Found in location: ", minVal, maxVal, minLoc, maxLoc)
    
    (top, left) = maxLoc
    
    # Get the bottom right corner of the bounding box
    bottom = top + small_image.shape[1]
    right = left + small_image.shape[0]
    
    print("Making the shape, based on the small image: ", small_image.shape[0], small_image.shape[1])
    
    # Return the bounding box
    return top, left, bottom, right

def get_landmark(filepath,predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    filepath = Path(filepath)
    print(f"{filepath.name}: Number of faces detected: {len(dets)}")
    shapes = [predictor(img, d) for k, d in enumerate(dets)]

    lms = [np.array([[tt.x, tt.y] for tt in shape.parts()]) for shape in shapes]

    return lms


def align_face(filepath,predictor):
    """
    :param filepath: str
    :return: list of PIL Images
    """

    lms = get_landmark(filepath,predictor)
    imgs = []
    bounding_coords = []
    for lm in lms:
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        img = PIL.Image.open(filepath)
        original_img = img

        output_size = 1024
        # output_size = 256
        transform_size = 4096
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]


        generated_bounding_coords = find_bounding_box(original_img, img, 0.9)
        print(generated_bounding_coords)
        print(np.asarray(original_img.crop(generated_bounding_coords)).shape, np.asarray(img).shape)
        
        display_image_list([np.asarray(original_img.crop(generated_bounding_coords)), np.asarray(img)])
        
        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
                
        
        # Save aligned image.
        imgs.append(img)
        bounding_coords.append(generated_bounding_coords)
    return imgs[0], bounding_coords[0]

predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")

def process_image(src_filepath):
    opts = Options().parse(jupyter=True)
    src_name = 'person'# source image name you want to edit

    image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    g_ema, mean_latent_code, seg = load_base_models(opts)
    ii2s = Embedding(opts, g_ema, mean_latent_code[0,0])
    
    # TODO: crop out the file and get the right image
    try:
        aligned_image, bounding_coords = align_face(src_filepath, predictor)
    except:
        return "Face not found."
    
    print("Received aligned images and bounding coords: ", aligned_image, bounding_coords)
    
    dirpath = tempfile.mkdtemp()
    src_latent_filepath = os.path.join(dirpath, "src.npz")
    aligned_image_filepath = os.path.join(dirpath, "image.jpg")
    aligned_image.save(aligned_image_filepath)
    

    if not os.path.isfile(src_latent_filepath):
        inverted_latent_w_plus, inverted_latent_F = ii2s.invert_image_in_FS(image_path=aligned_image_filepath)
        save_latent_path = src_latent_filepath
        np.savez(save_latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                    latent_F=inverted_latent_F.detach().cpu().numpy())
    src_latent = torch.from_numpy(np.load(src_latent_filepath)['latent_in']).cuda()
    src_feature = torch.from_numpy(np.load(src_latent_filepath)['latent_F']).cuda()
    src_image = image_transform(Image.open(aligned_image_filepath).convert('RGB')).unsqueeze(0).cuda()
    input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

    bald_proxy = BaldProxy(g_ema, opts.bald_path)
    text_proxy = TextProxy(opts, g_ema, seg, mean_latent_code)
    ref_proxy = RefProxy(opts, g_ema, seg, ii2s)
    sketch_proxy = SketchProxy(g_ema, mean_latent_code, opts.sketch_path)
    color_proxy = ColorProxy(opts, g_ema, seg)

    edited_hairstyle_img = src_image

    def hairstyle_editing(global_cond=None, local_sketch=False, paint_the_mask=False, \
                          src_latent=src_latent, src_feature=src_feature, input_mask=input_mask, src_image=src_image, \
                            latent_global=None, latent_local=None, latent_bald=None, local_blending_mask=None, painted_mask=None):
        if paint_the_mask:
            modified_mask = painting_mask(input_mask)
            input_mask = torch.from_numpy(modified_mask).unsqueeze(0).cuda().long().clone().detach()
            vis_modified_mask = vis_seg(modified_mask)
            display_image_list([src_image, vis_modified_mask])
            painted_mask = input_mask

        if local_sketch:
            latent_local, local_blending_mask, visual_local_list = sketch_proxy(input_mask)
            display_image_list(visual_local_list)

        if global_cond is not None:
            assert isinstance(global_cond, str)
            latent_bald, visual_bald_list = bald_proxy(src_latent)
            display_image_list(visual_bald_list)

            if global_cond.endswith('.jpg') or global_cond.endswith('.png'):
                latent_global, visual_global_list = ref_proxy(global_cond, src_image, painted_mask=painted_mask)
            else:
                latent_global, visual_global_list = text_proxy(global_cond, src_image, from_mean=True, painted_mask=painted_mask)
            display_image_list(visual_global_list)

        src_feature, edited_hairstyle_img = hairstyle_feature_blending(g_ema, seg, src_latent, src_feature, input_mask, latent_bald=latent_bald,\
                                                    latent_global=latent_global, latent_local=latent_local, local_blending_mask=local_blending_mask)
        return src_feature, edited_hairstyle_img
    
    src_image, edited_image = hairstyle_editing(src_image=src_image, global_cond="buzzcut.jpg")
    
    edited_image_pil = process_display_input(edited_image)
    edited_image_pil = Image.fromarray(np.uint8(edited_image_pil), 'RGB')

    return edited_image_pil
