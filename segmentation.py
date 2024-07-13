# %%writefile deep_script.py


import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

import argparse
import sys
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser

vitmatte_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ViTMatte'))
sys.path.append(vitmatte_path)

IMG_EXT = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')
CLASS_MAP = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
             "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
             "potted plant": 16, "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20}

def trimap(probs, size, conf_threshold):
    mask = (probs > 0.05).astype(np.uint8) * 255
    pixels = 2 * size + 1
    kernel = np.ones((pixels, pixels), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    remake = np.zeros_like(mask)
    remake[dilation == 255] = 127
    remake[probs > conf_threshold] = 255
    return remake

def parse_args():
    parser = argparse.ArgumentParser(description="Trimap and Alpha Matting Integration")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory to save the output results. (required)")
    parser.add_argument("--target_class", type=str, default='person', choices=CLASS_MAP.keys(), help="Type of the foreground object.")
    parser.add_argument("--show", action='store_true', help="Use to show results.")
    parser.add_argument("--conf_threshold", type=float, default=0.95, help="Confidence threshold for the foreground object.")
    args = parser.parse_args()
    return args

def generate_trimap(input_dir, target_class, show, conf_threshold):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    trimaps_path = os.path.join(input_dir, "trimaps")
    os.makedirs(trimaps_path, exist_ok=True)
    images_list = os.listdir(input_dir)
    for filename in images_list:
        if not filename.endswith(IMG_EXT):
            continue
        input_image = cv2.imread(os.path.join(input_dir, filename))
        original_image = input_image.copy()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)['out'][0]
            output = torch.softmax(output, 0)
        output_cat = output[CLASS_MAP[target_class], ...].numpy()
        trimap_image = trimap(output_cat, 7, conf_threshold)
        trimap_filename = f'trimapping_{filename}'
        cv2.imwrite(os.path.join(trimaps_path, trimap_filename), trimap_image)
    return os.path.join(trimaps_path, trimap_filename)

def resize_image(image, max_size=512):
    width, height = image.size
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_width, new_height = width, height
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(max_size * width / height)
        else:
            new_width, new_height = width, height
    return image.resize((new_width, new_height), Image.LANCZOS)

def infer_one_image(model, input, save_dir):
    output = model(input)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    output.save(save_dir)

def init_model(model_name, checkpoint, device):
    config = 'configs/common/model.py'
    # config='/home/localuser/Downloads/mth112/project/ViTMatte/configs/common/model.py'
    cfg = LazyConfig.load(config)
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint)
    return model

def get_data(image_dir, trimap_dir):
    image = Image.open(image_dir).convert('RGB')
    resized_img = resize_image(image)
    image = F.to_tensor(resized_img).unsqueeze(0)
    trimap = Image.open(trimap_dir).convert('L')
    resized_tri = resize_image(trimap)
    trimap = F.to_tensor(resized_tri).unsqueeze(0)
    return {'image': image, 'trimap': trimap}

def cal_foreground(image_dir, alpha_dir):
    image = Image.open(image_dir).convert('RGB')
    image = resize_image(image)
    alpha = Image.open(alpha_dir).convert('L')
    alpha = resize_image(alpha)
    alpha = F.to_tensor(alpha).unsqueeze(0)
    image = F.to_tensor(image).unsqueeze(0)
    foreground = image * alpha + (1 - alpha)
    foreground = foreground.squeeze(0).permute(1, 2, 0).numpy()
    return foreground

def main(input_dir, target_class, show, conf_threshold):
    # Generate trimap
    trimap_path = generate_trimap(input_dir, target_class, show, conf_threshold)

    # Initialize alpha matting model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model(model_name='vitmatte-s', checkpoint='ViTMatte/ViTMatte_S_Com.pth', device=device)

    # Get data for alpha matting
    images_list = [f for f in os.listdir(input_dir) if f.endswith(IMG_EXT)]
    if not images_list:
        raise ValueError("No valid image files found in the input directory.")
    image_dir = os.path.join(input_dir, images_list[0])
    input_data = get_data(image_dir, trimap_path)

    # Infer alpha matte
    alpha_save_path = os.path.join(input_dir, 'results', f'alpha_matte_{os.path.basename(image_dir)}')
    os.makedirs(os.path.dirname(alpha_save_path), exist_ok=True)
    infer_one_image(model, input_data, alpha_save_path)

    # Calculate and save foreground
    fg = cal_foreground(image_dir, alpha_save_path)
    fg_save_path = os.path.join(input_dir, 'results', f'foreground_{os.path.basename(image_dir)}')
    fg_save = Image.fromarray((fg * 255).astype(np.uint8))
    fg_save.save(fg_save_path)

    if show:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(image_dir))
        plt.subplot(1, 2, 2)
        plt.imshow(Image.open(trimap_path), cmap='gray')
        plt.show()

        plt.figure(figsize=(7, 7))
        plt.imshow(np.array(Image.open(alpha_save_path)), cmap='gray')
        plt.show()

        plt.figure(figsize=(7, 7))
        plt.imshow(fg)
        plt.show()

if __name__ == "__main__":
    # Check if script is run in an interactive environment
    # if 'ipykernel' in sys.modules:
    #     args = parse_args()
    # else:
    args = parse_args()
    main(args.input_dir, args.target_class, args.show, args.conf_threshold)
