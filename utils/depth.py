import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import cv2
import numpy as np
from transformers.utils import logging as hf_logging

# Set the Transformers logger to error level to suppress warnings
hf_logging.set_verbosity_error()

def get_depth_prediction(image, type):
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf", depth_estimation_type=type)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    depth = outputs['predicted_depth'].permute(1,2,0).squeeze(2)
    depth = depth.cpu().numpy()
    return depth


def depth_to_metric(depth, new_min=0.3, new_max=0.7, gamma=1.0):
    # Get original depth predictions
    old_min = depth.min()
    old_max = depth.max()

    # Reverse depth
    depth_reversed = old_max + old_min - depth

    # Rescale to [new_min, new_max]
    depth_min = depth_reversed.min()
    depth_max = depth_reversed.max()
    normalized = (depth_reversed - depth_min) / (depth_max - depth_min)  # Normalize to [0,1]

    # Apply non-linear scaling (e.g., gamma correction)
    non_linear = normalized ** gamma

    # Scale to new metric depth range
    metric_depth = non_linear * (new_max - new_min) + new_min
    return metric_depth


def add_gaussian_noise(depth, noise_std=0.05):
    noise = np.random.normal(loc=0.0, scale=noise_std, size=depth.shape)
    noisy_depth = depth + noise
    noisy_depth = np.clip(noisy_depth, 0.0, 1.0)  # Keep values in [0, 1]
    return noisy_depth


def calc_depth(image, type:str='relative', min=0.3, max=0.7, gamma=1.0):
    assert type in ['relative', 'metric'], f'Unknown depth type: {type}'
    print(f'Calculating image depth')

    if type == 'relative':
        relative_depth = get_depth_prediction(image*255, type)
        metric_depth = depth_to_metric(relative_depth, min, max, gamma)
        depth = metric_depth
    else:
        metric_depth = get_depth_prediction(image*255, type)
        depth = metric_depth
    
    # Resize original image
    image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
    image = image.astype('float32')
    return depth, image
   


