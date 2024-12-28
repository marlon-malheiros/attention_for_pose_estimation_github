import os
import sys
import time
import json
import random
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToTensor, functional as TF

from pycocotools.coco import COCO

"""
This file provides utilities for working with COCO-style annotations and image data, particularly focusing on person detection and keypoint visualization.

Functions and Classes Overview
------------------------------

plot_images_with_keypoints(image_dir, annotations_dict):
    Plots each image from the given annotations dictionary, overlaying keypoints categorized by their visibility values (v=1 and v=2).
    Also displays basic image metadata such as dimensions and total keypoints.

create_annotation_dict(coco_json_path, image_dir):
    Creates a dictionary mapping image filenames to their annotations, including image IDs and annotation details.
    Useful for quick lookups of annotations using just filenames.

class PersonDetectionDataset(Dataset):
    A PyTorch Dataset implementation that:
    - Loads images and corresponding COCO annotations for person detection tasks.
    - Returns (image, target) pairs, where target includes bounding boxes, labels, and image metadata.
    - Supports optional image transformations.

add_symmetric_padding_to_box(box, padding_percentage, image_width, image_height):
    Expands a given bounding box symmetrically by a percentage of the image dimensions.
    Ensures the padded box remains within image boundaries.

plot_image_with_bboxes(easy_imgs, n, image_dir):
    Plots the nth image from the provided dictionary along with its annotated bounding boxes.
    Useful for visual checking of annotation integrity.

filter_images_by_keypoints(img_ids_bboxes, annotations_file):
    Filters a list of (image_id, bbox) tuples based on whether all visible keypoints fall inside the specified bounding box.
    Returns two lists: one with valid samples (all keypoints inside) and one with invalid samples (some keypoints outside).

"""


def plot_images_with_keypoints(image_dir, annotations_dict):
    """
    Plot all images in the annotations dictionary, showing keypoints and metadata.
    
    Args:
        image_dir (str): Directory containing the image files.
        annotations_dict (dict): Dictionary where keys are filenames and values are annotations.
    """
    for filename, data in annotations_dict.items():
        image_path = os.path.join(image_dir, filename)
        
        # Open the image
        if not os.path.exists(image_path):
            print(f"Image {filename} not found in {image_dir}. Skipping...")
            continue
        
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        # Get annotations
        image_id = data.get("image_id")
        annotations = data.get("annotations", [])
        
        # Count keypoints and categorize by visibility
        total_keypoints = 0
        keypoints_v0 = 0
        keypoints_v1 = []
        keypoints_v2 = []
        
        for ann in annotations:
            keypoints = ann.get("keypoints", [])
            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i:i + 3]
                if v == 0:
                    keypoints_v0 += 1
                elif v == 1:
                    keypoints_v1.append((x, y))
                elif v == 2:
                    keypoints_v2.append((x, y))
            total_keypoints += len(keypoints) // 3
        
        # Plot the image
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)
        ax.set_title(
            f"Filename: {filename}\n"
            f"Image Dimensions: {image_width}x{image_height}\n"
            f"Total Keypoints: {total_keypoints}, v=0: {keypoints_v0}, v=1: {len(keypoints_v1)}, v=2: {len(keypoints_v2)}"
        )
        
        # Plot keypoints
        for x, y in keypoints_v1:
            ax.plot(x, y, 'yo', label='v=1 (yellow)' if 'v=1 (yellow)' not in ax.get_legend_handles_labels()[1] else "")  # Yellow
        for x, y in keypoints_v2:
            ax.plot(x, y, 'go', label='v=2 (green)' if 'v=2 (green)' not in ax.get_legend_handles_labels()[1] else "")  # Green

        
        plt.show()
        
def create_annotation_dict(coco_json_path, image_dir):
    """
    Create a dictionary with filenames as keys and their annotations as values.
    
    Args:
        coco_json_path (str): Path to the COCO annotations JSON file.
        image_dir (str): Directory containing the image files.
        
    Returns:
        dict: A dictionary where keys are filenames and values are the annotations for each file.
    """
    # Initialize COCO object
    coco = COCO(coco_json_path)
    
    # Initialize empty dictionary
    annotations_dict = {}
    
    # Get all image filenames in the directory
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Iterate over each image filename and get annotations
    for filename in image_filenames:
        # Get annotations using the function
        images = coco.loadImgs([img['id'] for img in coco.dataset['images'] if img['file_name'] == filename])
        if not images:
            annotations = {"image_id": None, "annotations": []}
        else:
            image_id = images[0]['id']
            annotations = {
                "image_id": image_id,
                "annotations": coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            }
        
        # Add to dictionary
        annotations_dict[filename] = annotations
    
    return annotations_dict

class PersonDetectionDataset(Dataset):
    def __init__(self, img_ids, annotations_file, image_dir, transform=None):
        """
        Args:
            img_ids (list): List of filtered image IDs (img_filtradas).
            annotations_file (str): Path to the COCO annotations JSON file.
            image_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to apply to each image.
        """
        self.img_ids = img_ids
        self.coco = COCO(annotations_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Preload image metadata for each image ID
        self.image_metadata = {
            image_id: self.coco.loadImgs(image_id)[0] for image_id in self.img_ids
        }

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # Get the image ID and associated metadata
        image_id = self.img_ids[idx]
        img_info = self.image_metadata[image_id]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get annotations for the image ID
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[self.coco.getCatIds(catNms=['person'])[0]], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(1)  # Label "1" for "person"
        
        # Convert lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Create target dictionary with image metadata (e.g., width, height, and filename)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "width": img_info["width"],
            "height": img_info["height"],
            "file_name": img_info["file_name"]  # Adding the file name here
        }
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
# def add_percentage_padding_to_box(box, padding_percentage, image_width, image_height):
#     """
#     Adds padding to a bounding box as a percentage of its dimensions, keeping it within image bounds.
    
#     Args:
#         box: Bounding box coordinates as [x_min, y_min, x_max, y_max].
#         padding_percentage: Padding as a percentage of the box dimensions (e.g., 15 for 15%).
#         image_width: Width of the image.
#         image_height: Height of the image.
    
#     Returns:
#         Padded bounding box coordinates.
#     """
#     x_min, y_min, x_max, y_max = box
    
#     # Calculate the width and height of the bounding box
#     box_width = x_max - x_min
#     box_height = y_max - y_min
    
#     # Calculate padding based on the percentage
#     x_padding = box_width * (padding_percentage / 100)
#     y_padding = box_height * (padding_percentage / 100)
    
#     # Apply padding and ensure the box stays within image bounds
#     x_min = max(0, x_min - x_padding)
#     y_min = max(0, y_min - y_padding)
#     x_max = min(image_width, x_max + x_padding)
#     y_max = min(image_height, y_max + y_padding)
    
#     return [x_min, y_min, x_max, y_max]
def add_symmetric_padding_to_box(box, padding_percentage, image_width, image_height):
    """
    Adds symmetric padding to a bounding box as a percentage of the image dimensions.

    Args:
        box: Bounding box coordinates as [x_min, y_min, x_max, y_max].
        padding_percentage: Padding as a percentage of the box dimensions.
        image_width: Width of the image.
        image_height: Height of the image.

    Returns:
        Padded bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, x_max, y_max = box

    # Calculate the padding based on image dimensions
    x_padding = image_width * (padding_percentage / 100)
    y_padding = image_height * (padding_percentage / 100)

    # Apply padding symmetrically and clamp within bounds
    padded_x_min = max(0, x_min - x_padding)
    padded_y_min = max(0, y_min - y_padding)
    padded_x_max = min(image_width, x_max + x_padding)
    padded_y_max = min(image_height, y_max + y_padding)

    return [int(padded_x_min), int(padded_y_min), int(padded_x_max), int(padded_y_max)]


    

def plot_image_with_bboxes(easy_imgs, n, image_dir):
    """
    Plots an image with its annotated bounding boxes.

    Parameters:
    - easy_imgs (dict): Dictionary containing image annotations.
    - n (int): Index of the image in the dictionary to plot.
    - image_dir (str): Path to the directory containing the images.

    Returns:
    - None: Displays the image with bounding boxes.
    """
    # Get the image ID and annotations from the nth element of the dictionary
    img_key = list(easy_imgs.keys())[n]
    img_data = easy_imgs[img_key]
    
    # Load the image
    img_path = f"{image_dir}/{img_key}"  # Adjust this path if necessary
    img = Image.open(img_path).convert("RGB")
    
    # Get the bounding boxes
    bboxes = [annotation['bbox'] for annotation in img_data['annotations']]
    
    # Create a plot
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"Image ID: {img_key}")
    
    # Plot each bounding box
    for bbox in bboxes:
        # COCO format bbox: [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.show()

def filter_images_by_keypoints(img_ids_bboxes, annotations_file):
    """
    Filter images based on whether all keypoints are inside the bounding box.
    
    Args:
        img_ids_bboxes (list): List of tuples with (image_id, bounding_box).
        annotations_file (str): Path to the COCO annotations JSON file.

    Returns:
        valid_samples (list): List of tuples (image_id, bounding_box) where all keypoints are inside the bbox.
        invalid_samples (list): List of tuples (image_id, bounding_box) where some keypoints are outside the bbox.
    """
    # Initialize COCO API for annotation loading
    coco = COCO(annotations_file)
    
    # Lists to store valid and invalid samples
    valid_samples = []
    invalid_samples = []

    for image_id, bbox in img_ids_bboxes:
        # Get annotations for the current image ID
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=[coco.getCatIds(catNms=['person'])[0]], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        # Check if there are keypoints available; if not, skip without marking as invalid
        if not anns or 'keypoints' not in anns[0]:
            continue  # Skip to the next image if there are no keypoints

        keypoints = anns[0]['keypoints']  # Assuming single person per image, format (x, y, v) for each keypoint
        keypoints = [keypoints[i:i + 3] for i in range(0, len(keypoints), 3)]  # Reshape to list of (x, y, v)

        # Extract bounding box coordinates correctly
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

        # Check if all keypoints are inside the bounding box
        all_inside = True
        for x, y, v in keypoints:
            if v == 2:  # Only consider visible keypoints
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    all_inside = False
                    break

        # Categorize based on keypoints' positions
        if all_inside:
            valid_samples.append((image_id, bbox))
        else:
            invalid_samples.append((image_id, bbox))

    return valid_samples, invalid_samples


def extract_point_from_heatmap_numpy(heatmap):
    """
    Extracts the point of maximum intensity from a heatmap.

    Args:
        heatmap (torch.Tensor): A single heatmap tensor of shape (h, w).

    Returns:
        Tuple: (x, y) coordinates of the maximum intensity point.
    """
    max_index = np.argmax(heatmap)
    y, x = divmod(max_index.item(), heatmap.shape[1])  # Convert flat index to (row, col)
    return x, y


def calculate_distance(point1, point2):
    """
    Calculates Euclidean distance between two points.

    Args:
        point1 (tuple): (x1, y1) coordinates.
        point2 (tuple): (x2, y2) coordinates.

    Returns:
        float: Euclidean distance between the points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    """
    2D Gaussian function.
    """
    return amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))


def fit_gaussian_to_heatmap(heatmap):
    """
    Fits a 2D Gaussian to a heatmap and returns the estimated center.

    Args:
        heatmap (np.ndarray): Heatmap array of shape (h, w).

    Returns:
        Tuple: (x, y) coordinates of the estimated center.
    """
    h, w = heatmap.shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Flatten the grid and heatmap
    x_data = x_grid.ravel()
    y_data = y_grid.ravel()
    z_data = heatmap.ravel()

    # Initial guesses for Gaussian parameters
    x0_guess = np.argmax(heatmap) % w
    y0_guess = np.argmax(heatmap) // w
    amplitude_guess = np.max(heatmap)
    sigma_guess = 2.0

    initial_guess = (x0_guess, y0_guess, sigma_guess, sigma_guess, amplitude_guess)

    # Fit Gaussian
    try:
        params, _ = curve_fit(
            lambda xy, x0, y0, sigma_x, sigma_y, amplitude: gaussian_2d(
                xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude
            ),
            (x_data, y_data),
            z_data,
            p0=initial_guess,
            bounds=(
                [0, 0, 0, 0, 0],
                [w, h, np.inf, np.inf, np.inf],
            ),
        )
        x0, y0, _, _, _ = params
    except RuntimeError:
        # Fallback to argmax if fitting fails
        y0, x0 = divmod(np.argmax(heatmap), w)

    return x0, y0


def plot_keypoint_heatmaps(
    predicted_heatmaps, 
    ground_truth_heatmaps, 
    ground_truth_visibility, 
    keypoint_names, 
    input_image=None, 
    point_extraction_method="argmax", 
    scale_to_input_image=False
):
    """
    Plots the predicted and ground truth keypoints on their respective heatmaps.

    Args:
        predicted_heatmaps (numpy.ndarray): Predicted heatmaps of shape (17, h, w).
        ground_truth_heatmaps (numpy.ndarray): Ground truth heatmaps of shape (17, h, w).
        ground_truth_visibility (list): Visibility flags for the keypoints (0, 1, or 2).
        keypoint_names (list): List of keypoint names for annotation.
        input_image (numpy.ndarray, optional): Input image to overlay keypoints on, scaled to (h, w).
        point_extraction_method (str): Method to extract points from heatmap ("argmax" or "gaussian").
        scale_to_input_image (bool): Whether to scale points to the input image size.
    """
    # Set up the figure and axes
    fig, axes = plt.subplots(
        5, 4,  # 5 rows, 4 columns
        figsize=(20, 25),  # Adjust figure size
        gridspec_kw={"wspace": 0.05, "hspace": 0.3}  # Minimize spacing
    )
    axes = axes.flatten()  # Flatten axes for iteration

    # Choose the appropriate extraction function
    if point_extraction_method == "argmax":
        extract_point_from_heatmap = extract_point_from_heatmap_numpy  # Assuming it's already defined
    elif point_extraction_method == "gaussian":
        extract_point_from_heatmap = fit_gaussian_to_heatmap  # Assuming it's already defined
    else:
        raise ValueError(f"Unknown point extraction method: {point_extraction_method}")

    for k, ax in enumerate(axes):
        if k >= 17:  # If we exceed the number of keypoints, hide unused subplots
            ax.axis("off")
            continue

        # Extract visibility
        visibility = ground_truth_visibility[k]
        keypoint_name = keypoint_names[k]

        # Visual distinction for visibility
        if visibility == 0:
            visibility_label = "Not Annotated (v=0)"
            color = "gray"
            background = np.zeros(predicted_heatmaps[k].shape)  # Black background
            cmap = "Greys"  # Black-and-white colormap
        elif visibility == 1:
            visibility_label = "Occluded (v=1)"
            color = "orange"
            background = np.zeros(predicted_heatmaps[k].shape)  # Black background
            cmap = "Greys"
        elif visibility == 2:
            visibility_label = "Visible (v=2)"
            color = "green"
            background = predicted_heatmaps[k]  # Predicted heatmap
            cmap = "viridis"
        else:
            visibility_label = "Unknown"
            color = "black"
            background = np.zeros(predicted_heatmaps[k].shape)
            cmap = "Greys"

        # Extract predicted and ground truth points
        pred_heatmap = predicted_heatmaps[k]
        gt_heatmap = ground_truth_heatmaps[k]

        pred_point = extract_point_from_heatmap(pred_heatmap)
        gt_point = extract_point_from_heatmap(gt_heatmap)

        # Optionally scale points to input image dimensions
        if scale_to_input_image and input_image is not None:
            scale_x = input_image.shape[1] / pred_heatmap.shape[1]  # Width scaling factor
            scale_y = input_image.shape[0] / pred_heatmap.shape[0]  # Height scaling factor
            pred_point = (pred_point[0] * scale_x, pred_point[1] * scale_y)
            gt_point = (gt_point[0] * scale_x, gt_point[1] * scale_y)

        # Calculate Euclidean distance
        distance = calculate_distance(pred_point, gt_point)

        # Use the input image as the background if provided
        if input_image is not None:
            ax.imshow(input_image, cmap="gray")  # Use input image as the background
        else:
            ax.imshow(background, cmap=cmap)  # Display heatmap or black background

        # Plot keypoints
        if visibility == 2:            
            ax.scatter(pred_point[0], pred_point[1], color="red", label=f"Prediction ({pred_point[0]:.1f}, {pred_point[1]:.1f})")
            ax.scatter(gt_point[0], gt_point[1], color=color, label=f"Ground Truth ({gt_point[0]:.1f}, {gt_point[1]:.1f})")          

        # Add title with coordinates
        ax.set_title(
            f"Keypoint {k+1} ({keypoint_name}):\n"
            f"Distance = {distance:.2f} | Visibility: {visibility_label}",
            fontsize=10
        )
        ax.legend(fontsize=8)
        ax.axis("off")

    # Ensure proper layout
    plt.tight_layout()
    plt.show()
    
    
def scale_points_to_input_image(keypoints, heatmap_shape, input_image_shape):
    """
    Scales keypoints from heatmap space to input image space.

    Args:
        keypoints (list of tuples): Keypoints in heatmap space [(x, y, ...)].
        heatmap_shape (tuple): Shape of the heatmap (height, width).
        input_image_shape (tuple): Shape of the input image (height, width).

    Returns:
        list of tuples: Scaled keypoints [(x, y, ...)] in input image space.
    """
    scale_x = input_image_shape[1] / heatmap_shape[1]
    scale_y = input_image_shape[0] / heatmap_shape[0]

    scaled_keypoints = []
    for x, y, *rest in keypoints:
        scaled_keypoints.append((x * scale_x, y * scale_y, *rest))

    return scaled_keypoints


def scale_keypoints_original_img_from_heatmaps(
    predicted_heatmaps, 
    paddings_input_img, 
    resized_size, 
    bbox_crop, 
    heatmap_size, 
    input_img_size,
    visibility,
    point_extraction_method="argmax"
):
    """
    Scales keypoints extracted from heatmaps back to the original image size via input image.

    Args:
        predicted_heatmaps (numpy.ndarray): Heatmaps of shape (num_keypoints, heatmap_h, heatmap_w).
        paddings_input_img (tuple): Tuple of paddings (pad_left, pad_top, pad_right, pad_bottom).
        resized_size (tuple): Dimensions of the resized image before padding (width, height).
        bbox_crop (list): Bounding box defining the crop (x_min, y_min, x_max, y_max).
        heatmap_size (tuple): Dimensions of the heatmap (width, height).
        input_img_size (tuple): Dimensions of the input image (height, width).
        point_extraction_method (str): Method to extract points from heatmap ("argmax" or "gaussian").

    Returns:
        list: Keypoints scaled back to the original image size.
    """
    # Step 1: Extract keypoints from heatmaps
    if point_extraction_method == "argmax":
        extract_point_from_heatmap = extract_point_from_heatmap_numpy  # Assuming defined
    elif point_extraction_method == "gaussian":
        extract_point_from_heatmap = fit_gaussian_to_heatmap  # Assuming defined
    else:
        raise ValueError(f"Unknown point extraction method: {point_extraction_method}")

    keypoints = []
    for heatmap_n in range(len(predicted_heatmaps)):
        heatmap = predicted_heatmaps[heatmap_n]
        keypoints.append((*extract_point_from_heatmap(heatmap), visibility[heatmap_n]))
    
    # Step 2: Scale keypoints from heatmap to input image size
    heatmap_height, heatmap_width = heatmap_size
    input_height, input_width = input_img_size

    scale_x = input_width / heatmap_width  # Scaling factor for width
    scale_y = input_height / heatmap_height  # Scaling factor for height

    keypoints_input_img = []
    for x_heatmap, y_heatmap, v in keypoints:
        x_input = x_heatmap * scale_x
        y_input = y_heatmap * scale_y
        keypoints_input_img.append((x_input, y_input, v))

    # Step 3: Remove the padding
    pad_left, pad_top, pad_right, pad_bottom = paddings_input_img

    kps_no_pad = []
    for x, y, v in keypoints_input_img:
        x_no_pad = x - pad_left
        y_no_pad = y - pad_top
        kps_no_pad.append((x_no_pad, y_no_pad, v))

    # Step 4: Undo resizing
    resized_width, resized_height = resized_size
    crop_width = bbox_crop[2] - bbox_crop[0]  # x_max - x_min
    crop_height = bbox_crop[3] - bbox_crop[1]  # y_max - y_min

    # Scaling factors
    scale_x = crop_width / resized_width  # Horizontal scaling
    scale_y = crop_height / resized_height  # Vertical scaling

    kps_crop_size = []
    for x_no_pad, y_no_pad, v in kps_no_pad:
        x_original = x_no_pad * scale_x
        y_original = y_no_pad * scale_y
        kps_crop_size.append((x_original, y_original, v))

    # Step 5: Undo cropping
    kps_original_size = []
    for x_cropped, y_cropped, v in kps_crop_size:
        x_original = x_cropped + bbox_crop[0]  # Add x_min
        y_original = y_cropped + bbox_crop[1]  # Add y_min
        kps_original_size.append((x_original, y_original, v))

    return kps_original_size


def plot_keypoints_with_distances(
    predicted_keypoints, 
    ground_truth_keypoints, 
    ground_truth_visibility, 
    keypoint_names, 
    input_image=None
):
    """
    Plots the predicted and ground truth keypoints on the input image with distance values.

    Args:
        predicted_keypoints (numpy.ndarray): Predicted keypoints of shape (17, 2) for (x, y) coordinates.
        ground_truth_keypoints (numpy.ndarray): Ground truth keypoints of shape (17, 2) for (x, y) coordinates.
        ground_truth_visibility (list): Visibility flags for the keypoints (0, 1, or 2).
        keypoint_names (list): List of keypoint names for annotation.
        input_image (numpy.ndarray, optional): Input image to overlay keypoints on.
    """
    # Set up the figure and axes
    fig, axes = plt.subplots(
        5, 4,  # 5 rows, 4 columns
        figsize=(20, 25),  # Adjust figure size
        gridspec_kw={"wspace": 0.05, "hspace": 0.3}  # Minimize spacing
    )
    axes = axes.flatten()  # Flatten axes for iteration

    for k, ax in enumerate(axes):
        if k >= 17:  # If we exceed the number of keypoints, hide unused subplots
            ax.axis("off")
            continue

        # Extract visibility
        visibility = ground_truth_visibility[k]
        keypoint_name = keypoint_names[k]

        # Visual distinction for visibility
        if visibility == 0:
            visibility_label = "Not Annotated (v=0)"
            color = "gray"
        elif visibility == 1:
            visibility_label = "Occluded (v=1)"
            color = "orange"
        elif visibility == 2:
            visibility_label = "Visible (v=2)"
            color = "green"
        else:
            visibility_label = "Unknown"
            color = "black"

        # Extract predicted and ground truth points
        pred_point = predicted_keypoints[k]
        gt_point = ground_truth_keypoints[k]

        # Calculate Euclidean distance
        distance = np.sqrt((pred_point[0] - gt_point[0])**2 + (pred_point[1] - gt_point[1])**2)

        # Use the input image as the background if provided
        if input_image is not None:
            ax.imshow(input_image, cmap="gray")  # Use input image as the background
        else:
            ax.imshow(np.zeros((224, 224)), cmap="gray")  # Default background if no image is provided

        # Plot keypoints
        if visibility == 2:  # Only plot visible keypoints
            ax.scatter(pred_point[0], pred_point[1], color="red", label=f"Prediction ({pred_point[0]:.1f}, {pred_point[1]:.1f})")
            ax.scatter(gt_point[0], gt_point[1], color=color, label=f"Ground Truth ({gt_point[0]:.1f}, {gt_point[1]:.1f})")

        # Add title with coordinates
        ax.set_title(
            f"Keypoint {k+1} ({keypoint_name}):\n"
            f"Distance = {distance:.2f} | Visibility: {visibility_label}",
            fontsize=10
        )
        ax.legend(fontsize=8)
        ax.axis("off")

    # Ensure proper layout
    plt.tight_layout()
    plt.show()
    

def create_distance_table(
    predicted_heatmaps, 
    ground_truth_heatmaps, 
    ground_truth_visibility, 
    keypoint_names, 
    point_extraction_method="argmax", 
    scale_to_input_image=False, 
    input_image=None
):
    """
    Computes the Euclidean distances between predicted and ground truth keypoints 
    and returns a table of the results.
    
    Args:
        predicted_heatmaps (numpy.ndarray): Predicted heatmaps of shape (17, h, w).
        ground_truth_heatmaps (numpy.ndarray): Ground truth heatmaps of shape (17, h, w).
        ground_truth_visibility (list): Visibility flags for the keypoints (0, 1, or 2).
        keypoint_names (list): List of keypoint names for annotation.
        point_extraction_method (str): Method to extract points from heatmap ("argmax" or "gaussian").
        scale_to_input_image (bool): Whether to scale points to the input image size.
        input_image (numpy.ndarray, optional): Input image to scale points to (should be (H, W, C) or (H, W)).
        
    Returns:
        pd.DataFrame: A DataFrame containing columns for keypoint index, name, visibility, 
                      predicted coordinates, ground truth coordinates, and distance.
    """
    # Choose the appropriate extraction function
    if point_extraction_method == "argmax":
        extract_point_from_heatmap = extract_point_from_heatmap_numpy  # Assuming defined elsewhere
    elif point_extraction_method == "gaussian":
        extract_point_from_heatmap = fit_gaussian_to_heatmap  # Assuming defined elsewhere
    else:
        raise ValueError(f"Unknown point extraction method: {point_extraction_method}")

    results = []

    for k in range(len(keypoint_names)):
        keypoint_name = keypoint_names[k]
        visibility = ground_truth_visibility[k]

        # Extract predicted and ground truth points
        pred_heatmap = predicted_heatmaps[k]
        gt_heatmap = ground_truth_heatmaps[k]

        pred_point = extract_point_from_heatmap(pred_heatmap)
        gt_point = extract_point_from_heatmap(gt_heatmap)

        # Optionally scale points to input image dimensions
        if scale_to_input_image and input_image is not None:
            scale_x = input_image.shape[1] / pred_heatmap.shape[1]  # Width scaling factor
            scale_y = input_image.shape[0] / pred_heatmap.shape[0]  # Height scaling factor
            pred_point = (pred_point[0] * scale_x, pred_point[1] * scale_y)
            gt_point = (gt_point[0] * scale_x, gt_point[1] * scale_y)

        # Calculate Euclidean distance
        distance = calculate_distance(pred_point, gt_point)

        # Store results
        results.append({            
            "Name": keypoint_name,
            "V": int(visibility),
            # "Predicted (x, y)": f"({pred_point[0]:.2f}, {pred_point[1]:.2f})",
            # "GT (x, y)": f"({gt_point[0]:.2f}, {gt_point[1]:.2f})",
            "Distance": f"{distance:.2f}"
        })

    df = pd.DataFrame(results)
    return df