# Please make sure you have required libraries to run this program

import os
import sys
import glob

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from search import uniform_cost_search, a_star_search, split_bev_into_grid


HOME = '/home/sagar'
IMAGE_PATH = os.path.join(HOME, "drive/Stanford/UnlabeledDataset/input")
OUT_IMAGE_PATH = os.path.join(HOME, "drive/Stanford/UnlabeledDataset/output")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_b_01ec64.pth") # Get model here https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


# Search details

low_cost = 50
high_cost = 150
max_cost = 250
grid_scale = 100 # size of each cell in mm

source = (19, 9)
destination = (0, 9)

frame_count = 0


# Homography details

rows,cols = (560,1280)

# Define the source points for the perspective transformation
x1, y1 = 0, rows    # bottom left point in image
x2, y2 = 575, 0     # top left
x3, y3 = 739, 0     # top right
x4, y4 = cols, rows  # bottom right

src_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

# Define the output size for the bird's eye view image
measurements = (600, 2220)

output_size = (1900, 2220)  # Specify the desired width and height
factor = output_size[0]-measurements[0]
top_crop = 220

a1, b1 = int(factor/2), measurements[1]
a2, b2 = int(factor/2), 0
a3, b3 = measurements[0]+int(factor/2), 0
a4, b4 = measurements[0]+int(factor/2), measurements[1]

# Define the destination points for the bird's eye view image
dst_points = np.float32([[a1, b1], [a2, b2], [a3, b3], [a4, b4]])

# Compute the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
inv_matrix = cv2.getPerspectiveTransform(dst_points, src_points)


# Initialize SAM (Segment Anything Model)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.98,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

# Find images

image_paths = sorted(glob.glob(IMAGE_PATH + "/*.png"))  # Modify the file extension if necessary

for image_path in image_paths:
    # Read the image
    image = cv2.imread(image_path)
    image_bgr = image[160:720,:]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)
    
    # find carpet
    carpet_idx = 0
    count_list = []

    for idx, mask in enumerate(sam_result):
      count = np.count_nonzero(mask['segmentation'])
      count_list.append(count)
      if mask['segmentation'][540][640]:
        carpet_idx = idx
    
    trim_size = 2
    main_indices = np.argsort(-np.asarray(count_list))[:trim_size]

    result_img = 255*np.ones((rows, cols), dtype = np.uint8)
    for idx in main_indices:
      mask = sam_result[idx]['segmentation']
      if carpet_idx == idx:
        masked = low_cost * mask
        result_img += masked.astype('uint8')
      else:
        masked = high_cost * mask
        result_img += masked.astype('uint8')

    bev_image_cost = cv2.warpPerspective(result_img, matrix, output_size, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    bev_image_cost = bev_image_cost[top_crop:output_size[1],:]
    grid_size = (int(bev_image_cost.shape[0]/grid_scale), int(bev_image_cost.shape[1]/grid_scale))
    cost_grid = split_bev_into_grid(bev_image_cost, grid_size)

    path = a_star_search(cost_grid, source, destination)

    if path:
      # int_cost_grid = np.uint8(np.clip(cost_grid, 0, 255))
      # result_grid = cv2.cvtColor(int_cost_grid,cv2.COLOR_GRAY2RGB)
      # size of the image
      (H , W) = cost_grid.shape[0], cost_grid.shape[1]
      # Blank image with RGBA = (0, 0, 0, 0)
      blank_image = np.full((H, W, 4), (0, 0, 0, 0), np.uint8)
      for y in range(len(blank_image)):
        for x in range(len(blank_image[y])):
          if (y,x) == source:
            blank_image[y][x] = [0,0,255,100]
          elif (y,x) == destination:
            blank_image[y][x] = [0,255,0,100] 
          elif (y,x) in path:
            blank_image[y][x] = [255,0,0,100]

      resized_blank_grid = cv2.resize(blank_image, (grid_scale*blank_image.shape[1], grid_scale*blank_image.shape[0]), interpolation = cv2.INTER_AREA)
      expanded_blank_grid = np.zeros((output_size[1],output_size[0],4), np.uint8)
      expanded_blank_grid[top_crop:output_size[1],:] = resized_blank_grid
      resulting_blank_image = cv2.warpPerspective(expanded_blank_grid, inv_matrix, (1280,560), cv2.INTER_LINEAR, borderValue=(255, 255, 255))

      # normalize alpha channels from 0-255 to 0-1
      image_color = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2BGRA)
      alpha_background = image_color[:,:,3] / 255.0
      alpha_foreground = resulting_blank_image[:,:,3] / 255.0

      # set adjusted colors
      for color in range(0, 3):
          image_color[:,:,color] = alpha_foreground * resulting_blank_image[:,:,color] + \
              alpha_background * image_color[:,:,color] * (1 - alpha_foreground)

      # set adjusted alpha and denormalize back to 0-255
      image_color[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
      image_filename = os.path.join(OUT_IMAGE_PATH, f'frame_{frame_count:04d}.png') 
      cv2.imwrite(image_filename, image_color)
      frame_count+=1