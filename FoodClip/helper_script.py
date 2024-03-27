import torch
import torch.nn as nn
import torchvision.models as pretrained_models
from transformers import GPT2Model,BertModel
from model import ModifiedResNet, ModifiedBert
import matplotlib.pyplot as plt
from PIL import Image
import os,csv
import numpy as np

#Check how the dimensions of the input images are to transform them to an appropriate size
image_path_ordered = []
food_title_ordered = []
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(f"{root_dir}/", 'Data/Food_Ingredients.csv')  # requires `import os`
# Open the CSV file
with open(csv_path, 'r') as file:
    # Create a CSV reader
    reader = csv.reader(file)
    next(reader, None)
    
    # Iterate over each row in the CSV file
    for row in reader:
        # Get the image name from the desired column (e.g., column index 0)
        image_name = row[4]
        
        # Construct the file path to the image
        image_path = os.path.join(f"{root_dir}/", f"Data/FoodImages/FoodImages/{image_name}.jpg")
        #Add path and title to the lists
        image_path_ordered.append(image_path)
        food_title_ordered.append(row[1])

# Assuming `image_path_ordered` contains the paths to your images
image_sizes = []

for image_path in image_path_ordered:
    try:
        with Image.open(image_path) as img:
            image_sizes.append(img.size)  # img.size is a tuple (width, height)
    except:
        print(f"Error loading image {image_path}")

# Separate widths and heights for easier analysis
widths, heights = zip(*image_sizes)

# Calculate average and median of widths
width_avg = np.mean(widths)
width_median = np.median(widths)

# Calculate average and median of heights
height_avg = np.mean(heights)
height_median = np.median(heights)

print(f"Widths average: {width_avg}")
print(f"Widths median: {width_median}")
print(f"Heights average: {height_avg}")
print(f"Heights median: {height_median}")

