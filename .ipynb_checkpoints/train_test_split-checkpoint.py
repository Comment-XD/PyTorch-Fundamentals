import os
import numpy as np
import shutil
import random

# Define the directories
data_dir = r"C:\Users\Brand\Downloads"
image_dir = data_dir + r"\Apini_2"
text_dir = data_dir + r"\labels"
img_val_dir = data_dir + r"\images\val"
txt_val_dir = data_dir + r"\labels\val"

print(image_dir)

# Create validation directories if it doesn't exist
# if not os.path.exists(img_val_dir):
#     os.makedirs(img_val_dir)
# if not os.path.exists(txt_val_dir):
#     os.makedirs(txt_val_dir)

# # Get a list of all image files
image_files = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
# print(np.array_split(image_files, 3))

def train_test_split(image_files, k):
    split_image_files = np.array_split(image_files, k)
    
    for i, image_file in enumerate(split_image_files):
        image_save_path = data_dir + r"\Apini_" + str(i + 4)
        
        for img in image_file:
            image_path = os.path.join(image_dir, img)
            shutil.move(image_path, image_save_path)
        
train_test_split(image_files, 2)
# # Calculate the number of files to move
# num_files_to_move = int(len(image_files) * 0.2)

# # Select random files to move
# files_to_move = random.sample(image_files, 2000)

# # Move selected files and their associated text files to validation directory
# for file_to_move in files_to_move:
#     image_path = os.path.join(image_dir, file_to_move)
#     text_path = os.path.join(text_dir, file_to_move.replace(".jpg", ".txt"))
    
#     validation_image_path = os.path.join(img_val_dir, file_to_move)
#     validation_text_path = os.path.join(txt_val_dir, file_to_move.replace(".jpg", ".txt"))
    
#     shutil.move(image_path, validation_image_path)
#     shutil.move(text_path, validation_text_path)
    
#     print(f"Moved {file_to_move} and its associated text file to validation directory.")

