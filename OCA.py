import csv
from skimage.metrics import structural_similarity as ssim
import SimpleITK as sitk
import os
import numpy as np
import random
import cProfile
import pandas as pd


# Set the directory paths for the two datasets
folder_path = "/home/workspace/dataset/FeTS/partition1"
domains = ['01', '18', '04', '21', '06', '13', '20']
tasks = ['ED', 'ET', 'NCR']

all_results = []
csv_file = "ssim_score.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "ssim"])
    
    writer.writeheader()

def compute_ssim(image1, image2, K1=0.3, K2=0.3, L=1):
    """
    Computes the Structural Similarity Index (SSIM) score of two input NumPy arrays.
    Args:
        arr1: NumPy array of shape (H, W) or (H, W, C) representing the first image.
        arr2: NumPy array of shape (H, W) or (H, W, C) representing the second image.
        K1: float, a small constant used to stabilize the division with weak denominator.
        K2: float, a small constant used to stabilize the division with weak denominator.
        L: float, the dynamic range of the pixel values (typically 255 for uint8 images).
    Returns:
        ssim_score: float, the computed SSIM score between arr1 and arr2.
    """
    arr1 = sitk.GetArrayFromImage(image1)
    arr2 = sitk.GetArrayFromImage(image2)
    
    # Constants used in the SSIM formula
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Mean and variance of the two input arrays
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1 = np.var(arr1)
    sigma2 = np.var(arr2)
    sigma12 = np.cov(arr1.flatten(), arr2.flatten())[0][1]
    
    # Compute SSIM score
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    ssim_score = numerator / denominator
    
    return ssim_score

def compute_distance(domain1, domain2):
    folder1_path = os.path.join(folder_path, domain1, "label")
    folder2_path = os.path.join(folder_path, domain2, "label")
    
    if domain1 == domain2:
        for i in range(3):
            for j in range(3):
                subfolder1_name = tasks[i]
                subfolder2_name = tasks[j]
                subfolder1_path = os.path.join(folder1_path, tasks[i])
                subfolder2_path = os.path.join(folder2_path, tasks[j])
                ssim_scores = []
                for file_name in random.sample(os.listdir(subfolder1_path), 10):
                    file1 = os.path.join(subfolder1_path, file_name)
                    file2 = os.path.join(subfolder2_path, file_name)
                    image1 = sitk.ReadImage(file1)
                    image2 = sitk.ReadImage(file2)
                    ssim_score = compute_ssim(image1, image2)
                    ssim_scores.append(ssim_score)
                avg_ssim = np.mean(ssim_scores) 
                
                result_entry = {
                    "node_a": f"{domain1}-{subfolder1_name}",
                    "node_b": f"{domain2}-{subfolder2_name}",
                    "ssim_score": avg_ssim
                }
                
                
                
                with open(csv_file, mode='a', newline='') as file:
                    
                    writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "ssim", "Normalized ssim"])
                    writer.writerow({
                        "Node A": result_entry["node_a"],
                        "Node B": result_entry["node_b"],
                        "ssim": f"{result_entry['ssim_score']:.4f}", 
                    })
                all_results.append(result_entry)   
                print(f"({domain1}-{subfolder1_name} , {domain2}-{subfolder2_name}) ssim score is {avg_ssim:.5f}.")
                
    else:
        for i in range(3):
            for j in range(3):
                subfolder1_name = tasks[i]
                subfolder2_name = tasks[j]
                subfolder1_path = os.path.join(folder1_path, tasks[i])
                subfolder2_path = os.path.join(folder2_path, tasks[j])
                ssim_scores = []
                # loop through the files in the sub-folders
                for file1_name in random.sample(os.listdir(subfolder1_path), 10):
                    # make sure the file exists in the other sub-folder
                    for file2_name in random.sample(os.listdir(subfolder2_path), 10):
                        file1 = os.path.join(subfolder1_path, file1_name)
                        file2 = os.path.join(subfolder2_path, file2_name)
                        image1 = sitk.ReadImage(file1)
                        image2 = sitk.ReadImage(file2)
                        ssim_score = compute_ssim(image1, image2)
                        #print(f'Compute the {i}th pair done!')
                        ssim_scores.append(ssim_score)
                avg_ssim = np.mean(ssim_scores) 
                
                result_entry = {
                    "node_a": f"{domain1}-{subfolder1_name}",
                    "node_b": f"{domain2}-{subfolder2_name}",
                    "ssim_score": avg_ssim
                }
                
                all_results.append(result_entry)
                
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "ssim", "Normalized ssim"])
                    writer.writerow({
                        "Node A": result_entry["node_a"],
                        "Node B": result_entry["node_b"],
                        "ssim": f"{result_entry['ssim_score']:.4f}", 
                    })
                
                print(f"({domain1}-{subfolder1_name} , {domain2}-{subfolder2_name}) ssim score is {avg_ssim:.5f}.")
                    
for i in range(7):
  for j in range(i,7):
      compute_distance(domains[i], domains[j])



print('Compute all pairs done, results saved in ssim_score.csv!')