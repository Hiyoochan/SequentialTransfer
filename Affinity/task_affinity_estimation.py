import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import nibabel as nib
import os
import random
import SimpleITK as sitk
from itertools import combinations_with_replacement
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
import csv

folder_path = "/home/workspace/dataset"
domain1 = "16-T2-NCR"
domain2 = "01-T1-ED"

ica_csv_file = "ICA.csv"
oca_csv_file = "OCA.csv"

all_ICA_results = []
all_OCA_results = []
all_results = []


def pca_transform(image, n_components):
    
    output_shape=(128, 128, 128)

    # Resize the image
    image_resized = resize(image, output_shape, anti_aliasing=True)

    height, width, depth = image_resized.shape

    image_2d = image_resized.reshape(height * width, depth)

    # Standardizing the features before applying PCA
    # scaler = StandardScaler()
    # image_array = scaler.fit_transform(image_2d)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(image_2d)

    return principal_components

def image_char_analysis(domain1, domain2, n_pca_components=0.75):
    dataset1_path = os.path.join(folder_path, domain1, "image")
    dataset2_path = os.path.join(folder_path, domain2, "image")
    distances = []
    # if the two datasets are from same patients with different modalities, you can compute the pair-wise distance, e.g. (patient01.t1.gii.gz,patient01.t1.nii.gz)
    
    for file1_name in random.sample(os.listdir(dataset1_path), 10):
        for file2_name in random.sample(os.listdir(dataset2_path), 10):
            file1 = os.path.join(dataset1_path, file1_name)
            file2 = os.path.join(dataset2_path, file2_name)
            image1 = nib.load(file1).get_fdata()
            image2 = nib.load(file2).get_fdata()
            pca_arr1 = pca_transform(image1, n_pca_components)
            pca_arr2 = pca_transform(image2, n_pca_components)
            distance = wasserstein_distance(pca_arr1.flatten(), pca_arr2.flatten())
            distances.append(distance)
    avg_wassdis = np.mean(distances) 
    
    print(f"({domain1} , {domain2}) wasserstein distance is {avg_wassdis:.5f}.")
    
    result_entry = {
        "node_a": f"{domain1}-{subfolder1_name}",
        "node_b": f"{domain2}-{subfolder2_name}",
        "wasserstein_distance": avg_wassdis
    }
    all_ICA_results.append(result_entry)

    with open(ica_csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "Wasserstein Distance"])
        writer.writerow({
            "Node A": result_entry["node_a"],
            "Node B": result_entry["node_b"],
            "ssim": f"{result_entry['wasserstein_distance']:.4f}",  
        })
        


def compute_ssim(image1, image2, K1=0.3, K2=0.3, L=1):
    """
    Computes the Structural Similarity Index (SSIM) score of two input NumPy arrays.
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

def object_char_analysis(domain1, domain2):
    dataset1_path = os.path.join(folder_path, domain1, "label")
    dataset2_path = os.path.join(folder_path, domain2, "label")

    ssim_scores = []
    for file1_name in random.sample(os.listdir(dataset1_path), 10):
        # make sure the file exists in the other sub-folder
        for file2_name in random.sample(os.listdir(dataset2_path), 10):
            file1 = os.path.join(dataset1_path, file1_name)
            file2 = os.path.join(dataset2_path, file2_name)
            image1 = sitk.ReadImage(file1)
            image2 = sitk.ReadImage(file2)
            ssim_score = compute_ssim(image1, image2)
            #print(f'Compute the {i}th pair done!')
            ssim_scores.append(ssim_score)
    avg_ssim = np.mean(ssim_scores) 
    
    result_entry = {
        "node_a": f"{domain1}",
        "node_b": f"{domain2}",
        "ssim_score": avg_ssim
    }
    
    all_OCA_results.append(result_entry)
    
    with open(oca_csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "ssim"])
        writer.writerow({
            "Node A": result_entry["node_a"],
            "Node B": result_entry["node_b"],
            "ssim": f"{result_entry['ssim_score']:.4f}",  
        })
                
    print(f"({domain1}, {domain2}) ssim score is {avg_ssim:.5f}.")
        
image_char_analysis(domain1, domain2)
object_char_analysis(domain1, domain2)

# Use the determinated alpha & beta, you can modify the weight
alpha = 0.0044
beta = -2.4756
C = 2.4756

# Read the data from the CSV file using pandas
df_wass = pd.read_csv(ica_csv_file)
df_ssim = pd.read_csv(oca_csv_file)

wass_row = df_wass[(df_wass['Node A'] == f"{domain1}") & (df_wass['Node B'] == f"{domain2}")]
ssim_row = df_ssim[(df_ssim['Node A'] == f"{domain1}") & (df_ssim['Node B'] == f"{domain2}")]
                    
if not wass_row.empty:
    wassdistance = wass_row.iloc[0]['Wasserstein Distance']  
if not ssim_row.empty:
    ssim = ssim_row.iloc[0]['ssim']   
        
distance = f"{(alpha*wassdistance + beta*ssim + C) :.4f}"
state = " "
all_results.append((domain1, domain2, state, distance))

# Create a DataFrame from the result
df_result = pd.DataFrame(all_results, columns=['Node A', 'Node B', 'State', 'Distance'])

# Save the result to a new CSV file
df_result.to_csv('task_affinity.csv', index=False)

print('Compute the task affinity done!') 


