import numpy as np
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

# Set the directory paths for the two datasets
folder_path = "/home/workspace/dataset/FeTS/partition1"
domains = ['01', '18', '04', '21', '06', '13', '20']
modality = ['flair', 't1', 't1ce', 't2']

all_results = []
csv_file = "wasserstein_distance.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "wasserstein_distance"])
    
    writer.writeheader()

def pca_transform(image, n_components):
    
    output_shape=(128, 128, 128)
    
    image_resized = resize(image, output_shape, anti_aliasing=True)

    height, width, depth = image_resized.shape
    image_2d = image_resized.reshape(height * width, depth)


    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(image_2d)

    return principal_components

def compute_distance(domain1, domain2,n_pca_components=0.75):
    folder1_path = os.path.join(folder_path, domain1, "image")
    folder2_path = os.path.join(folder_path, domain2, "image")
    
    if domain1 == domain2:
        for i in range(4):
            for j in range(4):
                subfolder1_name = modality[i]
                subfolder2_name = modality[j]
                subfolder1_path = os.path.join(folder1_path, modality[i])
                subfolder2_path = os.path.join(folder2_path, modality[j])
                distances = []
                for file_name in random.sample(os.listdir(subfolder1_path), 10):
                    basename = os.path.splitext(file_name)[0]
                    patient = "_".join(basename.split("_")[:2])
                    file1_name = file_name
                    file2_name = f"{patient}_{subfolder2_name}.nii.gz"
                    file1 = os.path.join(subfolder1_path, file1_name)
                    file2 = os.path.join(subfolder2_path, file2_name)
                    # image1 = sitk.ReadImage(file1)
                    # image2 = sitk.ReadImage(file2)
                    image1 = nib.load(file1).get_fdata()
                    image2 = nib.load(file2).get_fdata()
                    pca_arr1 = pca_transform(image1, n_pca_components)
                    pca_arr2 = pca_transform(image2, n_pca_components)
                    distance = wasserstein_distance(pca_arr1.flatten(), pca_arr2.flatten())
                    distances.append(distance)
                avg_wassdis = np.mean(distances) 
                print(f"({domain1}{subfolder1_name} , {domain2}{subfolder2_name}) wasserstein distance is {avg_wassdis:.5f}.")
                result_entry = {
                    "node_a": f"{domain1}-{subfolder1_name}",
                    "node_b": f"{domain2}-{subfolder2_name}",
                    "wasserstein_distance": avg_wassdis
                }
                all_results.append(result_entry)
    else:
    # loop through the sub-folders in folder1 and folder2
        for i in range(4):
            for j in range(4):
                subfolder1_name = modality[i]
                subfolder2_name = modality[j]
                subfolder1_path = os.path.join(folder1_path, modality[i])
                subfolder2_path = os.path.join(folder2_path, modality[j])
                distances = []
                #loop through the files in the sub-folders
                #for file1_name in os.listdir(subfolder1_path):
                for file1_name in random.sample(os.listdir(subfolder1_path), 10):
                    # make sure the file exists in the other sub-folder
                    for file2_name in random.sample(os.listdir(subfolder2_path), 10):
                        file1 = os.path.join(subfolder1_path, file1_name)
                        file2 = os.path.join(subfolder2_path, file2_name)
                        # image1 = sitk.ReadImage(file1)
                        # image2 = sitk.ReadImage(file2)
                        image1 = nib.load(file1).get_fdata()
                        image2 = nib.load(file2).get_fdata()
                        pca_arr1 = pca_transform(image1, n_pca_components)
                        pca_arr2 = pca_transform(image2, n_pca_components)
                        distance = wasserstein_distance(pca_arr1.flatten(), pca_arr2.flatten())
                        distances.append(distance)
                avg_wassdis = np.mean(distances) 
                
                print(f"({domain1}-{subfolder1_name} , {domain2}-{subfolder2_name}) wasserstein distance is {avg_wassdis:.5f}.")
                
                result_entry = {
                    "node_a": f"{domain1}-{subfolder1_name}",
                    "node_b": f"{domain2}-{subfolder2_name}",
                    "wasserstein_distance": avg_wassdis
                }
                all_results.append(result_entry)
                
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "Wasserstein Distance"])
                    writer.writerow({
                        "Node A": result_entry["node_a"],
                        "Node B": result_entry["node_b"],
                        "Wasserstein distance": f"{result_entry['wasserstein_distance']:.4f}"
                    })
                

for i in range(7):
  for j in range(i,7):
      compute_distance(domains[i], domains[j])


print('Compute all pairs done, results saved in wasserstein_distance.csv!')
   