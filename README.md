# Graph-guided Sequential Transfer for Medical Image Segmentation
## Overview
A novel sequential transfer scheme with a path selection strategy for medical image analysis.

### Task Affinity Estimation
We propose an analytical task affinity metric for medical image segmentation tasks to quantitatively predict the transfer performance of source tasks. 
Considering the characteristics of medical image segmentation tasks, we analyze the image and label similarity between tasks and compute the task affinity scores.

  <img width="679" alt="截屏2024-10-08 19 04 01" src="https://github.com/user-attachments/assets/1c7e1ee7-b895-43a8-980d-3813ce4af90b">
  
### Graph Construction
We calculate the distance between source tasks and then connect each source node, and set the edges accordingly then we construct the source graph.

### Optimal Sequential Transfer
Given the target task we estimate the transfer cost of paths and, consequently, select the best sequential transfer path toward the target task.

<img width="963" alt="截屏2024-10-08 19 25 16" src="https://github.com/user-attachments/assets/1a69c7cb-2c06-4030-8aba-0c4815a935c5">


## Try with Transferability Estimation
Run task_affinity_estimation.py to calculate the task affinity between task i and task j.
  ```
  python task_affinity_estimation.py
  ```

  ICA is for the image characteristics similarity estimation.
  ```
  python ICA.py
  ```

  OCA is for the object characteristics similarity estimation.
  ```
  python OCA.py
  ```



