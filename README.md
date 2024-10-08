# SeqTransfer
A novel sequential transfer scheme with a path selection strategy for medical image analysis.

1. we propose an analytical task affinity metric for medical image segmentation tasks to quantitatively predict the transfer performance of source tasks. 
Considering the characteristics of medical image segmentation tasks, we analyze the image and label similarity between tasks and compute the task affinity scores.

  <img width="679" alt="截屏2024-10-08 19 04 01" src="https://github.com/user-attachments/assets/1c7e1ee7-b895-43a8-980d-3813ce4af90b">
  
2. we calculate the distance between source tasks and then connect each source node and set the edges accordingly.

3. we construct the source graph.

4. given the target task we estimate the transfer cost of paths and, consequently, select the best sequential transfer path toward the target task.

<img width="963" alt="截屏2024-10-08 19 25 16" src="https://github.com/user-attachments/assets/1a69c7cb-2c06-4030-8aba-0c4815a935c5">

  Run task_affinity_estimation.py to calculate the task affinity between task i and task j.

  ICA.py is for the image characteristics similarity estimation.

  OCA.py is for the object characteristics similarity estimation.



