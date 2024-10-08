import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Load the edges data
edges_df = pd.read_csv('cost.csv')  # Replace with your file path
# Ensure your CSV has 'node_a', 'node_b', 'node_t', 'ssim_ab', 'wass_ab', 'ssim_bt', 'wass_bt', 'ssim_at', 'wass_bt',and 'performance'

# Define the cost function
def calculate_cost(alpha, beta, gamma, ssim_ab, wass_ab, ssim_bt, wass_bt,ssim_at, wass_at):
    return alpha * ssim_at + beta * wass_at + alpha * (ssim_ab + ssim_bt) + beta (wass_ab + wass_bt)


@use_named_args(dimensions=[Real(0.001, 10.000, name='alpha'), Real(0.001, 10.000, name='beta')])
def objective(alpha, beta):
    costs = edges_df.apply(lambda row: calculate_cost(alpha, beta, row['ssim_at'], row['wass_at'], row['ssim_ab'], row['wass_ab'], row['ssim_bt'], row['wass_bt']), axis=1)
    
    # Calculate the correlation between these costs and the real transfer performance
    correlation, _ = pearsonr(costs, edges_df['performance'])
    
    # Since we need to maximize correlation, minimize the negative correlation
    return -correlation

# Define the space of possible values for α and β
space = [Real(0.001, 10.0, name='alpha'), Real(0.001, 10.0, name='beta')]

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=50, random_state=0, n_points=10)

# Output the results
print(f"Optimal α: {result.x[0]}, Optimal β: {result.x[1]}")
print(f"Maximum correlation: {-result.fun}")

alpha = result.x[0]
beta = result.x[1]

domains = ['01', '18', '04', '21', '06', '13', '20']
modalities = ['flair', 't1', 't1ce', 't2']
tasks = ['ED', 'ET', 'NCR']

# Read the CSV files into DataFrames
df_ssim = pd.read_csv("/home/workspace/sequential/ssim/ssim_score.csv")  
df_wass = pd.read_csv("/home/workspace/sequential/wasserstein/wasserstein.csv")


result = []

for domaini in range(7):
  for domainj in range(domaini,7):
    for taski in range(3):
        for taskj in range(3):
            for modalityi in range(4):
                for modalityj in range(4):
                    domain1 = domains[domaini]
                    domain2 = domains[domainj]
                    task1 = tasks[taski]
                    task2 = tasks[taskj]
                    
                    modality1 = modalities[modalityi]
                    modality2 = modalities[modalityj]
                    name1 = f"{domain1}-{modality1}-{task1}"
                    name2 = f"{domain2}-{modality2}-{task2}"
                    distance = 0.0
                    state = ""
                    
                    if name1 == name2:
                        continue
                    
                    wassdistance = 0.0
                    ssim = 0.0
                    wass_row = df_wass[(df_wass['Node A'] == f"{domain1}-{modality1}") & (df_wass['Node B'] == f"{domain2}-{modality2}")]
                    ssim_row = df_ssim[(df_ssim['Node A'] == f"{domain1}-{task1}") & (df_ssim['Node B'] == f"{domain2}-{task2}")]
                    
                    if not wass_row.empty:
                        wassdistance = wass_row.iloc[0]['Sqrt Normalized WD']  
                    if not ssim_row.empty:
                        ssim = ssim_row.iloc[0]['Normalized ssim']   
                            
                    distance = f"{(alpha*wassdistance + beta*ssim) :.4f}"
                    
                    if distance == 0.0000 :
                        continue
                    
                    if task1 != task2 and modality1 != modality2:
                        state = "Unconnected"
                        result.append((name1, name2, state, distance))
                    else :
                        state = " "
                        result.append((name1, name2, state, distance))

# Create a DataFrame from the result
df_result = pd.DataFrame(result, columns=['Node A', 'Node B', 'State', 'Distance'])

# Save the result to a new CSV file
df_result.to_csv('graph.csv', index=False)

print('Construct the graph done!') 