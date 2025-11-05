import pandas as pd
import os
import numpy as np

# 🔧 Folder path
input_folder = r"Path"
output_folder = r"Path"

# ---------------------------------------------------D. salina---------------------------------------
os.makedirs(output_folder, exist_ok=True)

a, b = -0.4901, 7.1982

for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_excel(file_path)
        
        # β-carotene calculation
        y = df.iloc[:, 7].astype(float)  
        beta_values = (y - b) / a        # x = (y - b) / a                
        beta_values = np.where(beta_values < 0, 0, beta_values)     
        df["Betacarotene"] = beta_values        
        df.columns.values[8] = "Betacarotene"        
        save_path = os.path.join(output_folder, filename)
        df.to_excel(save_path, index=False)
        print(f"Saved: {save_path}")
        
# ---------------------------------------------------H. pluvialis---------------------------------------
os.makedirs(output_folder, exist_ok=True)

a, b = -2.73625, 45.60250

for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(input_folder, filename)        
        df = pd.read_excel(file_path)
        
        #Astaxanthin calculation
        chl_values = df.iloc[:, 7].astype(float)
        asta_values = a * chl_values + b          # y = a*x + b        
        asta_values = np.where(asta_values < 0, 0, asta_values)        
        df.insert(8, "Astaxanthin", asta_values)        
        save_path = os.path.join(output_folder, filename)
        df.to_excel(save_path, index=False)

        print(f"Saved: {save_path}")