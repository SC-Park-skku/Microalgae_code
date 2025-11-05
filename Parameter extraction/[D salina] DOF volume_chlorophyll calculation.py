import pandas as pd
import numpy as np
import math

excel_path = r"Path"
df = pd.read_excel(excel_path)

chl_a_dof_list = []
chl_a_total_list = []

DOF = 4.45  # μm
intensity_per_pg = 22.04

for idx, row in df.iterrows():
    try:
        a = row["Major Axis"] / 2  
        b = row["Minor Axis"] / 2  
        mean_intensity = row["Ellipse Mean Gray Scale"]
        
        V_total = (4/3) * math.pi * a * b**2
        
        h_half = DOF / 2
        if h_half > b or V_total == 0:
            chl_dof = np.nan
            chl_total = np.nan
        else:
            v_ratio = (3/4) * (h_half / b) - (1/4) * (h_half / b)**3
            V_dof = 2 * v_ratio * V_total
            
            chl_dof = mean_intensity / intensity_per_pg
             
            chl_total = chl_dof * (V_total / V_dof)

        chl_a_dof_list.append(chl_dof)
        chl_a_total_list.append(chl_total)
    except:
        chl_a_dof_list.append(np.nan)
        chl_a_total_list.append(np.nan)

df["chl_a_pg_DOF"] = chl_a_dof_list
df["chl_a_pg_total"] = chl_a_total_list

output_path = r"Save.xlsx"
df.to_excel(output_path, index=False)

print("✅ Saved:", output_path)