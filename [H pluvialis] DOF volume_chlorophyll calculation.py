import pandas as pd
import numpy as np
import math

# 엑셀 파일 불러오기
excel_path = r"C:\Python\workspace\Microalgae\final2\parameter excel\astaxanthin_cal_backup\H_pluvialis_25day_axes_overwritten.xlsx"
df = pd.read_excel(excel_path)

# 결과 저장 리스트
chl_a_dof_list = []
chl_a_total_list = []

# 상수
DOF = 4.45  # μm
intensity_per_pg = 16.78

# 계산 루프
for idx, row in df.iterrows():
    try:
        a = row["Major Axis"] / 2  # 장축 반지름
        b = row["Minor Axis"] / 2  # 단축 반지름 (z축)
        mean_intensity = row["Ellipse Mean Gray Scale"]

        # 전체 부피
        V_total = (4/3) * math.pi * a * b**2

        # DOF 기준 부피 비율 계산
        h_half = DOF / 2
        if h_half > b or V_total == 0:
            chl_dof = np.nan
            chl_total = np.nan
        else:
            v_ratio = (3/4) * (h_half / b) - (1/4) * (h_half / b)**3
            V_dof = 2 * v_ratio * V_total

            # 관측된 DOF 범위 내 질량
            chl_dof = mean_intensity / intensity_per_pg

            # 전체 chl a 질량 추정
            chl_total = chl_dof * (V_total / V_dof)

        chl_a_dof_list.append(chl_dof)
        chl_a_total_list.append(chl_total)
    except:
        chl_a_dof_list.append(np.nan)
        chl_a_total_list.append(np.nan)

# 결과 추가
df["chl_a_pg_DOF"] = chl_a_dof_list
df["chl_a_pg_total"] = chl_a_total_list

# 결과 저장
output_path = r"C:\Python\workspace\Microalgae\final2\parameter excel\H_pluvialis_chl_revised\H_pluvialis_25day.xlsx"
df.to_excel(output_path, index=False)

print("✅ 저장 완료:", output_path)