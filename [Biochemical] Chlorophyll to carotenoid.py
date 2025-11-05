import pandas as pd
import os
import numpy as np

# 🔧 폴더 경로 설정 (원하는 경로로 바꿔줘)
input_folder = r"C:\Python\workspace\Microalgae\final2\parameter excel\H_pluvialis_chl_revised"
output_folder = r"C:\Python\workspace\Microalgae\final2\parameter excel\H_pluvialis_chl_revised\Final_H_pluvialis"

# ---------------------------------------------------D. salina---------------------------------------
# # 🔧 출력 폴더가 없으면 생성 (D.salina)
# os.makedirs(output_folder, exist_ok=True)

# # 선형식 계수 (y = a*x + b)
# a, b = -0.4901, 7.1982

# # 엑셀 파일 반복 처리
# for filename in os.listdir(input_folder):
#     if filename.endswith(".xlsx"):
#         file_path = os.path.join(input_folder, filename)

#         # 엑셀 읽기
#         df = pd.read_excel(file_path)
        
#         # ✅ 2. 수식 적용: 8번째 열(y값)으로부터 x값(β-carotene) 계산
#         y = df.iloc[:, 7].astype(float)  # 8번째 열
#         beta_values = (y - b) / a        # x = (y - b) / a
        
#         # ✅ 3. 음수 값은 0으로 치환
#         beta_values = np.where(beta_values < 0, 0, beta_values)

#         # ✅ 3. 계산된 값들을 새 열로 추가
#         df["Betacarotene"] = beta_values

#         # ✅ 4. 9번째 열의 이름을 "Betacarotene"로 설정
#         df.columns.values[8] = "Betacarotene"

#         # 저장 경로 지정
#         save_path = os.path.join(output_folder, filename)
#         df.to_excel(save_path, index=False)

#         print(f"저장 완료: {save_path}")
        
# ---------------------------------------------------H. pluvialis---------------------------------------
        
# 🔧 출력 폴더가 없으면 생성 (D.salina)
os.makedirs(output_folder, exist_ok=True)

# 선형식 계수 (y = a*x + b)
a, b = -2.73625, 45.60250

# 엑셀 파일 반복 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(input_folder, filename)

        # 엑셀 읽기
        df = pd.read_excel(file_path)
        
        # ✅ 2. x값(Chl, 8번째 열) → y값(Astaxanthin) 계산
        chl_values = df.iloc[:, 7].astype(float)   # 8번째 열 (Chl)
        asta_values = a * chl_values + b          # y = a*x + b

        # ✅ 3. 음수 값은 0으로 치환
        asta_values = np.where(asta_values < 0, 0, asta_values)

        # ✅ 4. 새 열 추가 (9번째 열로 삽입)
        df.insert(8, "Astaxanthin", asta_values)

        # 저장 경로 지정
        save_path = os.path.join(output_folder, filename)
        df.to_excel(save_path, index=False)

        print(f"저장 완료: {save_path}")

        
        
# # 🔧 출력 폴더가 없으면 생성 (H_plu)
# os.makedirs(output_folder, exist_ok=True)

# # 로지스틱 모델 파라미터 (예시)
# L = 25
# k = 1.4963
# x0 = 12.7794

# # 엑셀 파일 반복 처리
# for filename in os.listdir(input_folder):
#     if filename.endswith(".xlsx"):
#         file_path = os.path.join(input_folder, filename)

#         # 엑셀 읽기
#         df = pd.read_excel(file_path)

#         # ✅ chl 값을 기반으로 astaxanthin 계산 (8번째 열이 chl임)
#         chl_values = df.iloc[:, 7]  # chl 열
#         astaxanthin_values = L / (1 + np.exp(k * (chl_values - x0)))

#          # ✅ 열 개수에 따라 대체 or 추가
#         if df.shape[1] >= 9:
#             df.iloc[:, 8] = astaxanthin_values
#             df.columns.values[8] = "Astaxanthin"
#         else:
#             df["Astaxanthin"] = astaxanthin_values

#         # 결과 저장
#         save_path = os.path.join(output_folder, filename)
#         df.to_excel(save_path, index=False)

#         print(f"저장 완료: {save_path}")