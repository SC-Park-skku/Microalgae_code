import cv2
import numpy as np
import math
import os
import pandas as pd
from scipy.interpolate import interp1d

# 폴더 경로 설정
folder_path = r"C:\Python\workspace\Microalgae\final2\parameter excel"

# 경로 설정
src_dirs = [r"C:\Python\workspace\Microalgae\final2\H_pluvialis_25day\crops\plu"]

output_filenames = ["H_pluvialis_25day.xlsx"]

# DataFrame을 결과에 저장하기 위해 생성
for src_dir, output_filename in zip(src_dirs, output_filenames):
    data = pd.DataFrame(columns=["Filename", "Major Axis", "Minor Axis", "Size(μm²)", "Eccentricity", "FWHM of scattered light", "Ellipse Mean Gray Scale"])
    
    # 디렉터리 내의 모든 파일 처리
    for i, src_filename in enumerate(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, src_filename)
    
        if not os.path.isfile(src_path) or not src_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 이미지 파일만 처리
    
        # 이미지 로드
        src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        
        
        # ✅ 이미지 대비 향상 (CLAHE 적용)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        src = clahe.apply(src)
        
        # ✅ Adaptive Threshold 적용 (지역적 밝기 차이 해결)
        dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 2)
        
        # # ✅ Histogram Equalization 적용 (전체 밝기 개선)
        # src = cv2.equalizeHist(src)
        
        # # ✅ OpenCV Otsu's 방법을 사용하여 최적 threshold 자동 선택
        # _, dst2 = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate the current average pixel intensity
        current_average = np.mean(src)
    
        # Adjust the image intensity so that the average becomes 128
        src = src + (126 - current_average)
    
        # Ensure the pixel values are within [0, 255]
        src = np.clip(src, 0, 255).astype(np.uint8)
    
        # Initialize best values
        best_alpha2 = None
        best_threshold = None
        best_eccentricity = float('inf')
        best_ellipse = None
        best_area = float('inf')
    
        # Try alpha2 values from 1.0 to 5.0 and threshold values from 130 to 250
        for alpha2 in np.arange(1.0, 5.0, 0.1):
            for threshold in range(150, 155, 1):
                # Adjust contrast
                dst2 = np.clip((1 + alpha2) * src - 128 * alpha2, 0, 255).astype(np.uint8)
    
                # Threshold
                _, dst2 = cv2.threshold(dst2, threshold, 255, cv2.THRESH_BINARY)
    
                # Find contours in the image
                contours, _ = cv2.findContours(dst2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
                for contour in contours:
                    if contour.shape[0] >= 5:  # fitEllipse requires at least 5 points
                        ellipse = cv2.fitEllipse(contour)
                        (center, axes, orientation) = ellipse
                        original_major_axis = max(axes)
                        original_minor_axis = min(axes)
                    
                        # 타원의 두께 보정
                        thickness_values = []
                        for point in contour:
                            distance = math.sqrt((point[0][0] - center[0])**2 + (point[0][1] - center[1])**2)
                            if distance < original_major_axis / 2:
                                thickness_values.append(abs(original_major_axis / 2 - distance))

                        average_thickness = np.mean(thickness_values) if thickness_values else 0
                        major_axis = original_major_axis + (2 * average_thickness)
                        minor_axis = original_minor_axis + (2 * average_thickness)

                        # 면적 계산
                        ellipse = (center, (major_axis, minor_axis), orientation)
                        area = np.pi * (major_axis / 2) * (minor_axis / 2)
                        contour_area = cv2.contourArea(contour)
                        contour_perimeter = cv2.arcLength(contour, True)

                        # Shape factor (원형도) 계산
                        shape_factor = (4 * np.pi * contour_area) / (contour_perimeter ** 2) if contour_perimeter > 0 else 0
                        
                    
                        
                        # 표준 이심률 공식 적용
                        if major_axis > 0 and minor_axis > 0:
                            eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
                        else:
                            continue  # 잘못된 타원은 스킵

                        # ✅ 최적화된 필터링 적용
                        if 15 <= minor_axis <= 80 and (major_axis / minor_axis) < 1.5 and shape_factor > 0.7:
                            if eccentricity < best_eccentricity:
                                best_eccentricity = eccentricity
                                best_ellipse = ellipse
                                best_alpha2 = alpha2
                                best_threshold = threshold
                                best_area = area

        if best_alpha2 is None:
            print(f"⚠️ 경고: {src_filename}에서 적절한 타원이 검출되지 않았습니다!")
            continue

                        # Adjust contrast using best alpha2
        dst3 = np.clip((1 + best_alpha2) * src - 128 * best_alpha2, 0, 255).astype(np.uint8)
        
        # 타원을 원본 밝기의 이미지에 그리기 위해 원본 이미지를 다시 불러오기
        original_img = cv2.imread(src_path, cv2.IMREAD_COLOR)  # 원본 이미지 유지 (컬러)
        
        # best_ellipse가 존재하면 타원을 그린 후 저장
        if best_ellipse is not None:
            cv2.ellipse(original_img, best_ellipse, (0, 255, 0), 1)  # 초록색 타원 그리기
            
            # ✅ best_ellipse에서 다시 정확한 축 길이와 이심률 추출
            (center, (fitted_major_axis, fitted_minor_axis), orientation) = best_ellipse
            
            # fitted_major/minor_axis의 순서 정리 (항상 major > minor 보장)
            if fitted_major_axis < fitted_minor_axis:
                fitted_major_axis, fitted_minor_axis = fitted_minor_axis, fitted_major_axis
            
            # ✅ 이심률 재계산
            if fitted_major_axis > 0 and fitted_minor_axis > 0:
                fitted_eccentricity = np.sqrt(1 - (fitted_minor_axis ** 2) / (fitted_major_axis ** 2))
            else:
                fitted_eccentricity = 0  # 또는 np.nan


            # 타원 그린 이미지 저장 폴더 설정
            output_image_folder = r"C:\Python\workspace\Microalgae\final2\H_pluvialis_25day\crops\draw2"

            # 폴더가 존재하지 않으면 생성
            os.makedirs(output_image_folder, exist_ok=True)

            # 저장할 경로 설정
            output_image_path = os.path.join(output_image_folder, f"{src_filename}")
            
            # 이미지 저장
            success = cv2.imwrite(output_image_path, original_img)
            if success:
                print(f"✅ 저장 성공: {output_image_path}")
            else:
                print(f"❌ 저장 실패: {output_image_path}")



                        # 🔥 타원 내부 grayscale 평균값 계산 추가
            src_original = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                        
            ellipse_mask = np.zeros_like(src_original, dtype=np.uint8)  # src와 동일한 크기
            cv2.ellipse(ellipse_mask, best_ellipse, 255, -1)  # 타원 마스크 생성
            
            # ✅ 예외 처리: ellipse_mask가 비어있으면 스킵
            if np.sum(ellipse_mask) == 0:
                print(f"⚠️ {src_filename}: ellipse_mask가 비어 있습니다! 스킵합니다.")
                continue  # 다음 이미지로 넘어감
            
            ellipse_pixels = src_original[ellipse_mask == 255]
            ellipse_gray_mean = np.mean(ellipse_pixels) if len(ellipse_pixels) > 0 else 0
            
            # ✅ 내부 타원 픽셀 수 체크
            internal_pixel_count = np.sum(ellipse_mask == 255)
            
            # ✅ 두께 포함한 전체 타원 영역 마스크 생성
            kernel_size = 1  # 두께를 얼마나 확장할지 결정
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            thick_ellipse_mask = cv2.dilate(ellipse_mask, kernel, iterations=1)
            
            # ✅ 예외 처리: thick_ellipse_mask가 비어있으면 스킵
            if np.sum(thick_ellipse_mask) == 0:
                print(f"⚠️ {src_filename}: thick_ellipse_mask가 비어 있습니다! 스킵합니다.")
                continue
            
            # ✅ 전체 타원 픽셀 수 계산
            total_pixel_count = np.sum(thick_ellipse_mask == 255)
            
            # ✅ 픽셀 수를 면적으로 변환 (1 픽셀 = 0.75μm x 0.75μm)
            size_um2 = total_pixel_count * (0.75 ** 2)

            
            data = pd.concat([data, pd.DataFrame({"Filename": [src_filename],
                                                  "Major Axis": [fitted_major_axis * 0.75],
                                                  "Minor Axis": [fitted_minor_axis * 0.75],
                                                  "Size(μm²)": [size_um2],
                                                  "Eccentricity": [fitted_eccentricity],
                                                  "Ellipse Mean Gray Scale": [ellipse_gray_mean]})],
                             ignore_index=True)

    with pd.ExcelWriter(os.path.join(folder_path, output_filename), engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False)
