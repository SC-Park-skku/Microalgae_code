import cv2
import numpy as np
import math
import os
import pandas as pd
from scipy.interpolate import interp1d

folder_path = r"Your path"

# 경로 설정
src_dirs = [r"Directory path"]

output_filenames = ["H_pluvialis.xlsx"]

for src_dir, output_filename in zip(src_dirs, output_filenames):
    data = pd.DataFrame(columns=[
        "Filename", "Major Axis", "Minor Axis", "Size(μm²)", "Eccentricity", "Ellipse Mean Gray Scale",
        "Integrated Density",
        "Perimeter(μm)", "Equivalent Diameter(μm)", "Solidity", "Convexity", "Circularity",
        "Aspect Ratio", "Extent", "Rectangularity",
        "Deformation Index", "Intensity Std", "Intensity CoV"
    ])
        
    for i, src_filename in enumerate(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, src_filename)
    
        if not os.path.isfile(src_path) or not src_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  

        src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        
       
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        src = clahe.apply(src)
        
        dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 2)
   
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
        best_contour = None  
    
        # Try alpha2 values from 1.0 to 5.0 and threshold values from 130 to 250
        for alpha2 in np.arange(1.0, 5.0, 0.1):
            for threshold in range(150, 155, 1): #10, 25 day: 150, 155 / 5day: 160, 165 / 2day: 180, 185 / 0day: 190, 195
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
                                            
                        thickness_values = []
                        for point in contour:
                            distance = math.sqrt((point[0][0] - center[0])**2 + (point[0][1] - center[1])**2)
                            if distance < original_major_axis / 2:
                                thickness_values.append(abs(original_major_axis / 2 - distance))

                        average_thickness = np.mean(thickness_values) if thickness_values else 0
                        major_axis = original_major_axis + (2 * average_thickness)
                        minor_axis = original_minor_axis + (2 * average_thickness)
                        
                        ellipse = (center, (major_axis, minor_axis), orientation)
                        area = np.pi * (major_axis / 2) * (minor_axis / 2)
                        contour_area = cv2.contourArea(contour)
                        contour_perimeter = cv2.arcLength(contour, True)
                       
                        shape_factor = (4 * np.pi * contour_area) / (contour_perimeter ** 2) if contour_perimeter > 0 else 0
                                                                   
                        if major_axis > 0 and minor_axis > 0:
                            eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
                        else:
                            continue  

                        if 15 <= minor_axis <= 80 and (major_axis / minor_axis) < 1.5 and shape_factor > 0.7:
                            if eccentricity < best_eccentricity:
                                best_eccentricity = eccentricity
                                best_ellipse = ellipse
                                best_alpha2 = alpha2
                                best_threshold = threshold
                                best_area = area
                                best_contour = contour            

        if best_alpha2 is None:           
            continue

        # Adjust contrast using best alpha2
        dst3 = np.clip((1 + best_alpha2) * src - 128 * best_alpha2, 0, 255).astype(np.uint8)
               
        original_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                
        if best_ellipse is not None:
            cv2.ellipse(original_img, best_ellipse, (0, 255, 0), 1) 
                        
            (center, (fitted_major_axis, fitted_minor_axis), orientation) = best_ellipse
            
            if fitted_major_axis < fitted_minor_axis:
                fitted_major_axis, fitted_minor_axis = fitted_minor_axis, fitted_major_axis
                        
            if fitted_major_axis > 0 and fitted_minor_axis > 0:
                fitted_eccentricity = np.sqrt(1 - (fitted_minor_axis ** 2) / (fitted_major_axis ** 2))
            else:
                fitted_eccentricity = 0  

            output_image_folder = r"Save path"

            os.makedirs(output_image_folder, exist_ok=True)

            output_image_path = os.path.join(output_image_folder, f"{src_filename}")
           
            success = cv2.imwrite(output_image_path, original_img)
            if success:
                print(f"✅ Saved: {output_image_path}")
            else:
                print(f"❌ Failed: {output_image_path}")

            src_original = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                        
            ellipse_mask = np.zeros_like(src_original, dtype=np.uint8) 
            cv2.ellipse(ellipse_mask, best_ellipse, 255, -1) 
           
            if np.sum(ellipse_mask) == 0:                
                continue  
            
            ellipse_pixels = src_original[ellipse_mask == 255]
            ellipse_gray_mean = np.mean(ellipse_pixels) if len(ellipse_pixels) > 0 else 0
            
            internal_pixel_count = np.sum(ellipse_mask == 255)
            
            kernel_size = 1  
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            thick_ellipse_mask = cv2.dilate(ellipse_mask, kernel, iterations=1)
                        
            if np.sum(thick_ellipse_mask) == 0:                
                continue
                        
            total_pixel_count = np.sum(thick_ellipse_mask == 255)
                      
            size_um2 = total_pixel_count * (0.75 ** 2)
            
            if 0 < fitted_eccentricity < 1:
                major_axis_um = np.sqrt((4 * size_um2) / (np.pi * np.sqrt(1 - fitted_eccentricity ** 2)))
                minor_axis_um = (4 * size_um2) / (np.pi * major_axis_um)
            else:
                major_axis_um = 0
                minor_axis_um = 0

            contour_area = cv2.contourArea(best_contour)
            contour_perimeter = cv2.arcLength(best_contour, True)
            hull = cv2.convexHull(best_contour)
            hull_area = cv2.contourArea(hull)
            hull_perimeter = cv2.arcLength(hull, True)

            solidity = (contour_area / hull_area) if hull_area > 0 else np.nan
            convexity = (hull_perimeter / contour_perimeter) if contour_perimeter > 0 else np.nan
            circularity = (4 * np.pi * contour_area) / (contour_perimeter ** 2) if contour_perimeter > 0 else np.nan

            x, y, w, h = cv2.boundingRect(best_contour)
            extent = contour_area / (w * h) if (w * h) > 0 else np.nan

            rect = cv2.minAreaRect(best_contour)
            (rcx, rcy), (rw, rh), rangle = rect
            rect_area = rw * rh if (rw and rh) else 0
            rectangularity = (contour_area / rect_area) if rect_area > 0 else np.nan

            perimeter_um = contour_perimeter * 0.75
            equiv_diameter_um = (np.sqrt(4 * contour_area / np.pi) * 0.75) if contour_area > 0 else np.nan
            aspect_ratio = (fitted_major_axis / fitted_minor_axis) if fitted_minor_axis > 0 else np.nan
            deformation_index = ((fitted_major_axis - fitted_minor_axis) /
                                 (fitted_major_axis + fitted_minor_axis)) if (fitted_major_axis + fitted_minor_axis) > 0 else np.nan
           
            ellipse_sum_intensity = float(np.sum(ellipse_pixels)) if len(ellipse_pixels) > 0 else 0.0
            intensity_std = float(np.std(ellipse_pixels)) if len(ellipse_pixels) > 0 else np.nan
            intensity_cov = (intensity_std / ellipse_gray_mean) if (ellipse_gray_mean and ellipse_gray_mean > 0) else np.nan
           
            data = pd.concat([data, pd.DataFrame({
                "Filename": [src_filename],
                "Major Axis": [major_axis_um],
                "Minor Axis": [minor_axis_um],
                "Size(μm²)": [size_um2],
                "Eccentricity": [fitted_eccentricity],
                "Ellipse Mean Gray Scale": [ellipse_gray_mean],
                "Integrated Density": [ellipse_sum_intensity],
                "Perimeter(μm)": [perimeter_um],
                "Equivalent Diameter(μm)": [equiv_diameter_um],
                "Solidity": [solidity],
                "Convexity": [convexity],
                "Circularity": [circularity],
                "Aspect Ratio": [aspect_ratio],
                "Extent": [extent],
                "Rectangularity": [rectangularity],
                "Deformation Index": [deformation_index],
                "Intensity Std": [intensity_std],
                "Intensity CoV": [intensity_cov],
            })], ignore_index=True)

    with pd.ExcelWriter(os.path.join(folder_path, output_filename), engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False)
