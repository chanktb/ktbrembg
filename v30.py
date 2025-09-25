# -*- coding: utf-8 -*-

"""
Script tự động xử lý ảnh. (Phiên bản v29.1 - Xử lý hàng loạt & Thử nhiều Tolerance)
- Chạy lần lượt với tất cả các ảnh trong thư mục InputImage.
- Với mỗi ảnh, tự động thử nhiều giá trị MAGICWAND_TOLERANCE (20, 30, 40, 50).
- Lưu kết quả với tên file tương ứng với giá trị tolerance đã dùng.
"""

import os
import cv2
import numpy as np
from PIL import Image
import subprocess # <<< THÊM MỚI 1: Import module để chạy lệnh command line
from datetime import datetime # <<< THÊM MỚI 2: Import để lấy ngày giờ cho commit message
# ==============================================================================
# CẤU HÌNH DỰ ÁN
# ==============================================================================
INPUT_FOLDER = "InputImage"
OUTPUT_FOLDER = "OutputImage"
CANVAS_WIDTH = 4200
CANVAS_HEIGHT = 4800
PADDING_TOP = 50
PADDING_LEFT = 50
PADDING_RIGHT = 50
TARGET_DPI = 300

# --- CẤU HÌNH TINH CHỈNH VIỀN ---
# Độ phân giải mục tiêu để làm việc khi tinh chỉnh viền.
# Giá trị cao hơn (vd: 6000, 8000) cho viền mịn hơn, nhưng xử lý chậm hơn.
REFINE_TARGET_SIZE = 10000

# ==============================================================================
# HẾT PHẦN CẤU HÌNH
# ==============================================================================
# <<< THÊM MỚI 3: Hàm để chạy các lệnh Git và tự động push >>>
def git_push_results():
    """Tự động thêm, commit và push các kết quả lên GitHub."""
    try:
        print("\n" + "="*60)
        print("🚀 Bắt đầu quá trình tự động push lên GitHub...")

        # Lấy ngày giờ hiện tại để tạo commit message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Auto-commit: Processed images on {current_time}"

        # Chạy lệnh 'git add .'
        print("   1. Đang thêm các file vào staging area (git add .)...")
        subprocess.run(["git", "add", "."], check=True)

        # Chạy lệnh 'git commit'
        print(f"   2. Đang commit với message: '{commit_message}'...")
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Chạy lệnh 'git push'
        print("   3. Đang đẩy các thay đổi lên remote repository (git push)...")
        subprocess.run(["git", "push"], check=True)

        print("✅ Push lên GitHub thành công!")
        print("="*60)

    except FileNotFoundError:
        print("❌ LỖI: Lệnh 'git' không tồn tại. Hãy chắc chắn rằng bạn đã cài đặt Git.")
    except subprocess.CalledProcessError as e:
        print(f"❌ LỖI: Một lệnh Git đã thất bại (mã lỗi: {e.returncode}).")
        print(f"   Lỗi chi tiết: {e.stderr}")
        print("   Vui lòng kiểm tra lại cấu hình Git, remote repository và quyền truy cập.")
    except Exception as e:
        print(f"❌ Đã có lỗi không xác định xảy ra: {e}")

def vector_like_refine_mask(mask, target_size=REFINE_TARGET_SIZE):
    """Sử dụng kỹ thuật Dò và Vẽ lại Viền để tạo ra mặt nạ sắc nét."""
    print(f"✨ Thực hiện tinh chỉnh viền sắc nét (Làm việc ở độ phân giải ~{target_size}px)...")
    if np.all(mask == 0):
        print("⚠️ Cảnh báo: Mặt nạ trống, không có gì để tinh chỉnh.")
        return mask
        
    h_orig, w_orig = mask.shape
    # Tự động tính hệ số phóng to dựa trên kích thước ảnh và độ phân giải mục tiêu
    scale_factor = max(1, int(target_size / max(h_orig, w_orig, 1)))
    
    if scale_factor <= 1:
        print("ℹ️ Ảnh đã đủ lớn, chỉ thực hiện khử nhiễu cơ bản.")
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    h_up, w_up = h_orig * scale_factor, w_orig * scale_factor
    upscaled_mask = cv2.resize(mask, (w_up, h_up), interpolation=cv2.INTER_CUBIC)
    _, binary_upscaled_mask = cv2.threshold(upscaled_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_upscaled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    perfect_mask_upscaled = np.zeros((h_up, w_up), dtype=np.uint8)
    cv2.drawContours(perfect_mask_upscaled, contours, -1, (255), thickness=cv2.FILLED)
    final_mask = cv2.resize(perfect_mask_upscaled, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
    _, final_mask_binary = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    print("✅ Viền đã được tái tạo sắc nét.")
    return final_mask_binary

# <<< THAY ĐỔI 1: Thêm tham số 'magicwand_tolerance' vào hàm >>>
def process_image(input_path, output_path, magicwand_tolerance):
    print(f"🚀 Bắt đầu xử lý file: {os.path.basename(input_path)} với Tolerance = {magicwand_tolerance}")
    original_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if original_image is None: return print(f"❌ Lỗi: Không thể đọc file ảnh {input_path}")

    # --- Bước 1: Tách nền bằng Logic Magic Wand Toàn cục ---
    bgr_image = original_image[:,:,:3] if len(original_image.shape) > 2 and original_image.shape[2] == 4 else original_image
    h, w = bgr_image.shape[:2]

    # --- CẢI TIẾN: Lấy màu TRUNG BÌNH ở 4 góc để tăng độ chính xác ---
    sample_size = 10
    corners = [
        bgr_image[0:sample_size, 0:sample_size],                       # Top-left
        bgr_image[0:sample_size, w-sample_size:w],                     # Top-right
        bgr_image[h-sample_size:h, 0:sample_size],                     # Bottom-left
        bgr_image[h-sample_size:h, w-sample_size:w]                    # Bottom-right
    ]
    # Tính toán màu trung bình cho mỗi góc
    corner_colors = [np.mean(corner, axis=(0, 1)) for corner in corners]
    
    combined_background_mask = np.zeros((h, w), np.uint8)

    for color in corner_colors:
        color_int = color.astype(int)
        # <<< THAY ĐỔI 2: Sử dụng giá trị tolerance được truyền vào thay vì biến toàn cục >>>
        tolerance = magicwand_tolerance
        lower_bound = np.array([max(0, c - tolerance) for c in color_int])
        upper_bound = np.array([min(255, c + tolerance) for c in color_int])
        
        mask = cv2.inRange(bgr_image, lower_bound, upper_bound)
        combined_background_mask = cv2.bitwise_or(combined_background_mask, mask)

    inverse_mask = cv2.bitwise_not(combined_background_mask)
    print("✅ Tách nền bằng Magic Wand toàn cục thành công.")
    
    # --- Bước 2: Tinh chỉnh viền ---
    refined_mask = vector_like_refine_mask(inverse_mask, target_size=REFINE_TARGET_SIZE)
    
    # --- Bước 3: Áp dụng mặt nạ và Cắt gọn ---
    bgra_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra_image[:, :, 3] = refined_mask

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return print("❌ Lỗi: Không tìm thấy đối tượng.")
    
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    cropped_image = bgra_image[y : y + h, x : x + w]
    print(f"✅ Cắt gọn đối tượng. Kích thước gốc: {w}x{h}px")

    # --- Bước 4: Scale ---
    if w == 0 or h == 0: return print("❌ Lỗi: Kích thước ảnh sau khi cắt không hợp lệ.")
    safe_area_width = CANVAS_WIDTH - PADDING_LEFT - PADDING_RIGHT
    safe_area_height = CANVAS_HEIGHT - PADDING_TOP
    img_aspect_ratio = w / h
    safe_area_aspect_ratio = safe_area_width / safe_area_height
    if img_aspect_ratio > safe_area_aspect_ratio:
        target_w = safe_area_width
        target_h = int(target_w / img_aspect_ratio)
    else:
        target_h = safe_area_height
        target_w = int(target_h * img_aspect_ratio)
    scaled_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    print(f"✅ Scale ảnh. Kích thước mới: {target_w}x{target_h}px")
    
    # --- Bước 5: Làm nét ---
    bgr = scaled_image[:, :, :3]
    alpha = scaled_image[:, :, 3]
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3)
    sharpened_bgr = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)
    sharpened_image_cv = cv2.merge([sharpened_bgr, alpha])
    print("✅ Làm nét ảnh thành công.")

    # --- Bước 6: Đặt vào khung ---
    sharpened_image_pil = Image.fromarray(cv2.cvtColor(sharpened_image_cv, cv2.COLOR_BGRA2RGBA))
    canvas = Image.new('RGBA', (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    paste_x = (CANVAS_WIDTH - target_w) // 2
    paste_y = PADDING_TOP
    paste_coords = (paste_x, paste_y)
    print(f"🎨 Căn giữa và dán ảnh tại tọa độ (X, Y): {paste_coords}")
    canvas.paste(sharpened_image_pil, paste_coords, mask=sharpened_image_pil)
    print(f"✅ Đặt thiết kế vào khung thành công.")

    # --- Bước 7: Lưu ảnh cuối cùng ---
    canvas.save(output_path, 'PNG', dpi=(TARGET_DPI, TARGET_DPI))
    print(f"🎉 Hoàn thành! File đã được lưu tại: {output_path}")
    print("-" * 50) # Thêm dòng ngăn cách cho dễ nhìn

def main():
    print("=======================================================================")
    print("=== SCRIPT XỬ LÝ ẢNH IN ÁO - KTB REMBG TOOL (v29.1 - Multi-Tol) ===")
    print("=======================================================================")
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"📂 Thư mục '{INPUT_FOLDER}' đã được tạo. Vui lòng thêm ảnh vào và chạy lại script.")
        return
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    # <<< THAY ĐỔI 3: Danh sách các giá trị tolerance cần thử >>>
    tolerances_to_test = [60, 70]
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]
    if not files: 
        print(f"📂 Không tìm thấy file ảnh nào trong thư mục '{INPUT_FOLDER}'.")
        return

    total_files = len(files)
    total_processes = total_files * len(tolerances_to_test)
    current_process = 0
    
    # <<< THAY ĐỔI 4: Vòng lặp để xử lý TẤT CẢ các file >>>
    for image_file in files:
        # <<< THAY ĐỔI 5: Vòng lặp lồng nhau để thử TẤT CẢ các giá trị tolerance >>>
        for tolerance_value in tolerances_to_test:
            current_process += 1
            print(f"\n🔄 XỬ LÝ LƯỢT {current_process}/{total_processes} 🔄")
            
            input_file_path = os.path.join(INPUT_FOLDER, image_file)
            filename, _ = os.path.splitext(image_file)
            
            # <<< THAY ĐỔI 6: Tạo tên file đầu ra độc nhất, chứa giá trị tolerance >>>
            output_filename = f"{filename}_tol{tolerance_value}_processed.png"
            output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Gọi hàm xử lý với giá trị tolerance cụ thể
            process_image(input_file_path, output_file_path, tolerance_value)
            
    print("\n=======================================================================")
    print(f"✅✅✅ ĐÃ XỬ LÝ XONG TOÀN BỘ {total_files} ẢNH! ✅✅✅")
    print("=======================================================================")
    # <<< THÊM MỚI 4: Gọi hàm push lên GitHub sau khi tất cả công việc hoàn thành >>>
    git_push_results()

if __name__ == "__main__":
    main()