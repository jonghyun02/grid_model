import cv2
import numpy as np
from pathlib import Path

MIN_BLACK_RATIO = 0.005 

def grid_extract(image_path, output_dir, debug=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    
    if img is None:
        return 0
    
    grid_count = extract_by_grid_lines(img, output_dir, debug)
      
    return grid_count


def get_black_ratio(img_roi):
    if img_roi is None or img_roi.size == 0:
        return 0.0
        
    if len(img_roi.shape) == 3:
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_roi
        
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    black_pixels = np.sum(binary == 0)
    total_pixels = binary.size
    
    return black_pixels / total_pixels

def cluster_lines(lines, threshold=15):
    if not lines: return []
    lines.sort()
    clusters = [[lines[0]]]
    for line in lines[1:]:
        if line - clusters[-1][-1] <= threshold:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return [int(sum(c)/len(c)) for c in clusters]

def extract_by_grid_lines(img, output_dir, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                            minLineLength=min(width, height)//5, maxLineGap=20)
    
    if lines is None:
        return 0
    
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 10 or angle > 170: h_lines.append((y1 + y2) // 2)
        elif 80 < angle < 100: v_lines.append((x1 + x2) // 2)
            
    h_lines = cluster_lines(h_lines, threshold=15)
    v_lines = cluster_lines(v_lines, threshold=15)
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return 0
        
    if 0 not in h_lines: h_lines.insert(0, 0)
    if height not in h_lines: h_lines.append(height)
    if 0 not in v_lines: v_lines.insert(0, 0)
    if width not in v_lines: v_lines.append(width)
    
    h_lines.sort()
    v_lines.sort()

  
    if debug:
        debug_img = img.copy()
        for y in h_lines: cv2.line(debug_img, (0, y), (width, y), (0, 255, 0), 2)
        for x in v_lines: cv2.line(debug_img, (x, 0), (x, height), (255, 0, 0), 2)
        cv2.imwrite(f"{output_dir}/debug_grid_lines.png", debug_img)

    count = 0
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i+1]
            x1, x2 = v_lines[j], v_lines[j+1]
            
            if (y2-y1) < 20 or (x2-x1) < 20: continue
            
            margin = 5
            roi = img[y1+margin:y2-margin, x1+margin:x2-margin]
            
            ratio = get_black_ratio(roi)
            
            if ratio >= MIN_BLACK_RATIO:
                output_path = f"{output_dir}/grid_{count:03d}.png"
                cv2.imwrite(output_path, roi)
                count += 1
                
    return count

if __name__ == "__main__":
    for i in range(0, 9):
        f = f"test{i+1}.png"
        
        try:
            output = f"result_{f.split('.')[0]}"
            grid_extract(f, output_dir=output, debug=True)
        except Exception as e:

            print(f"[{f}] 처리 제외: {e}\n")

