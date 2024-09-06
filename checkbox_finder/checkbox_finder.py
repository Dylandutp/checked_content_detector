import cv2
import numpy as np
import os
from pathlib import Path

class CheckboxFinder():
    def __init__(self, root_path, template_threshold=0.8, show_matched=False):
        self.__output_path = os.path.join(root_path, "debug", "temp")
        self.__template_path = os.path.join(root_path, "resources", "checked_checkbox")
        self.__template_threshold = template_threshold
        self.__colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.__show_matched = show_matched
        self.__count = 0
        self.dets = []
        self.cls_dets = []
        self.edges = []

        if not os.path.isdir(self.__output_path):
            os.makedirs(self.__output_path)
        else:
            os.system(f'rm -rf {self.__output_path}')
            os.makedirs(self.__output_path)

    def __initialize(self) -> None:
        self.__count = 0
        self.dets = []
        self.cls_dets = []
        self.edges = []

    def __py_nms(self, dets, thresh) -> list:
        """純 Python NMS 基線。"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def __template(self, img_gray, template) -> list:
        """模板匹配並使用 NMS 過濾重疊的矩形框。"""
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.__template_threshold)
        w, h = template.shape[::-1]
        all_dets = []
        for pt in zip(*loc[::-1]):
            all_dets.append((pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[1], pt[0]]))
        dets = np.array(all_dets)
        # Can not find any part that matches template images
        if len(dets) == 0:
            return [], []
        
        keep = self.__py_nms(dets, 0.3)
        self.dets = sorted(dets[keep], key=lambda x: x[2])
        
        return all_dets

    def __detect_vertical_lines(self) -> list:
        """
        檢測垂直線。
        
        參數:
        dets (list): 檢測到的矩形框列表，每個矩形框表示為 (x1, y1, x2, y2, score)。
        
        返回:
        list: 檢測到的垂直線列表，每條線表示為 (x1, y1, x2, y2)。
        """
        vertical_lines = []
        for box in self.dets:
        # for box in boxes:
            x1, y1, x2, y2 = box[:4]
            vertical_lines.append((x2, y1, x2, y2))
        
        return vertical_lines

    def __merge_vertical_lines(self, lines: list, template_width: int) -> list:
        """
        合併相鄰距離小於模板寬度的垂直線。
        
        參數:
        lines (list): 延伸到頁首和頁尾的垂直線列表，每條線表示為 (x1, y1, x2, y2)。
        template_width (int): 模板的寬度。
        
        返回:
        list: 合併後的垂直線列表。
        """
        if not lines:
            return []

        lines = sorted(lines, key=lambda x: x[0])
        merged_lines = [lines[0]]
    
        for current in lines[1:]:
            previous = merged_lines[-1]
            if current[0] - previous[0] < template_width:
                merged_lines[-1] = (previous[0], previous[1], current[2], current[3])
            else:
                merged_lines.append(current)

        return merged_lines

    def __extend_lines_to_edges(self, lines: list, img_height: int) -> list:
        """
        延伸垂直線到頁首和頁尾。
        
        參數:
        lines (list): 檢測到的垂直線列表，每條線表示為 (x1, y1, x2, y2)。
        img_height (int): 圖像的高度。
        
        返回:
        list: 延伸到頁首和頁尾的垂直線列表。
        """
        extended_lines = []
        for line in lines:
            x1, _, x2, _ = line
            extended_lines.append((x1, 0, x2, img_height))
        return extended_lines

    def __select_lines_for_three_parts(self, lines: list, img_width: int) -> list:
        """
        選擇最可能將圖像分割成三等份的兩條垂直線。

        參數:
        lines (list): 檢測到的垂直線列表，每條線表示為 (x1, y1, x2, y2)。
        template_width (int): 模板的寬度。

        返回:
        list: 選擇的兩條垂直線。
        """
        if len(lines) < 2:
            return lines

        third_width = img_width // 3

        # 找到最接近三分之一和三分之二寬度的兩條垂直線
        lines = sorted(lines, key=lambda x: x[0])
        left_line = min(lines, key=lambda x: abs(x[0] - third_width))
        right_line = min(lines, key=lambda x: abs(x[0] - 2 * third_width))

        return [left_line, right_line]

    def __find_checkbox(self, image):
        img_rgb = image
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # Get all template images in template folder.
        templates = [os.path.join(self.__template_path, f) for f in os.listdir(self.__template_path)]
        # Try to match templates in the target image.
        for idx, template_path in enumerate(templates):
            template = cv2.imread(template_path, 0)
            if template is None:
                print(f"Warning: Template image {template_path} not found or unable to load.")
                continue
            # Find the coordinates of each checkbox on the target image.
            all_dets = self.__template(img_gray, template)
            # If it can not find any checkbox in the image that matched a template, try another template.
            if len(self.dets) == 0:
                continue
                        
            # 檢測垂直線
            vertical_lines = self.__detect_vertical_lines()
            # 延伸垂直線到頁首和頁尾
            extended_lines = self.__extend_lines_to_edges(vertical_lines, img_rgb.shape[0])
            # 合併垂直線
            merged_lines = self.__merge_vertical_lines(extended_lines, template.shape[1])
            # 選擇最可能將圖像分割成三等份的垂直線
            selected_lines = self.__select_lines_for_three_parts(merged_lines, img_rgb.shape[1])

            if self.__show_matched:
                color = self.__colors[idx % len(self.__colors)]
                # 顯示所有匹配結果
                for coord in all_dets:
                    cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 0), 1)
                    cv2.putText(img_rgb, f"{coord[4]:.2f}", (int(coord[0]), int(coord[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
                # 顯示 NMS 過濾後的結果
                for coord in self.dets:
                    self.__count += 1
                    cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), color, 2)
                    cv2.putText(img_rgb, str(self.__count), (int(coord[0]), int(coord[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    print(f"Detection {self.__count}: ({int(coord[0])}, {int(coord[1])}), ({int(coord[2]), int(coord[3])}), Score: {coord[4]:.2f}, IoU Threshold: {self.__template_threshold}")

                # 排序標籤位置
                label_positions = []
                for i, line in enumerate(selected_lines):
                    mid_y = img_rgb.shape[0] // 2
                    label_positions.append((line[0], mid_y, chr(65 + i)))  # 65 是 'A' 的 ASCII 值
                
                # 檢查並調整標籤位置以避免重疊
                label_positions.sort(key=lambda x: x[1])  # 按Y軸排序
                for i in range(1, len(label_positions)):
                    if label_positions[i][1] - label_positions[i-1][1] < 30:  # 假設每個標籤高度為30
                        label_positions[i] = (label_positions[i][0], label_positions[i-1][1] + 30, label_positions[i][2])
                
                # 繪製垂直線和標籤
                for x, y, label in label_positions:
                    cv2.line(img_rgb, (int(x), 0), (int(x), img_rgb.shape[0]), (0, 255, 255), 2)
                    cv2.putText(img_rgb, label, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

            # Extract the the edge of each block. (only x_axis)
            self.edges = [0]
            for line in selected_lines:
                self.edges.append(int(line[2]) + 5)

            # Classify all detected checkbox into 3 groups. (According to its position)
            temp = []
            i = 0
            for idx, coord in enumerate(self.dets):
                # After the second deviding line is the last(third) group
                if i >= len(selected_lines):
                    self.cls_dets.append(self.dets[idx-1:])
                    break
                if int(coord[2]) <= int(selected_lines[i][2]):
                    temp.append(coord)
                else:
                    self.cls_dets.append(temp)
                    temp = [coord]
                    i += 1

            # Sort all the checkbox pos according to its y_axis (from top to bottom).
            temp = []
            for dets in self.cls_dets:
                dets = sorted(dets, key = lambda x : x[1])
                temp.append(dets)
            self.cls_dets = temp
            return img_rgb

    def run(self, file_path, work_order):
        self.__initialize()
        img_rgb = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_rgb is None:
            print(f"{file_path} -> Can't open/read the image {file_path}")
            return
        # Romove the target images' title part.
        img_rgb = img_rgb[1000:, :, :]
        
        img_rgb = self.__find_checkbox(img_rgb)
        # Can not find any part that matches template images
        if len(self.dets) == 0:
            print(f"Can't find any checkbox in '{Path(file_path).stem}'.")
            return
        
        # Crop images for OCR use.
        output_path = os.path.join(self.__output_path, work_order, Path(file_path).stem)
        for i, dets in enumerate(self.cls_dets):
            output = os.path.join(output_path, f"block_{i+1}")
            if not os.path.isdir(output):
                os.makedirs(output)
            for idx, det in enumerate(dets, start=1):
                x1, y1, x2, y2, _ = det
                crop_img = img_rgb[int(y1)-10: int(y2), self.edges[i]: int(x1)]
                cv2.imwrite(os.path.join(output, f"cropped_{idx}_({int(x1)},{int(y1)},{int(x2)},{int(y2)}).jpg"), crop_img)
        if self.__show_matched:
            output = f"{output_path}_checked.jpg"
            cv2.imwrite(output, img_rgb)

        print(f"Result image saved to {self.__output_path}.")