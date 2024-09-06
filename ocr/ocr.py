import os
import cv2
import json
import io
from paddleocr import PaddleOCR
from pathlib import Path

COLUMN_NO = ['A', 'B', 'C']

class Ocr():
    def __init__(self, root_path):
        self.__input_path = os.path.join(root_path, "debug", "temp")
        self.__output_path = os.path.join(root_path, "output")
        self.__ocr = PaddleOCR(use_angle_cls=True, lang='chinese_cht')
        self.__data = []

        if not os.path.isdir(self.__output_path):
            os.makedirs(self.__output_path)

    def __insert_data(self, work_order: str, file_name: str, pos: list, cls: str, text: str) -> None:
        x_pos, y_pos, x2, y2 = pos
        width = int(x2) - int(x_pos)
        height = int(y2) - int(y_pos)
        temp = {}
        temp['工令號'] = work_order
        temp['檔名'] = file_name
        temp['Checkbox_X_Position'] = x_pos
        temp['Checkbox_Y_Position'] = y_pos
        temp['Checkbox_Width'] = width
        temp['Checkbox_Height'] = height
        temp['column_no'] = cls
        temp['Text'] = text
        self.__data.append(temp)

    def run(self):
        work_orders = [os.path.join(self.__input_path, x) for x in os.listdir(self.__input_path)
                        if os.path.isdir(os.path.join(self.__input_path, x))]
        for work_order in work_orders:
            images = [os.path.join(work_order, x) for x in os.listdir(work_order)
                       if os.path.isdir(os.path.join(work_order, x))]
            for image in images:
                print(f"\nFile: {Path(image).stem}")
                blocks = [os.path.join(image, x) for x in os.listdir(image)
                           if os.path.isdir(os.path.join(image, x))]
                for index, block in enumerate(blocks):
                    crops = [os.path.join(block, x) for x in os.listdir(block)]
                    # For each crop, use PaddleOCR to recognize the character.
                    for crop in crops:
                        # Check the position of each crop of the original image.
                        name = os.path.basename(crop)
                        s, e = name.find('('), name.find(')')
                        pos = name[s+1:e].split(',')
                        print(f"Checkbox position -> ({name[s+1:e]})")
                        # OCR process
                        img = cv2.imread(crop)
                        ocr_results = self.__ocr.ocr(img, cls=True)
                        if ocr_results is None or len(ocr_results) == 0:
                            print("辨識結果為空")  # 顯示辨識結果為空
                        else:
                            try:
                                ocr_text = " ".join([line[1][0] for line in ocr_results[0]])
                                # Transfer the OCR result into json formet
                                self.__insert_data(Path(work_order).stem, Path(image).stem, pos, COLUMN_NO[index], ocr_text)
                                print(f"OCR 辨識結果: {ocr_text}")  # 列印 OCR 結果
                            except TypeError:
                                print("OCR 辨識結果包含 None 值，跳過該結果")  # 顯示錯誤信息

        # Write the data into JSON file
        path = os.path.join(self.__output_path, f"result.json")
        with io.open(path, 'w', encoding='utf-8') as file:
            w = json.dumps(self.__data, indent=2, ensure_ascii=False)
            file.write(w)
        print(f"Result JSON file saved to {self.__output_path}.")