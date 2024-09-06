import os
import configparser
from checkbox_finder import CheckboxFinder
from config_reader import ConfigReader
from ocr import Ocr
from pdf2image import convert_from_path
from pathlib import Path

def convert_from_pdf(dir_pth):
    for file in os.listdir(dir_pth):
        file_path = os.path.join(dir_pth, file)
        if file.endswith(".pdf"):
            images = convert_from_path(file_path)
            for i, image in enumerate(images, start=1):
                image_path = f"{file_path[:-4]}_{i}.jpg"
                image.save(image_path, 'JPEG')

def main():
    cf = ConfigReader()
    ROOT_PATH = cf.get_root_path()

    checkbox_finder = CheckboxFinder(ROOT_PATH, show_matched=True)
    ocr = Ocr(ROOT_PATH)

    input_path = os.path.join(ROOT_PATH, "input")
    input_list = [ os.path.join(input_path, x) for x in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, x))]
    for dir in input_list:
        work_orders = [ os.path.join(dir, x) for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]
        for work_order in work_orders:
            print(f"\nwork order: {Path(work_order).stem}")
            convert_from_pdf(work_order)
            input_files = [ os.path.join(work_order, x) for x in os.listdir(work_order) if x.endswith(".jpg")]
            for input_file in input_files:
                checkbox_finder.run(input_file, Path(work_order).stem)

    ocr.run()