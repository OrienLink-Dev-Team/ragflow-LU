
import os
from PIL import Image
import time

import cv2

from paddleocr.ppstructure.table.predict_table import TableSystem
from paddleocr.ppstructure.utility import init_args
from deepdoc.parser.mineru_lib_constants import *

import numpy as np

class ppTableModel(object):
    """
        This class is responsible for converting image of table into HTML format using a pre-trained model.

        Attributes:
        - table_sys: An instance of TableSystem initialized with parsed arguments.

        Methods:
        - __init__(config): Initializes the model with configuration parameters.
        - img2html(image): Converts a PIL Image or NumPy array to HTML string.
        - parse_args(**kwargs): Parses configuration arguments.
    """

    def __init__(self, config):
        """
        Parameters:
        - config (dict): Configuration dictionary containing model_dir and device.
        """
        args = self.parse_args(**config)
        # print('args = ', args)
        self.table_sys = TableSystem(args)

    def img2html(self, image):
        """
        Parameters:
        - image (PIL.Image or np.ndarray): The image of the table to be converted.

        Return:
        - HTML (str): A string representing the HTML structure with content of the table.
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pred_res, _ = self.table_sys(image)
        pred_html = pred_res["html"]
        # res = '<td><table  border="1">' + pred_html.replace("<html><body><table>", "").replace(
        # "</table></body></html>","") + "</table></td>\n"
        return pred_html

    def parse_args(self, **kwargs):
        parser = init_args()
        model_dir = kwargs.get("model_dir")
        table_model_dir = os.path.join(model_dir, TABLE_MASTER_DIR)
        table_char_dict_path = os.path.join(model_dir, TABLE_MASTER_DICT)
        det_model_dir = os.path.join(model_dir, DETECT_MODEL_DIR)
        rec_model_dir = os.path.join(model_dir, REC_MODEL_DIR)
        rec_char_dict_path = os.path.join(model_dir, REC_CHAR_DICT)
        device = kwargs.get("device", "cpu")
        use_gpu = True if device.startswith("cuda") else False
        config = {
            "use_gpu": use_gpu,
            "table_max_len": kwargs.get("table_max_len", TABLE_MAX_LEN),
            "table_algorithm": "TableMaster",
            "table_model_dir": table_model_dir,
            "table_char_dict_path": table_char_dict_path,
            "det_model_dir": det_model_dir,
            "rec_model_dir": rec_model_dir,
            "rec_char_dict_path": rec_char_dict_path,
        }
        parser.set_defaults(**config)
        return parser.parse_args([])



def table_model_init(table_model_type, model_path, max_time, _device_='cpu'):
    if table_model_type == 'STRUCT_EQTABLE':
        pass
    else:
        config = {
            "model_dir": model_path,
            "device": _device_
        }
        table_model = ppTableModel(config)
    return table_model

class AtomicModel:
    Layout = "layout"
    MFD = "mfd"
    MFR = "mfr"
    OCR = "ocr"
    Table = "table"

class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):
        if atom_model_name not in self._models:
            self._models[atom_model_name] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[atom_model_name]
    

def atom_model_init(model_name: str, **kwargs):

  
    atom_model = table_model_init(
        kwargs.get("table_model_type"),
        kwargs.get("table_model_path"),
        kwargs.get("table_max_time"),
        kwargs.get("device")
    )

    return atom_model




if __name__ == "__main__":
    table_model_dir = "TabRec/TableMaster"
            # self.table_model = table_model_init(self.table_model_type, str(os.path.join(models_dir, table_model_dir)),
        #                                     max_time=self.table_max_time, _device_=self.device)
    start_time = time.time()
    atom_model_manager = AtomModelSingleton()
    table_model = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.Table,
        table_model_type="TableMaster",
        table_model_path='/root/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit/snapshots/a29caa466f6d07be0e4863bba64204009128931a/models/TabRec/TableMaster',
        table_max_time=400,
        device="cuda"
    )
    end_time1 = time.time()
    print(f"Time taken: {end_time1 - start_time} seconds")
    image_dir = "images"
    image_path_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    for image_path in image_path_list:
        image = Image.open(image_path)
        print(image)
        html_code = table_model.img2html(image)
        end_time = time.time()
        print(f"Time taken: {end_time - end_time1} seconds, shape = {image.size}, image_path = {image_path}")
        end_time1 = end_time
    print(html_code)

