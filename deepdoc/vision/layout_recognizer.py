#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys
sys.path.append('/root/miniconda3/envs/MinerU/lib/python3.10/site-packages')

import os
import re
from collections import Counter
from copy import deepcopy
import numpy as np
from huggingface_hub import snapshot_download

from api.utils.file_utils import get_project_base_directory
from deepdoc.vision import Recognizer


import copy
import fitz
import numpy as np
from PIL import Image, ImageDraw
import time
import torch
from mineru.magic_pdf.model.pek_sub_modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from mineru.magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from mineru.magic_pdf.rw.DiskReaderWriter import DiskReaderWriter



class LayoutRecognizer():
    # labels = [
    #     "_background_",
    #     "Text",
    #     "Title",
    #     "Figure",
    #     "Figure caption",
    #     "Table",
    #     "Table caption",
    #     "Header",
    #     "Footer",
    #     "Reference",
    #     "Equation",
    # ]
    labels = [
        "title",  # 标题
        "plain_text",  # 文本
        "abandon",  # 包括页眉页脚页码和页面注释
        "figure",  # 图片
        "figure_caption",  # 图片描述
        "table",  # 表格
        "table_caption",  # 表格描述
        "table_footnote",  # 表格注释
        "isolate_formula",  # 行间公式
        "formula_caption",  # 行间公式的标号
    ]
    
    def __init__(self):
        self.weights = "/root/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit/snapshots/a29caa466f6d07be0e4863bba64204009128931a/models/Layout/model_final.pth"
        self.config_file = "/root/miniconda3/envs/MinerU/lib/python3.10/site-packages/magic_pdf/resources/model_config/layoutlmv3/layoutlmv3_base_inference.yaml"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layout_model = Layoutlmv3_Predictor(self.weights, self.config_file, self.device)

        self.garbage_layouts = ["footer", "header", "reference"]

    
    def layout_inference(self, images):
        # label_map = {
        #     0: 'title', 1: 'plain_text', 2: 'abandon', 3: 'figure',
        #     4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote'
        # }
        label_map = {
            0: 'title', 1: 'text', 2: 'reference', 3: 'figure',
            4: 'figure caption', 5: 'table', 6: 'table caption', 7: 'table footnote'
        }
        colors = {
            'title': 'red', 'text': 'blue', 'reference': 'green', 'figure': 'yellow',
            'figure caption': 'purple', 'table': 'orange', 'table caption': 'pink', 'table footnote': 'cyan'
        }
        layouts = []
        for page_index, img in enumerate(images):
            start_time = time.time()
            layout_dets = self.layout_model(np.array(img))
            print("===========================================")
            print(f"Page {page_index} detected {time.time() - start_time:.2f} seconds")
            assert isinstance(layout_dets, list), "Output should be a list"
            
            page_img = copy.deepcopy(img)
            draw = ImageDraw.Draw(page_img)
            
            filtered_layout_dets = []
            for det in layout_dets:
                assert "category_id" in det, "Detection should have 'category_id'"
                assert "poly" in det, "Detection should have 'poly'"
                assert "score" in det, "Detection should have 'score'"
                
                # Only process categories 0-7
                if det['category_id'] in label_map:
                    # Convert poly to bbox
                    poly = det['poly']
                    bbox = [poly[0], poly[1], poly[4], poly[5]]
                    
                    # Replace category_id with type
                    det['type'] = label_map[det.pop('category_id')]
                    det['bbox'] = bbox
                    del det['poly']
                    filtered_layout_dets.append(det)

                    # Draw bounding box
                    draw.rectangle(bbox, width=4,outline=colors[det['type']])
                    
                    text_position = (bbox[0], bbox[1] - 10)  # Adjust position as needed
                    draw.text(text_position, det['type'], fill=colors[det['type']])
                
                else:
                    print("==================== det['category_id'] not in label_map ==================== ")
                    print(f"page_index: {page_index}, det: {det}")


                
            # Save the image with detections
            output_dir = "/mnt/mydisk/wenke.deng/ragflow-LU/mineru/output/test"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = f"{output_dir}/output_page_{len(os.listdir(output_dir))}.png"
            page_img.save(output_path)
            # print(f"Page {page_index} Layout Detections:", layout_dets)
            print(f"Page {page_index} processed in {time.time() - start_time:.2f} seconds")
            
            layouts.append(filtered_layout_dets)
        return layouts
    
    
    def __call__(self, image_list, ocr_res, scale_factor=3,
                 thr=0.2, batch_size=16, drop=True):
        def __is_garbage(b):
            patt = [r"^•+$", r"(版权归©|免责条款|地址[:：])", r"\.{3,}", "^[0-9]{1,2} / ?[0-9]{1,2}$",
                    r"^[0-9]{1,2} of [0-9]{1,2}$", "^http://[^ ]{12,}",
                    "(资料|数据)来源[:：]", "[0-9a-z._-]+@[a-z0-9-]+\\.[a-z]{2,3}",
                    "\\(cid *: *[0-9]+ *\\)"
                    ]
            return any([re.search(p, b["text"]) for p in patt])

        
        # layouts = self.layout_model(image_list, thr, batch_size)
        layouts = []
        layouts_list = self.layout_inference(image_list)
        #print(f"================= layouts recognizer result =================")
        for p_i, p_layout in enumerate(layouts_list):
            layouts.append([layout for layout in p_layout if layout["score"] >= thr])
            # print(type(p_layout))
            # print(f"p_i: {p_i}, p_layout: {p_layout}")
        
        # save_results(image_list, layouts, self.labels, output_dir='output/', threshold=0.7)
        assert len(image_list) == len(ocr_res)
        # Tag layout type
        boxes = []
        assert len(image_list) == len(layouts)
        garbages = {}
        page_layout = []
        print(f"================= layouts  =================")
        
        for pn, lts in enumerate(layouts):
            bxs = ocr_res[pn]
            
            lts = [{"type": b["type"],
                    "score": float(b["score"]),
                    "x0": b["bbox"][0] / scale_factor, "x1": b["bbox"][2] / scale_factor,
                    "top": b["bbox"][1] / scale_factor, "bottom": b["bbox"][-1] / scale_factor,
                    "page_number": pn,
                    } for b in lts if float(b["score"]) >= 0.8 or b["type"] not in self.garbage_layouts]
            lts = Recognizer.sort_Y_firstly(lts, np.mean(
                [l["bottom"] - l["top"] for l in lts]) / 2)
            lts = Recognizer.layouts_cleanup(bxs, lts)
            
            page_layout.append(lts)
            
            # Tag layout type, layouts are ready
            def findLayout(ty):
                nonlocal bxs, lts, self, pn
                lts_ = [lt for lt in lts if lt["type"] == ty]
                
                i = 0
                while i < len(bxs):
                    
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue
                    if __is_garbage(bxs[i]):
                        bxs.pop(i)
                        continue
                    
                    ii = Recognizer.find_overlapped_with_threashold(bxs[i], lts_,
                                                              thr=0.4)
                    if ii is None:  # belong to nothing
                        bxs[i]["layout_type"] = ""
                        i += 1
                        continue
                    lts_[ii]["visited"] = True
                    keep_feats = [
                        lts_[
                            ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].size[1] * 0.9 / scale_factor,
                        lts_[
                            ii]["type"] == "header" and bxs[i]["top"] > image_list[pn].size[1] * 0.1 / scale_factor,
                    ]
                    if drop and lts_[
                            ii]["type"] in self.garbage_layouts and not any(keep_feats):
                        if lts_[ii]["type"] not in garbages:
                            garbages[lts_[ii]["type"]] = []
                        garbages[lts_[ii]["type"]].append(bxs[i]["text"])
                        bxs.pop(i)
                        continue

                    bxs[i]["layoutno"] = f"{ty}-{ii}"
                    bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[
                        ii]["type"] != "equation" else "figure"
                    i += 1

            for lt in ["footer", "header", "reference", "figure caption",
                "table caption", "title", "table", "text", "figure", "equation"]:
                findLayout(lt)
            
            # add box to figure layouts which has not text box
            for i, lt in enumerate(
                    [lt for lt in lts if lt["type"] in ["figure", "equation"]]):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                del lt["type"]
                lt["text"] = ""
                lt["layout_type"] = "figure"
                lt["layoutno"] = f"figure-{i}"
                bxs.append(lt)

            boxes.extend(bxs)

        ocr_res = boxes

        garbag_set = set()
        for k in garbages.keys():
            garbages[k] = Counter(garbages[k])
            for g, c in garbages[k].items():
                if c > 1:
                    garbag_set.add(g)

        ocr_res = [b for b in ocr_res if b["text"].strip() not in garbag_set]
        return ocr_res, page_layout
