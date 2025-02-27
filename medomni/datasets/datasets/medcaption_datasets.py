"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from medomni.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset
from PIL import Image
import json
import ipdb


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class MedCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_root=vis_root, ann_paths=ann_paths)
        
        prompt_json_file = './medomni/datasets/datasets/prompts.json'
        self.model_type = 'MedOmni'
        with open(prompt_json_file, 'r') as f:
            self.prompt_set = json.load(f)
            f.close()
