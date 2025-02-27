import os
from PIL import Image
import webdataset as wds
from medomni.datasets.datasets.base_dataset import BaseDataset
from medomni.datasets.datasets.medcaption_datasets import MedCaptionDataset
from torch.utils.data import Dataset
from torchvision import transforms
import torchio as tio
import torchvision.transforms.functional as TF
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import random
import nibabel as nib
import pickle as pkl
import ipdb

def aug_seg_2d(image, mask, img_sz=224):
    target_size = img_sz
    # crop = transforms.RandomResizedCrop(target_size)
    resize = transforms.Resize((target_size, target_size))
    resize_mask = transforms.Resize((target_size, target_size), interpolation = Image.NEAREST)
    # params = crop.get_params(image, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    # image = transforms.functional.crop(image, *params)
    # mask = transforms.functional.crop(mask, *params)

    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    image = resize(image)
    mask = resize_mask(mask)

    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalizer = transforms.Normalize(mean, std)
    image = normalizer(image)
    mask = mask != 0
    return image, mask

def aug_seg_3d(img, msk):
    image = tio.ScalarImage(tensor=img)
    label = tio.LabelMap(tensor=msk)
    transform = tio.Compose([
        # tio.CropOrPad((64, 192, 192)),
        # tio.Resize((64, 192, 192)), # tio.RescaleIntensity((-1, 1)),
        # tio.RescaleIntensity((-1, 1)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RandomFlip(),
    ])
    subject = tio.Subject(image=image, label=label)
    outputs = transform(subject)
    image = outputs['image']['data']
    mask = outputs['label']['data']
    return image, mask
    
def merge_mask(mask1, mask2):
    mask = ((mask1 + mask2) > 0)
    return mask

def trans_box(box):
    box = box[1:-1]
    x_min, y_min, x_max, y_max = box.split(',')
    x_min = round(float(x_min) / 224., 2)
    y_min = round(float(y_min) / 224., 2)
    x_max = round(float(x_max) / 224., 2)
    y_max = round(float(y_max) / 224., 2)    
    return str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max)

def trans_points(points):
    points = points[1:-1]
    x, y = points.split(',')
    return str(x)+ ',' + str(y)

def extract(data_info, prompt_set, transform, cont_len=1, img_sz=224, num_imgs=0, question_id=None, subtask=None, incontext=None, tgt_cat=None):
    task_type = data_info['task_type']
    img_paths = ''
    category = ''
    modal = data_info['modal']
    if modal == 'ct':
        image_tensor = torch.zeros((1, 64, 192, 192))
        answer_img = np.zeros((1, 64, 192, 192))
    else:
        image_tensor = torch.zeros((cont_len, 3, 224, 224))
        answer_img = np.zeros((1, 1, 224, 224))
    if task_type == 'report':
        img_paths = data_info['image_path'].split('|||')
        cxt = '<context>'
        if pd.isna(data_info['age']):
            pass
        else:
            cxt += 'Age:' + str(data_info['age']) + '.'
        if pd.isna(data_info['gender']):
            pass
        else:
            cxt += 'Gender:' + data_info['gender'] + '.'
        if pd.isna(data_info['indication']):
            pass
        else:
            cxt += str(data_info['indication'])
        if pd.isna(data_info['comparison']):
            pass
        else:
            cxt += str(data_info['comparison'])
        cxt += '</context>'
        rnd = random.random()
        if subtask is not None:
            pass
        else:
            if rnd <= 0.35 and not pd.isna(data_info['findings']):
                subtask = 'findings'
            elif rnd <= 0.7 and not pd.isna(data_info['impression']):
                subtask = 'impression'
            elif not pd.isna(data_info['findings']) and not pd.isna(data_info['impression']):
                subtask = 'report'
            else:
                if not pd.isna(data_info['findings']): # at least one of them is not empty
                    subtask = 'findings'
                else:
                    subtask = 'impression'
        
        if question_id is None:
            question = random.choice(prompt_set[task_type][subtask])
            question_id = prompt_set[task_type][subtask].index(question)
            question = cxt + question
        else:
            question = prompt_set[task_type][subtask][question_id]
            question = cxt + question            
        
        if not pd.isna(data_info[subtask]):
            answer = data_info[subtask]
        else:
            print(str(answer))
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_tensor.shape[-1],
                    scale=(0.5, 1.0),
                ),
                transforms.ToTensor(),
                transform.transforms[2],
            ]
        )
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        if incontext is None:
            question = question.replace('_*_', main_imgs)
        else:
            question = main_imgs + ':'
    elif task_type == 'classification':
        img_paths = data_info['image_path'].split('|||')
        issue_list = ['airspace opacity',
        'atelectasis',
        'bone lesion',
        'bronchiectasis',
        'calcified nodule',
        'clavicle fracture',
        'consolidation',
        'costophrenic angle blunting',
        'cyst/bullae',
        'diaphragmatic eventration (benign)',
        'elevated hemidiaphragm',
        'enlarged cardiac silhouette',
        'enlarged hilum',
        'hernia',
        'hydropneumothorax',
        'hyperaeration',
        'increased reticular markings/ild pattern',
        'infiltration',
        'linear/patchy atelectasis',
        'lobar/segmental collapse',
        'lung lesion',
        'lung opacity',
        'mass/nodule (not otherwise specified)',
        'mediastinal displacement',
        'mediastinal widening',
        'multiple masses/nodules',
        'pleural effusion',
        'pleural/parenchymal scarring',
        'pneumomediastinum',
        'pneumothorax',
        'pulmonary edema/hazy opacity',
        'scoliosis',
        'shoulder osteoarthritis',
        'spinal degenerative changes',
        'spinal fracture',
        'sub-diaphragmatic air',
        'superior mediastinal mass/enlargement',
        'tortuous aorta',
        'vascular calcification',
        'vascular congestion',
        'vascular redistribution']

        if '|||' in data_info['labels']:
            subtask = 'classification_binary'
            location = data_info['location']
            issue = data_info['labels'].split('|||')[0]
            answer = data_info['labels'].split('|||')[1]
            if question_id is None:
                question = random.choice(prompt_set[subtask])
                question_id = prompt_set[subtask].index(question)
            else:
                question = prompt_set[subtask][question_id]
            question = question.replace('-*-', issue)
            location = ' in the ' + location + '?'
            question = question.replace('?',  location)
        else:
            if random.random() <= 1.0:
                subtask = 'classification_multilabel'
                cxt = '<context>'
                if 'age' in data_info.keys():
                    cxt += 'Age:' + str(data_info['age']) + '.'
                if 'sex' in data_info.keys():
                    cxt += 'Gender:' + data_info['sex'] + '.'
                if 'location' in data_info.keys():
                    cxt += 'Location:' + data_info['location'] + '.'
                cxt += '</context>'
                if question_id is None:
                    question = random.choice(prompt_set[subtask])
                    question_id = prompt_set[subtask].index(question)
                else:
                    question = prompt_set[subtask][question_id]
                question = cxt + question
                answer = data_info['labels']
            else:
                subtask = 'classification_binary'
                issue = random.choice(issue_list)
                if question_id is None:
                    question = random.choice(prompt_set[subtask])
                    question_id = prompt_set[subtask].index(question)
                else:
                    question = prompt_set[subtask][question_id]
                question = question.replace('-*-', issue)
                if issue in data_info['labels'].lower():
                    answer = 'Yes.'
                else:
                    answer = 'No.'
                
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_tensor.shape[-1],
                    scale=(0.5, 1.0),
                ),
                transforms.ToTensor(),
                transform.transforms[2],
            ]
        )
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        question = question.replace('_*_', main_imgs)
    elif task_type == 'detection':
        subtask = task_type
        img_paths = data_info['image_path'].split('|||')
        cls = ''
        answer = ''
        max_num = 9
        boxes = random.sample(data_info['boxes'], random.randint(1, max_num if len(data_info['boxes']) > max_num else len(data_info['boxes'])))
        c1 = 0
        for box in boxes:
            cls += box[0] + ','
            if box[1] == 'n/a':
                answer += box[0] + ',<N/A>.'
            else:
                c1 += 1
                answer += box[0] + ',<DET>.'
        cls = cls[:-1]
        category = cls
        answer += '|||'
        c2 = 0
        for box in boxes:
            if box[1] != 'n/a':
                c2 += 1
                box = trans_box(box[1])
            else:
                box = 'n/a'
            answer += box + ';'
        answer = answer[:-1]
        if c1 != c2:
            ipdb.set_trace()
        if question_id is None:
            question = random.choice(prompt_set[task_type])
            question_id = prompt_set[task_type].index(question)
        else:
            question = prompt_set[task_type][question_id]
        question = question.replace('-*-', cls)
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        if incontext is None:
            question = question.replace('_*_', main_imgs)
        else:
            question = main_imgs + ':'
    elif task_type == 'keypoint':
        subtask = task_type
        img_paths = data_info['image_path'].split('|||')
        cls = ''
        answer = ''
        max_num = 9
        points = data_info['keypoints']
        c1 = 0
        for point in points:
            cls += point[0] + ','
            if point[1] == 'n/a':
                answer += point[0] + ',<N/A>.'
            else:
                c1 += 1
                answer += point[0] + ',<2DPOINT>.'
        cls = cls[:-1]
        category = cls
        answer += '|||'
        c2 = 0
        for point in points:
            if point[1] != 'n/a':
                c2 += 1
                point = trans_points(point[1])
            else:
                point = 'n/a'
            answer += point + ';'
        answer = answer[:-1]
        if c1 != c2:
            ipdb.set_trace()
        if question_id is None:
            question = random.choice(prompt_set[task_type])
            question_id = prompt_set[task_type].index(question)
        else:
            question = prompt_set[task_type][question_id]
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'

        question = question.replace('-*-', cls)
        if incontext is None:
            question = question.replace('_*_', main_imgs)
        else:
            question = main_imgs + ':'
    elif task_type == 'segmentation':
        subtask = task_type
        img_paths = data_info['image_path'].split('|||')
        seg_paths = data_info['seg_path'].split('|||')
        if modal == 'ct':
            data = torch.load(seg_paths[0])
            masks =  data['mask']
            # pass
        else:
            masks = np.load(seg_paths[0])
        if tgt_cat is None:
            if modal == 'ct':
                target_list = list(masks.keys())
                # target_list = ['kidney']
                # mask = torch.ones((1, 64, 192, 192))
            else:
                target_list = masks.files
                mask = np.zeros(masks[target_list[0]].shape)
            target_list = random.sample(target_list, random.randint(1, len(target_list)))
            if modal == 'ct':
                target_list = random.sample(target_list, 1)
        category = tgt_cat
        if question_id is None:
            question = random.choice(prompt_set[task_type])
            question_id = prompt_set[task_type].index(question)
        else:
            question = prompt_set[task_type][question_id]
        tgt_cat = ''
        # target_list = ['spleen']
        for target in target_list:
            tgt_cat += target.lower() + ','
            if modal == 'ct':
                mask = masks[target] > 0
            else:
                mask = merge_mask(mask, masks[target] > 0)

        # answer = '<SEG>.' # commented on Feb 1
        if modal == 'ct':
            answer_img = mask
            answer = 'The segmentation mask of ' + tgt_cat[:-1] + ' is <3DSEG>.'
        else:
            answer_img = Image.fromarray(mask, mode='L')
            answer = 'The segmentation mask of ' + tgt_cat[:-1] + ' is <2DSEG>.'
        tgt_cat = tgt_cat[:-1]
        question = question.replace('-*-', tgt_cat.lower())
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        if incontext is None:
            question = question.replace('_*_', main_imgs)
        else:
            question = main_imgs + ':'
    elif task_type == 'vqa':
        prefix = 'In _*_, '
        img_paths = data_info['image_path'].split('|||')
        question = prefix + data_info['question'].lower()
        answer = data_info['answer']
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        question = question.replace('_*_', main_imgs)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_tensor.shape[-1],
                    scale=(0.5, 1.0),
                ),
                transforms.ToTensor(),
                transform.transforms[2],
            ]
        )
    elif task_type == 'referring':
        img_paths = data_info['image_path'].split('|||')
        box = data_info['box']
        answer = data_info['caption']
        if question_id is None:
            question = random.choice(prompt_set[task_type])
            question_id = prompt_set[task_type].index(question)
        else:
            question = prompt_set[task_type][question_id]
        question = question.replace('-*-', box)
        main_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        question = question.replace('_*_', main_imgs)
    elif task_type == 'temporal':
        img_paths = data_info['main_path'].split('|||')
        past_path = data_info['past_path'].split('|||')
        main_imgs = ''
        ref_imgs = ''
        for i in range(len(img_paths)):
            main_imgs += '<img' + str(num_imgs+i) + '>'
        for i in range(len(past_path)):
            if i + len(img_paths) < cont_len:
                ref_imgs += '<img' + str(num_imgs+len(img_paths)+i) + '>'
        if question_id is None:
            question = random.choice(prompt_set[task_type])
            question_id = prompt_set[task_type].index(question)
        else:
            question = prompt_set[task_type][question_id]
        question = question.replace('-*-', ref_imgs).replace('_*_', main_imgs)
        answer = data_info['answer']
        img_paths.extend(past_path)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_tensor.shape[-1],
                    scale=(0.5, 1.0),
                ),
                transforms.ToTensor(),
                transform.transforms[2],
            ]
        )

    num_imgs_per_std = 0
    for img_path in img_paths:
        try:
            if modal == 'ct':
                image = data['data']
                image = image.unsqueeze(0)
                answer_img = answer_img.unsqueeze(0) # .permute(0, 3, 1, 2)
                # image = torch.randn((1, 64, 192, 192))
                # answer_img = torch.ones((1, 64, 192, 192))
                image, answer_img = aug_seg_3d(image, answer_img)
            else:
                image = Image.open(img_path).convert('RGB') 
                assert(os.path.exists(img_path))  
                if task_type == 'segmentation':
                    image, answer_img = aug_seg_2d(image, answer_img)
                else:
                    image = transform(image)
                # image = image.unsqueeze(-1) # c,w,h,d
        except:
            print(image_path)
            image = np.random.randn(3,img_sz,img_sz)
            
        if num_imgs_per_std < cont_len:
            image_tensor[num_imgs_per_std] = image
            # image_list.append(image)
            num_imgs_per_std += 1
        else:
            break
    
    return question, str(answer), answer_img, image_tensor, num_imgs_per_std, task_type, category, question_id, subtask

class MedDataset(BaseDataset):
    def __init__(self, location):
        super().__init__()

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("png", "json", handler=wds.warn_and_continue),
        )

def incontext_examples(self, task, subtask):
    if subtask in ['findings', 'impression', 'report']:
        for _ in range(10000):
            index = random.randint(0, len(self.annotation[task])-1)
            ann = self.annotation[task][index]
            if not pd.isna(ann[subtask]):
                break
    else:
        index = random.randint(0, len(self.annotation[task])-1)
        ann = self.annotation[task][index]        
    return ann

def combine(question_list, answer_list):
    question = ''
    answer = ''
    for i in range(len(question_list)-1):
        question += question_list[i] + answer_list[i]
    question += question_list[-1]
    answer = answer_list[-1]
    return question, answer

def replace_hints(data):
    data = data.replace('Findings:', '').replace('Impression:', '').replace('findings:', '').replace('impression:', '')
    return data

def ret_ex(annos, subtask, category=None):
    if subtask == 'report' or subtask == 'findings' or subtask == 'impression':
        for _ in range(100000):
            index = random.randint(0, len(annos['report'])-1)
            ann = annos['report'][index]
            if not pd.isna(ann[subtask]):
                break
    elif subtask == 'detection' and category is not None:
        for _ in range(100000):
            index = random.randint(0, len(annos[subtask])-1)
            ann = annos[subtask][index]
            if ann['class'] == category:
                break        
    elif subtask == 'segmentation' and category is not None:
        for _ in range(100000):
            index = random.randint(0, len(annos[subtask])-1)
            ann = annos[subtask][index]
            if not pd.isna(ann[category]):
                break    
    return ann

class MedAlignDataset(MedCaptionDataset):

    def __getitem__(self, index):
        modal = self.target_modal
        task = self.target_task

        ## Uncomment for doing a single task (e.g., keypoint detection on xrays)
        # modal = 'xray'
        # task = 'keypoint'

        index = random.randint(0, len(self.annotation[modal][task])-1)
        ann = self.annotation[modal][task][index]
        cont_len = 5
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        transform = transforms.Compose([                        
                transforms.Resize([224, 224]), # 224 for pretraining, 448 for keypoints
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])  
        
        if task == 'detection':
            num_shot = random.randint(1, 10)
        elif task == 'segmentation':
            num_shot = random.randint(1, 2)
        else:
            num_shot = 1

        question_list = []
        answer_list = []
        num_imgs = 0
        if modal == 'ct':
            image_tensor = torch.zeros((1, 64, 192, 192))
        else:
            image_tensor = torch.zeros((20, 3, 224, 224)) # 20 = max number of input images
        if random.random() <= -1 and task != 'vqa' and task != 'referring' and task != 'temporal' and task != 'classification': # Incontext training is disabled
            question, answer, sub_image_tensor, num_imgs_per_std, task_type, category, question_id, subtask = extract(ann, self.prompt_set, transform, cont_len=cont_len, img_sz=224, num_imgs=0, incontext='yes')
            question_list.append(question)
            answer_list.append(answer)
            num_imgs += num_imgs_per_std
            image_tensor[:num_imgs_per_std] = sub_image_tensor[:num_imgs_per_std]
            for _ in range(num_shot):
                ann = ret_ex(self.annotation, subtask, category)
                question, answer, sub_image_tensor, num_imgs_per_std, task_type, _, _, _ = extract(ann, self.prompt_set, transform, cont_len=cont_len, img_sz=224, num_imgs=num_imgs, question_id=question_id, subtask=subtask, incontext='yes', tgt_cat=category)
                image_tensor[num_imgs:num_imgs+num_imgs_per_std] = sub_image_tensor[:num_imgs_per_std]
                question_list.append(question)
                answer_list.append(answer)
                num_imgs += num_imgs_per_std
            question, answer = combine(question_list, answer_list)
            question, answer = replace_hints(question), replace_hints(answer)
        else:
            question, answer, answer_img, sub_image_tensor, num_imgs_per_std, task_type, _, _, _ = extract(ann, self.prompt_set, transform, cont_len=cont_len, img_sz=224)
            image_tensor[:num_imgs_per_std] = sub_image_tensor[:num_imgs_per_std]
            num_imgs += num_imgs_per_std

        return {
            "image": image_tensor,
            "question": question,
            "answer": answer,
            "answer_img": answer_img,
            "num_imgs": num_imgs,
            "task_type": task_type,
            "modal": modal,
        }