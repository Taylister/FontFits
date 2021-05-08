import os
from skimage import io
from skimage.transform import resize
import numpy as np
import random
import cfg
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TGNet_dataset(Dataset):
    def __init__(self, data_dir, torp, ex_n, transforms = None):
        
        self.data_dir = data_dir
        self.batch_size = cfg.batch_size
        self.data_shape = cfg.data_shape
        self.torp = torp
        self.ex_num = ex_n

        if self.ex_num == 1:
            if self.torp == 'train':
                self.name_list = os.listdir(os.path.join(self.data_dir,cfg.style,cfg.train_data_dir))
            
            elif self.torp == 'validation':
                self.name_list = os.listdir(os.path.join(self.data_dir,cfg.style,cfg.validation_data_dir))
            
            else:
                self.name_list = os.listdir(os.path.join(self.data_dir,cfg.style))
        else:
            if self.torp == 'train':
                self.name_list = os.listdir(os.path.join(self.data_dir,cfg.style_mask,cfg.train_data_dir))
            
            elif self.torp == 'validation':
                self.name_list = os.listdir(os.path.join(self.data_dir,cfg.style_mask,cfg.validation_data_dir))
            
            else:
                self.name_list = os.listdir(os.path.join(self.data_dir,cfg.style_mask))
      
    def __len__(self):

        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        if self.ex_num == 1:
            if self.torp == 'train':
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, cfg.train_data_dir, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, cfg.train_data_dir, img_name), as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, cfg.train_data_dir, img_name))

                c_mask = io.imread(os.path.join(self.data_dir, cfg.c_mask, cfg.train_data_dir, img_name), as_gray = True)
                c_skeleton = io.imread(os.path.join(self.data_dir, cfg.c_skeleton, cfg.train_data_dir, img_name), as_gray = True)

                e_title = io.imread(os.path.join(self.data_dir, cfg.extracted_title, cfg.train_data_dir, img_name))

            elif self.torp == 'validation':
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, cfg.validation_data_dir, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, cfg.validation_data_dir, img_name), as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, cfg.validation_data_dir, img_name))

                c_mask = io.imread(os.path.join(self.data_dir, cfg.c_mask, cfg.validation_data_dir, img_name), as_gray = True)
                c_skeleton = io.imread(os.path.join(self.data_dir, cfg.c_skeleton, cfg.validation_data_dir, img_name), as_gray = True)

                e_title = io.imread(os.path.join(self.data_dir, cfg.extracted_title, cfg.validation_data_dir, img_name))

            else:
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, img_name), as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, img_name))

                c_mask = io.imread(os.path.join(self.data_dir, cfg.c_mask, img_name), as_gray = True)
                c_skeleton = io.imread(os.path.join(self.data_dir, cfg.c_skeleton, img_name), as_gray = True)

                e_title = io.imread(os.path.join(self.data_dir, cfg.extracted_title, img_name))
        
        else:
            # filename = "0761525696_01.jpg"
            filename_without_ext, ext = os.path.splitext(os.path.basename(img_name))
            # filename_without_ext = "0761525696_01" ,ext = ".jpg"
            style_filename_without_ext = filename_without_ext.rsplit("_",1)[0]
            # style_filename_without_ext = "0761525696"
            style_filename = style_filename_without_ext + ext

            if self.torp == 'train':
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, cfg.train_data_dir, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, cfg.train_data_dir, img_name), as_gray = True)

                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, cfg.train_data_dir, style_filename))

                c_mask = io.imread(os.path.join(self.data_dir, cfg.c_mask, cfg.train_data_dir, img_name), as_gray = True)
                c_skeleton = io.imread(os.path.join(self.data_dir, cfg.c_skeleton, cfg.train_data_dir, img_name), as_gray = True)

                e_title = io.imread(os.path.join(self.data_dir, cfg.extracted_title, cfg.train_data_dir, img_name))

            elif self.torp == 'validation':
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, cfg.validation_data_dir, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, cfg.validation_data_dir, img_name), as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, cfg.validation_data_dir, style_filename))

                c_mask = io.imread(os.path.join(self.data_dir, cfg.c_mask, cfg.validation_data_dir, img_name), as_gray = True)
                c_skeleton = io.imread(os.path.join(self.data_dir, cfg.c_skeleton, cfg.validation_data_dir, img_name), as_gray = True)

                e_title = io.imread(os.path.join(self.data_dir, cfg.extracted_title, cfg.validation_data_dir, img_name))

            else:
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, img_name), as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, style_filename))

                c_mask = io.imread(os.path.join(self.data_dir, cfg.c_mask, img_name), as_gray = True)
                c_skeleton = io.imread(os.path.join(self.data_dir, cfg.c_skeleton, img_name), as_gray = True)

                e_title = io.imread(os.path.join(self.data_dir, cfg.extracted_title, img_name))


        return [input_text, style_mask, style_inpaint, c_mask, c_skeleton, e_title]
        
class Example_dataset(Dataset):
    
    def __init__(self, data_dir, torp = None, ex_n = 1, transform = None):

        self.data_dir = data_dir
        self.torp = torp
        self.ex_num = ex_n

        if self.ex_num == 1:
            if self.torp == 'test':
                self.name_list = os.listdir(os.path.join(self.data_dir, cfg.style, cfg.test_data_dir))
            else:
                self.name_list = os.listdir(os.path.join(self.data_dir, cfg.style))
        else:
            if self.torp == 'test':
                self.name_list = os.listdir(os.path.join(self.data_dir, cfg.style_mask, cfg.test_data_dir))
            elif self.torp == None:
                self.name_list = os.listdir(os.path.join(self.data_dir, cfg.style_mask))
            else:
                self.name_list = os.listdir(os.path.join(self.data_dir, cfg.style_mask, self.torp))
        
        self.transform = transform
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        if self.ex_num == 1:
            if self.torp == 'test':
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, cfg.test_data_dir, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, cfg.test_data_dir, img_name),as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, cfg.test_data_dir, img_name))

                to_scale = (cfg.data_shape[0],cfg.data_shape[0])

                input_text = resize(input_text, to_scale, preserve_range=True)
                style_mask = np.expand_dims(resize(style_mask, to_scale, preserve_range=True), axis = -1)
                style_inpaint = resize(style_inpaint, to_scale, preserve_range=True)

            else:    
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, img_name),as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, img_name))

                to_scale = (cfg.data_shape[0],cfg.data_shape[0])

                input_text = resize(input_text, to_scale, preserve_range=True)
                style_mask = np.expand_dims(resize(style_mask, to_scale, preserve_range=True), axis = -1)
                style_inpaint = resize(style_inpaint, to_scale, preserve_range=True)
        else:
            # filename = "0761525696_01.jpg"
            filename_without_ext, ext = os.path.splitext(os.path.basename(img_name))
            # filename_without_ext = "0761525696_01" ,ext = ".jpg"
            style_filename_without_ext = filename_without_ext.rsplit("_",1)[0]
            # style_filename_without_ext = "0761525696"
            style_filename = style_filename_without_ext + ext

            if self.torp == 'test':
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, cfg.test_data_dir, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, cfg.test_data_dir, img_name),as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, cfg.test_data_dir, style_filename))

                to_scale = (cfg.data_shape[0],cfg.data_shape[0])

                input_text = resize(input_text, to_scale, preserve_range=True)
                style_mask = np.expand_dims(resize(style_mask, to_scale, preserve_range=True), axis = -1)
                style_inpaint = resize(style_inpaint, to_scale, preserve_range=True)

            elif self.torp == None:    
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, img_name),as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, style_filename))

                to_scale = (cfg.data_shape[0],cfg.data_shape[0])

                input_text = resize(input_text, to_scale, preserve_range=True)
                style_mask = np.expand_dims(resize(style_mask, to_scale, preserve_range=True), axis = -1)
                style_inpaint = resize(style_inpaint, to_scale, preserve_range=True)

            else:
                input_text = io.imread(os.path.join(self.data_dir, cfg.input_text, self.torp, img_name))
                style_mask = io.imread(os.path.join(self.data_dir, cfg.style_mask, self.torp, img_name),as_gray = True)
                style_inpaint = io.imread(os.path.join(self.data_dir, cfg.style, self.torp, style_filename))

                to_scale = (cfg.data_shape[0],cfg.data_shape[0])

                input_text = resize(input_text, to_scale, preserve_range=True)
                style_mask = np.expand_dims(resize(style_mask, to_scale, preserve_range=True), axis = -1)
                style_inpaint = resize(style_inpaint, to_scale, preserve_range=True)

        # print(input_text.shape)
        sample = (input_text, style_mask, style_inpaint, img_name.split('.')[0])   

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class InputText_dataset(Dataset):
    def __init__(self, 
                i_t_data_dir = cfg.pred_i_t_dir,
                pred_style_path = cfg.pred_style_path,
                pred_style_mask_path = cfg.pred_style_mask_path,
                transform = None):

        self.i_t_data_dir = i_t_data_dir
        self.name_list = os.listdir(os.path.join(self.i_t_data_dir))
        
        self.transform = transform

        style_mask = io.imread(pred_style_mask_path,as_gray = True)
        style_inpaint = io.imread(pred_style_path)
        
        self.to_scale = (cfg.data_shape[0],cfg.data_shape[0])
        self.style_mask = np.expand_dims(resize(style_mask, self.to_scale, preserve_range=True), axis = -1)
        self.style_inpaint = resize(style_inpaint, self.to_scale, preserve_range=True)
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        img_name = self.name_list[idx]

        input_text = io.imread(os.path.join(self.i_t_data_dir, img_name))
        #io.imsave('sample.png',input_text)
     
        input_text = resize(input_text, self.to_scale, preserve_range=True)

        filename_without_ext, ext = os.path.splitext(os.path.basename(img_name))
        sample = (input_text, self.style_mask, self.style_inpaint, str(filename_without_ext))

        #io.imsave('style_mask.png',self.style_mask)
        #io.imsave('style_inpaint.png',self.style_inpaint)


        if self.transform:
            sample = self.transform(sample)
        
        return sample

class To_tensor(object):
    def __call__(self, sample):
        input_text, style_mask, style_inpaint, img_name = sample
        
        input_text = input_text.transpose((2, 0, 1)) /127.5 - 1
        style_mask = style_mask.transpose((2, 0, 1)) / 255.
        style_inpaint = style_inpaint.transpose((2, 0, 1)) /127.5 - 1

        input_text = torch.from_numpy(input_text)
        style_mask = torch.from_numpy(style_mask)
        style_inpaint = torch.from_numpy(style_inpaint)

        return (input_text.float(),style_mask.float(),style_inpaint.float(), img_name)