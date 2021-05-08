import numpy as np
import os
import torch
import torchvision.transforms
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from model import Generator
from utils import get_train_name
from tqdm import tqdm
from torchvision import models, transforms, datasets
from dataset import InputText_dataset, Example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader

def main():
    
    eval_name = get_train_name()

    print('===> Loading datasets')
 
    trfms = To_tensor()
    torp = 'test'
    if cfg.pred_mode == 'multiple':
        example_data = InputText_dataset(transform = trfms)
    else:
        example_data = Example_dataset(cfg.example_data_dir, torp=torp, ex_n = 2,transform = trfms)    
    
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)


    print('===> Loading models')

    if cfg.pred_gpu :
        gpu = torch.device('cuda:0')
        G = Generator(in_channels_title = 3, in_channels_style = 4).to(gpu)
        try:
            checkpoint = torch.load(cfg.model_weight)
            G.load_state_dict(checkpoint['generator'])
            print('===> loading Generator weight')

        except FileNotFoundError:
            print('===> checkpoint not found')
            pass  
       
    else:
        G = Generator(in_channels_title = 3, in_channels_style = 4)
        try:
            checkpoint = torch.load(cfg.model_weight, map_location = torch.device('cpu'))
            G.load_state_dict(checkpoint['generator'])
            print('===> loading Generator weight')

        except FileNotFoundError:
            print('===> checkpoint not found')
            pass  
    
        
    print('===> Creating directory to save output')

    savedir_sk = os.path.join(cfg.pred_result_dir, eval_name, "o_sk")
    savedir_t = os.path.join(cfg.pred_result_dir, eval_name, "o_t")
    
    if not os.path.exists(savedir_sk):
        os.makedirs(savedir_sk)

    if not os.path.exists(savedir_t):
        os.makedirs(savedir_t)
            
    example_iter = iter(example_loader)
    G.eval()
    torch.set_grad_enabled(False)

    pbar = tqdm(len(example_iter),total=len(example_iter))
    pbar.set_description("Generating Title part")
    
    for ex_iter, batch in enumerate(example_iter):
        if cfg.pred_gpu:
            i_t = batch[0].to(gpu)
            i_s_mask = batch[1].to(gpu)
            i_s_inpaint = batch[2].to(gpu)
            style = torch.cat((i_s_inpaint, i_s_mask), dim=1)
            name = str(batch[3][0])                
            o_sk, o_t = G(i_t, style, (i_t.shape[2], i_t.shape[3]))
            o_sk = o_sk.squeeze(0).to('cpu')
            o_t = o_t.squeeze(0).to('cpu')
        else:
            i_t = batch[0]
            i_s_mask = batch[1]
            i_s_inpaint = batch[2]
            style = torch.cat((i_s_inpaint, i_s_mask), dim=1)
            name = str(batch[3][0])                
            o_sk, o_t = G(i_t, style, (i_t.shape[2], i_t.shape[3]))
            o_sk = o_sk.squeeze(0)
            o_t = o_t.squeeze(0)

        o_sk = F.to_pil_image(o_sk)
        o_t = F.to_pil_image((o_t + 1)/2)
        o_sk.save(os.path.join(savedir_sk, name + '_sk.png'))
        o_t.save(os.path.join(savedir_t, name + '_t.png'))

        if(torp == 'train' and  1000 < ex_iter ):
            break

        pbar.update(1)
    
    pbar.close()

if __name__ == "__main__":
    main()