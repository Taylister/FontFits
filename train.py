import numpy as np
import os
import torch
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from model import Generator, Discriminator, Vgg19
from torchvision import models, transforms, datasets
from loss import build_generator_loss, build_discriminator_loss, GANLoss

from dataset import TGNet_dataset, Example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def custom_collate(batch):
    
    input_text_batch, style_mask_batch, style_inpaint_batch = [],[],[] 
    c_mask_batch, c_skeleton_batch = [],[] 
    e_title_batch = [] 

    to_scale = (cfg.data_shape[0],cfg.data_shape[0])
    
    for item in batch:
   
        input_text, style_mask, style_inpaint, c_mask, c_skeleton, e_title = item

        input_text = resize(input_text, to_scale, preserve_range=True)
        style_mask = np.expand_dims(resize(style_mask, to_scale, preserve_range=True), axis = -1)
        style_inpaint = resize(style_inpaint, to_scale, preserve_range=True)
        c_mask = np.expand_dims(resize(c_mask, to_scale, preserve_range=True), axis = -1)
        c_skeleton = np.expand_dims(resize(c_skeleton, to_scale, preserve_range=True), axis = -1)
        e_title = resize(e_title, to_scale, preserve_range=True)

        input_text = input_text.transpose((2, 0, 1))
        style_mask = style_mask.transpose((2, 0, 1))
        style_inpaint = style_inpaint.transpose((2, 0, 1))
        c_mask = c_mask.transpose((2, 0, 1))
        c_skeleton = c_skeleton.transpose((2, 0, 1))
        e_title = e_title.transpose((2, 0, 1))
    
        input_text_batch.append(input_text)
        style_mask_batch.append(style_mask)
        style_inpaint_batch.append(style_inpaint)
        c_mask_batch.append(c_mask) 
        c_skeleton_batch.append(c_skeleton) 
        e_title_batch.append(e_title)
    
    input_text_batch = np.stack(input_text_batch)
    style_mask_batch = np.stack(style_mask_batch)
    style_inpaint_batch = np.stack(style_inpaint_batch)
    c_mask_batch = np.stack(c_mask_batch)
    c_skeleton_batch = np.stack(c_skeleton_batch) 
    e_title_batch = np.stack(e_title_batch)

    input_text_batch = torch.from_numpy(input_text_batch.astype(np.float32) / 127.5 - 1.) 
    style_mask_batch = torch.from_numpy(style_mask_batch.astype(np.float32) / 255.)
    style_inpaint_batch = torch.from_numpy(style_inpaint_batch.astype(np.float32) / 127.5 - 1.)
    c_mask_batch = torch.from_numpy(c_mask_batch.astype(np.float32) / 255.) 
    c_skeleton_batch = torch.from_numpy(c_skeleton_batch.astype(np.float32) / 255.) 
    e_title_batch = torch.from_numpy(e_title_batch.astype(np.float32) / 127.5 - 1.) 
      
    return [input_text_batch, style_mask_batch, style_inpaint_batch, c_mask_batch, c_skeleton_batch, e_title_batch]

def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def main():
    # ================================================
    # Preparation
    # ================================================
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    
    gpu = torch.device('cuda:0')

    train_name = get_train_name()
 
    print('===> Loading datasets')
    train_data = TGNet_dataset(cfg.data_dir, torp='train', ex_n = 2)
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)
    
    trfms = To_tensor()
    example_data = Example_dataset(cfg.example_data_dir, torp='test', ex_n = 2,transform = trfms)    
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)

        
    G = Generator(in_channels_title = 3, in_channels_style = 4).to(gpu)    
    D = Discriminator(in_channels_title = 6,img_shape = (cfg.data_shape[0],cfg.data_shape[0])).to(gpu)
    perception_features = Vgg19().to(gpu)   
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    criterionGAN = GANLoss().to(gpu)


    try:
        checkpoint = torch.load(cfg.train_ckpt_path)
        G.load_state_dict(checkpoint['generator'])
        D.load_state_dict(checkpoint['discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print('Resuming after loading...')

    except FileNotFoundError:
        print('checkpoint not found')
      
        pass  

    trainiter = iter(train_data)
    example_iter = iter(example_loader)
    
    K = torch.nn.ZeroPad2d((0, 1, 1, 0))
    
    writer = SummaryWriter(os.path.join(cfg.train_result_dir, train_name ,cfg.checkpoint_savedir))

    for step in tqdm(range(cfg.ex_max_iter)):
        if ((step + 1) % cfg.save_ckpt_interval == 0 or step == 0):
            save_dir = os.path.join(cfg.train_result_dir, train_name ,cfg.checkpoint_savedir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(
                {
                    'generator': G.state_dict(),
                    'discriminator': D.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                save_dir + '/' + f'train_step-{step+1}.model',
            )
        
        try:
          i_t, style_mask, style_cover, true_mask, true_sk, true_title = trainiter.next()

        except StopIteration:

          trainiter = iter(train_data)
          i_t, style_mask, style_cover, true_mask, true_sk, true_title = trainiter.next()

        G.train()
        
        #########################
        # Training (D)
        #########################
        d_optimizer.zero_grad()
        requires_grad(G, False)
        requires_grad(D, True)

        # Input of Generator
        i_t = i_t.to(gpu)
        style = torch.cat((style_cover,style_mask), dim=1)
        style = style.to(gpu)

        # Ground Truth
        true_sk = true_sk.to(gpu)
        true_mask = true_mask.to(gpu)
        true_title = true_title.to(gpu)
        
        labels = [true_sk, true_title, true_mask]
        
        _, pred_title = G(i_t, style)) #Adding dim info
    
        pred_title = K(pred_title)
                
        # input_discriminator
        i_d_title_true = torch.cat((true_title, i_t) ,dim=1)
        i_d_title_pred = torch.cat((pred_title, i_t) ,dim=1) 

        o_d_true = D(i_d_title_true)
        o_d_fake = D(i_d_title_pred)

        d_loss_true = criterionGAN(o_d_true,True)
        d_loss_fake = criterionGAN(o_d_fake,False)
        
        d_loss = build_discriminator_loss(d_loss_true, d_loss_fake)

        d_loss.backward()
        d_optimizer.step()

        clip_grad(D)
        
        #########################
        # Training (G)
        #########################
        g_optimizer.zero_grad()
        requires_grad(G, True)
        requires_grad(D, False)
        
        pred_sk, pred_title = G(i_t, style)
        
        pred_title = K(pred_title)
        pred_sk = K(pred_sk)
        out_g = [pred_sk, pred_title]

        i_d_title_pred = torch.cat((pred_title, i_t), dim=1) 
        o_d_fake = D(i_d_title_pred)
        g_loss_adv = criterionGAN(o_d_fake,True)
        
        i_vgg = torch.cat((true_title, pred_title), dim=0)
        out_vgg = perception_features(i_vgg)
        
        labels = [true_sk, true_title, true_mask]

        g_loss = build_generator_loss(out_g, g_loss_adv, out_vgg, labels) 

        g_loss.backward()
        g_optimizer.step()

  
        if ((step + 1) % cfg.write_log_interval == 0 or step == 0):
            print('Iter: {}/{} | Gen: {} | Dis{}'.format(step+1, cfg.ex2_max_iter, g_loss.item(), d_loss.item()))
            writer.add_scalar('G_loss',g_loss.item(), step + 1)
            writer.add_scalar('D_loss',d_loss.item(), step + 1)
            
        if ((step + 1) % cfg.gen_example_interval == 0 or step == 0):
            
            savedir = os.path.join(cfg.train_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))
            
            with torch.no_grad():
                G.eval()
                example_iter = iter(example_loader)
                for ex_iter, batch in enumerate(example_iter):

                    i_t = batch[0].to(gpu)
                    i_s_mask = batch[1].to(gpu)
                    i_s_inpaint = batch[2].to(gpu)

                    style = torch.cat((i_s_inpaint, i_s_mask), dim=1)

                    name = str(batch[3][0])                

                    o_sk, o_t = G(i_t, style, (i_t.shape[2], i_t.shape[3]))

                    o_sk = o_sk.squeeze(0).to('cpu')
                    o_t = o_t.squeeze(0).to('cpu')

                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    o_sk = F.to_pil_image(o_sk)
                    o_t = F.to_pil_image((o_t + 1)/2)

                    o_sk.save(os.path.join(savedir, name + '_sk.png'))
                    o_t.save(os.path.join(savedir, name + '_t.png'))
             
if __name__ == '__main__':
    main()
