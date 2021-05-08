import torch
import tensorflow as tf
import cfg
import torch.nn as nn
from torch.nn import BCELoss, DataParallel, BCEWithLogitsLoss

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

def build_discriminator_loss(true_loss, fake_loss):
    return (true_loss + fake_loss)*0.5

def build_dice_loss(x_t, x_o):
       
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat*tflat).sum()
    
    return 1. - torch.mean((2. * intersection + cfg.epsilon)/(iflat.sum() +tflat.sum()+ cfg.epsilon))

def build_l1_loss(x_t, x_o):
        
    return torch.mean(torch.abs(x_t - x_o))

def build_l1_loss_with_mask(x_t, x_o, mask):
    mask_ratio = 1. - mask.view(-1).sum() / mask.view(-1).shape[0]
    l1 = torch.abs(x_t - x_o)
    return mask_ratio * torch.mean(l1 * mask) + (1. - mask_ratio) * torch.mean(l1 * (1. - mask))

def build_perceptual_loss(x):        
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_gram_matrix(x):

    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram

def build_style_loss(x):
        
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] *f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_vgg_loss(x):
        
    splited = []

    for i, f in enumerate(x):
        splited.append(torch.chunk(f, 2))
        #print(f.shape)
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style

def build_generator_loss(out_g, adv_loss, out_vgg, labels):
        
    pred_sk, pred_title = out_g
    true_sk, true_title, true_mask = labels
    
    # gan loss
    l_adv = adv_loss

    #skeleton loss
    l_sk = build_dice_loss(true_sk, pred_sk)
    
    #l1 loss
    l_l1 = build_l1_loss_with_mask(true_title, pred_title, true_mask)

    # vgg loss
    l_vgg_per, l_vgg_style = build_vgg_loss(out_vgg)
    l_vgg_per = cfg.theta_per * l_vgg_per
    l_vgg_style = cfg.theta_style * l_vgg_style
    l_vgg = l_vgg_per + l_vgg_style
  
    l =  l_adv * cfg.theta_gan + l_l1 + l_sk + l_vgg 

    return l
