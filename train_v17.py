import argparse
import os
import random
import yaml
import time
import logging
import pprint

import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import grad
from torch.cuda.amp import GradScaler, autocast
from easydict import EasyDict

from data.train import CreateDataLoader as train_loader
from data.eval import CreateDataLoader as val_loader
from utils import create_logger, save_checkpoint, load_state, get_scheduler, AverageMeter, calculate_fid, calculate_lpips
from models.standard_final_256 import *

parser = argparse.ArgumentParser(description='PyTorch Colorization Training')

parser.add_argument('--config', default='experiments_v17_256/origin/config.yaml')
# parser.add_argument('--resume', default='experiments_v15/origin/ckpt.pth.tar', type=str, help='path to checkpoint')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')


def mask_gen():
    maskS = config.image_size // 4

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS, device=config.device).ge(X.rvs(1)[0]).float() for _ in range(config.batch_size // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS, device=config.device).float() for _ in range(config.batch_size - config.batch_size // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    return mask

def get_z_random(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, nz, device=config.device) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz, device=config.device)
    return z.detach()

def get_z_random_same_in_batch(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(1, nz, device=config.device) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(1, nz, device=config.device)

    z = torch.cat([z for _ in range(batch_size)], 0)

    return z.detach()

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = criterionBCE(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = config.batch_size
    grad_dout = torch.autograd.grad(
        outputs=scalerD.scale(d_out.sum()), inputs= x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    with autocast(enabled=ampD):
        inv_scale = 1./scalerD.get_scale()
        grad_dout = grad_dout * inv_scale 
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def main():
    global args, config, X, ampD, criterionBCE, scalerD
    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))

    config.save_path = os.path.dirname(args.config)
    gpu_id = 0


    ####### regular set up
    assert torch.cuda.is_available()
    device = torch.device("cuda:"+str(gpu_id))
    config.device = device

    # random seed setup
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    ####### regular set up end


    netG = torch.nn.DataParallel(NetG(ngf=config.ngf, nz = config.nz), device_ids =[gpu_id])
    netD = torch.nn.DataParallel(NetD(ndf=config.ndf), device_ids =[gpu_id])

    netF = torch.nn.DataParallel(NetF(),  device_ids =[gpu_id])
    netT = torch.nn.DataParallel(CST(),  device_ids =[gpu_id])

    def map_func(storage, location):
        return storage.cuda()
    path = 'CST.pth.tar'
    checkpoint = torch.load(path, map_location=map_func)
    netT.load_state_dict(checkpoint['state_CST'], strict=False)

    for param in netT.parameters():
        param.requires_grad = False
    for param in netF.parameters():
        param.requires_grad = False

    criterion_MSE = nn.MSELoss()
    criterionL1 = torch.nn.L1Loss()
    criterionBCE = torch.nn.BCEWithLogitsLoss()

    fixed_sketch = torch.tensor(0, device=device).float()
    fixed_skeleton = torch.tensor(0, device=device).float()
    # fixed_region = torch.tensor(0, device=device).float()
    fixed_hint = torch.tensor(0, device=device).float()
    # fixed_sketch_feat = torch.tensor(0, device=device).float()
    fixed_z1 = get_z_random_same_in_batch(config.batch_size, config.nz)
    fixed_z2 = get_z_random_same_in_batch(config.batch_size, config.nz)
    fixed_z3 = get_z_random_same_in_batch(config.batch_size, config.nz)
    ####################
    netD = netD.to(device)
    netG = netG.to(device)
    netF = netF.to(device)
    netT = netT.to(device)
    # netI = netI.to(device)
    criterion_MSE = criterion_MSE.to(device)
    criterionL1 = criterionL1.to(device)
    criterionBCE = criterionBCE.to(device)

    # setup optimizer
    ampG=True
    ampD=True
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_schedulerG.base_lr, betas=(0, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr_schedulerD.base_lr, betas=(0, 0.99))

    scalerG = torch.cuda.amp.GradScaler(enabled=ampG)
    scalerD = torch.cuda.amp.GradScaler(enabled=ampD)

    last_iter = -1
    best_fid = 1e6
    lpips = 0

    if args.resume:
        best_fid, last_iter = load_state(args.resume, netG, netD, optimizerG, optimizerD, scalerG, scalerD)


    config.lr_schedulerG['last_iter'] = last_iter
    config.lr_schedulerD['last_iter'] = last_iter
    config.lr_scheduler['last_iter'] = last_iter

    config.lr_schedulerG['optimizer'] = optimizerG
    lr_schedulerG = get_scheduler(config.lr_schedulerG)
    config.lr_schedulerD['optimizer'] = optimizerD
    lr_schedulerD = get_scheduler(config.lr_schedulerD)

    tb_logger = SummaryWriter(config.save_path + '/events')
    logger = create_logger('global_logger', config.save_path + '/log.txt')
    logger.info(f'args: {pprint.pformat(args)}')
    logger.info(f'config: {pprint.pformat(config)}')

    batch_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    flag = 1
    mu, sigma = 1, 0.012
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
    i = 0
    curr_iter = last_iter + 1

    dataloader = train_loader(config)
    data_iter = iter(dataloader)

    end = time.time()
    while i < len(dataloader):
        lr_schedulerG.step(curr_iter)
        lr_schedulerD.step(curr_iter)
        current_lr = lr_schedulerG.get_lr()[0]
        ############################
        # (1) Update D network
        ###########################
        data_end = time.time()
        colour, colour_down, sketch, skeleton = data_iter.next()
        data_time.update(time.time() - data_end)

        i += 1
        colour, colour_down, sketch, skeleton = colour.to(device), colour_down.to(device), sketch.to(device), skeleton.to(device)

        with autocast(enabled=ampD):
            colour.requires_grad_()

            mask = mask_gen()
            hint = torch.cat((colour_down * mask, mask), 1)

            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in netG.parameters():
                p.requires_grad = False  # to avoid computation ft_params

            netD.zero_grad()
                
            z = get_z_random(config.batch_size, config.nz)
            with torch.no_grad():
                half = config.batch_size //2
                D_fake_colour = netG(sketch, hint, z, sketch_feat=None, has_hint=True)

            D_fake = netD(D_fake_colour)
            D_real = netD(colour)

            loss_real = adv_loss(D_real, 1)
            loss_fake = adv_loss(D_fake, 0)

        loss_reg = r1_reg(D_real, colour)
        #Optimise
        combine_lossD = loss_real + loss_fake + loss_reg

        scalerD.scale(combine_lossD).backward()
        scalerD.step(optimizerD)
        scalerD.update()
        ############################
        # (2) Update G network
        ############################

        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True
        netG.zero_grad()

        colour, colour_down, sketch, skeleton = data_iter.next()
        colour, colour_down, sketch, skeleton = colour.to(device), colour_down.to(device), sketch.to(device), skeleton.to(device)
        i += 1

        if flag:  # fix samples
            mask = mask_gen()
            hint = torch.cat((colour_down * mask, mask), 1)

            tb_logger.add_image('target imgs', vutils.make_grid(colour.mul(0.5).add(0.5), nrow=4))
            tb_logger.add_image('sketch imgs', vutils.make_grid(sketch.mul(0.5).add(0.5), nrow=4))
            tb_logger.add_image('skeleton imgs', vutils.make_grid(skeleton.mul(0.5).add(0.5), nrow=4))
            tb_logger.add_image('hint', vutils.make_grid((colour_down * mask).mul(0.5).add(0.5), nrow=4))

            fixed_sketch.resize_as_(sketch).copy_(sketch)
            fixed_skeleton.resize_as_(skeleton).copy_(skeleton)
            fixed_hint.resize_as_(hint).copy_(hint)

            flag -= 1


        freeze = 0
        if curr_iter > freeze:

            with torch.cuda.amp.autocast(enabled=ampG):
                mask = mask_gen()
                hint = torch.cat((colour_down * mask, mask), 1)
                z1 = get_z_random(config.batch_size, config.nz)
                half = config.batch_size // 2
                G_fake_colour, fake_skeleton, output_z1, decoder_output = netG(sketch, hint, z1, skeleton_output=True, has_hint=True, output_z = True, guide_decoder_output=True)

                G_adv_loss= netD(G_fake_colour)
                G_loss_adv = adv_loss(G_adv_loss, 1)
        
                #content_loss + shading loss
                feature_fake1 = netF(G_fake_colour)
                feature_fakedo = netF(decoder_output)
                G_fake_colours_v = netT(G_fake_colour)[:,2:,:,:]
                with torch.no_grad():
                    feature_real = netF(colour)
                    real_colour_v = netT(colour)[:,2:,:,:]


                content_loss = criterion_MSE(feature_fake1, feature_real)
                shading_loss = criterionL1(G_fake_colours_v, real_colour_v) * 10
                guide_decoder_loss = criterion_MSE(feature_fakedo, feature_real) * 0.9

                # rec_loss_outputz=0
                no_hint = torch.cat([torch.zeros(1, 4, config.image_size//4, config.image_size//4, device=config.device).float() for _ in range(config.batch_size)], 0)

                with torch.no_grad():
                    G_fake_colours_hint_useoutputz = netG(sketch, no_hint, output_z1)
                    feature_fake_useoutz = netF(G_fake_colours_hint_useoutputz)
                rec_loss_outputz = torch.mean(torch.abs(feature_fake1 - feature_fake_useoutz)) * 0.7


                # 3, modes seeking loss
                z2 = get_z_random(config.batch_size, config.nz)
                G_fake_colour2, output_z2 = netG(sketch, hint, z2, has_hint=True, output_z = True)
                G_fake_colour2_v = netT(G_fake_colour2)[:,2:,:,:]

                feature_fake2 = netF(G_fake_colour2)
                VGGz_loss = (criterionL1(feature_fake1, feature_fake2) + criterionL1(G_fake_colours_v, G_fake_colour2_v)) 
                # VGGz_loss = (criterion_MSE(feature_fake1, feature_fake2) + criterionL1(G_fake_colours_v, G_fake_colour2_v) * 10) * 0.5
                lz = VGGz_loss / (torch.mean(torch.abs(output_z1.clone().detach() - output_z2.clone().detach())))
                eps = 1 * 1e-5            # prevent divide by 0
                loss_lz = (1 / (lz + eps) ) 

                #4 L1 skeleton loss
                skeleton_loss = criterionL1(fake_skeleton, skeleton)  * 0.9

                #5
                distri_loss = criterionL1(torch.mean(torch.abs(output_z1 - output_z2)),(torch.mean(torch.abs(z1 - z2)))) * 0.9

                combine_lossG = G_loss_adv + content_loss + loss_lz + skeleton_loss  + rec_loss_outputz + guide_decoder_loss + distri_loss


            scalerG.scale(combine_lossG).backward()
            scalerG.step(optimizerG)
            scalerG.update()

        batch_time.update(time.time() - end)

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        curr_iter += 1

        # if curr_iter % 500 == 0:
            # print(scalerD.get_scale())
            # print(scalerG.get_scale())
            # print()

        if curr_iter > freeze and curr_iter % config.print_freq == 0 and curr_iter > 0:
            tb_logger.add_scalar('VGG MSE Loss', content_loss.item(), curr_iter)
            tb_logger.add_scalar('lz Loss', loss_lz.item(), curr_iter)
            # tb_logger.add_scalar('lz2 Loss', loss_lz2_2.item(), curr_iter)
            # tb_logger.add_scalar('wasserstein distance_enc', errD.item(), curr_iter)
            tb_logger.add_scalar('errD_real', loss_real.item(), curr_iter)
            tb_logger.add_scalar('errD_fake', loss_fake.item(), curr_iter)
            tb_logger.add_scalar('errG_ske', skeleton_loss.item(), curr_iter)
            tb_logger.add_scalar('shading_loss', shading_loss.item(), curr_iter)
            
            tb_logger.add_scalar('Gnet loss toward real', G_loss_adv.item(), curr_iter)
            tb_logger.add_scalar('r1', loss_reg.item(), curr_iter)
            tb_logger.add_scalar('lr', current_lr, curr_iter)

            tb_logger.add_scalar('rec_z', rec_loss_outputz.item(), curr_iter)
            tb_logger.add_scalar('guide_decoder', guide_decoder_loss.item(), curr_iter)
            tb_logger.add_scalar('distri_loss', distri_loss.item(), curr_iter)


            logger.info(f'Iter: [{curr_iter}/{len(dataloader)//(config.diters+1)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'errG {G_loss_adv.item():.4f}\t'
                        f'errD {loss_real.item():.4f}\t'
                        # f'err_D_real {errD_real.item():.4f}\t'
                        # f'err_D_fake {errD_fake.item():.4f}\t'
                        f'VGG loss {content_loss.item():.4f}\t'
                        f'LR {current_lr:.6f}'
                        )

        if curr_iter > freeze and  curr_iter % config.print_img_freq == 0 and curr_iter > 0:
            with torch.no_grad():

                tb_logger.add_image('current batch_full',
                    vutils.make_grid(G_fake_colour.detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                    curr_iter)
                tb_logger.add_image('current batch_full2',
                    vutils.make_grid(G_fake_colour2.detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                    curr_iter)
                tb_logger.add_image('current batch_rec',
                    vutils.make_grid(G_fake_colours_hint_useoutputz.detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                    curr_iter)

                with autocast():
                    fake1 = netG(fixed_sketch, fixed_hint, fixed_z1, skeleton_output=False,  has_hint=True)
                    fake2 = netG(fixed_sketch, fixed_hint, fixed_z2, skeleton_output=False,  has_hint=True  )
                    fake3 = netG(fixed_sketch, fixed_hint, fixed_z3, skeleton_output=False,  has_hint=True  )

                    
                    # fake1_has = netG(fixed_sketch[:half], fixed_hint[:half], fixed_z1[:half], has_hint=True)
                    # fake1_no = netG(fixed_sketch[half:], fixed_hint[half:], fixed_z1[half:])
                    # fake1 = torch.cat([fake1_has, fake1_no],0)

                    # fake2_has = netG(fixed_sketch[:half], fixed_hint[:half], fixed_z2[:half], has_hint=True)
                    # fake2_no = netG(fixed_sketch[half:], fixed_hint[half:], fixed_z2[half:])
                    # fake2 = torch.cat([fake2_has, fake2_no],0)

                    # fake3_has = netG(fixed_sketch[:half], fixed_hint[:half], fixed_z3[:half], has_hint=True)
                    # fake3_no = netG(fixed_sketch[half:], fixed_hint[half:], fixed_z3[half:])
                    # fake3 = torch.cat([fake3_has, fake3_no],0)


                
                tb_logger.add_image('colored imgs (fixed z)',
                                    vutils.make_grid(torch.cat([fake1,fake2,fake3],0).detach().mul(0.5).add(0.5), nrow=config.batch_size),
                                    curr_iter)


                rand_z1 = get_z_random(config.batch_size, config.nz)
                rand_z2 = get_z_random(config.batch_size, config.nz)
                rand_z3 = get_z_random(config.batch_size, config.nz)

                with autocast():
                    fake1 = netG(fixed_sketch, fixed_hint, rand_z1, skeleton_output=False,  has_hint=True  )
                    fake2 = netG(fixed_sketch, fixed_hint, rand_z2, skeleton_output=False,  has_hint=True  )
                    fake3 = netG(fixed_sketch, fixed_hint, rand_z3, skeleton_output=False,  has_hint=True  )
                    # fake1_has = netG(fixed_sketch[:half], fixed_hint[:half], rand_z1[:half], has_hint=True)
                    # fake1_no = netG(fixed_sketch[half:], fixed_hint[half:], rand_z1[half:])
                    # fake1 = torch.cat([fake1_has, fake1_no],0)

                    # fake2_has = netG(fixed_sketch[:half], fixed_hint[:half], rand_z2[:half], has_hint=True)
                    # fake2_no = netG(fixed_sketch[half:], fixed_hint[half:], rand_z2[half:])
                    # fake2 = torch.cat([fake2_has, fake2_no],0)

                    # fake3_has = netG(fixed_sketch[:half], fixed_hint[:half], rand_z3[:half], has_hint=True)
                    # fake3_no = netG(fixed_sketch[half:], fixed_hint[half:], rand_z3[half:])
                    # fake3 = torch.cat([fake3_has, fake3_no],0)



                tb_logger.add_image('colored imgs (rand z)',
                    vutils.make_grid(torch.cat([fake1,fake2,fake3],0).detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                    curr_iter)

                # hint = mask_gen()
                hint = torch.cat([torch.zeros(1, 4, config.image_size//4, config.image_size//4, device=config.device).float() for _ in range(config.batch_size)], 0)

                with autocast():
                    fake,o_z = netG(fixed_sketch, hint, fixed_z1, has_hint=True, output_z=True)
                    fake1,o_z1 = netG(fixed_sketch, hint, fixed_z2, has_hint=True, output_z=True)
                    fake2,o_z2 = netG(fixed_sketch, hint, fixed_z3, has_hint=True, output_z=True)
                    fakes = torch.cat([fake,fake1,fake2], 0)
                tb_logger.add_image('colored imgs (fixed z), no hint, has hint False',
                    vutils.make_grid(fakes.detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                    curr_iter)

                with autocast():

                    fake = netG(fixed_sketch, hint, o_z, )
                    fake1 = netG(fixed_sketch, hint, o_z1, )
                    fake2 = netG(fixed_sketch, hint, o_z2, )
                    fakes = torch.cat([fake,fake1,fake2], 0)
                tb_logger.add_image('colored imgs (fixed z), no hint rec',
                    vutils.make_grid(fakes.detach().mul(0.5).add(0.5),  nrow=config.batch_size),
                    curr_iter)


                target1 = fixed_z1[0:1]
                target2 = fixed_z2[0:1]
                scale = [0.125,0.25,0.375,0.5,0.625,0.75,0.875]
                z121 = torch.lerp(target1,target2, scale[0])
                z122 = torch.lerp(target1,target2, scale[1])
                z123 = torch.lerp(target1,target2, scale[2])
                z124 = torch.lerp(target1,target2, scale[3])
                z125 = torch.lerp(target1,target2, scale[4])
                z126 = torch.lerp(target1,target2, scale[5])
                z127 = torch.lerp(target1,target2, scale[6])

                with autocast():
                    fake = netG(sketch[0:1].repeat(len(scale)+2,1,1,1), hint[0:1].repeat(len(scale)+2,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0), sketch_feat=None, skeleton_output=False, has_hint=True)
                    fake2 = netG(sketch[-1:].repeat(len(scale)+2,1,1,1), hint[-1:].repeat(len(scale)+2,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0), sketch_feat=None, skeleton_output=False, has_hint=True)
                tb_logger.add_image('colored imgs interpolation',
                        vutils.make_grid(torch.cat([fake, fake2],0).detach().mul(0.5).add(0.5),  nrow=len(scale)+2),
                        curr_iter)

                with autocast():
                    fake = netG(sketch[0:1].repeat(len(scale)+2,1,1,1), hint[0:1].repeat(len(scale)+2,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0), sketch_feat=None, skeleton_output=False)
                    fake2 = netG(sketch[-1:].repeat(len(scale)+2,1,1,1), hint[-1:].repeat(len(scale)+2,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0), sketch_feat=None, skeleton_output=False)
                tb_logger.add_image('colored imgs interpolation (no has hint)',
                        vutils.make_grid(torch.cat([fake, fake2],0).detach().mul(0.5).add(0.5),  nrow=len(scale)+2),
                        curr_iter)

        if curr_iter % config.val_freq == 0 or curr_iter % config.lr_schedulerG.lr_steps[0] == 0:
            save_checkpoint({
                    'step': curr_iter - 1,
                    'state_dictG': netG.state_dict(),
                    'state_dictD': netD.state_dict(),
                    'best_fid': best_fid,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'scalerG': scalerG.state_dict(),
                    'scalerD': scalerD.state_dict(),


            }, False, config.save_path + '/ckpt')

            fid, var, lpips = validate(netG)
            tb_logger.add_scalar('fid_val', fid, curr_iter)
            tb_logger.add_scalar('fid_variance', var, curr_iter)
            tb_logger.add_scalar('LPIPS', lpips, curr_iter)
            logger.info(f'fid: {fid:.3f} ({var})\t')
            logger.info(f'LPIPS: {lpips:.3f} \t')

            # remember best fid and save checkpoint
            is_best = fid < best_fid
            best_fid = min(fid, best_fid)

            if(curr_iter == config.lr_schedulerG.lr_steps[0]):
                save_checkpoint({
                    'step': curr_iter - 1,
                    'state_dictG': netG.state_dict(),
                    'state_dictD': netD.state_dict(),
                    'best_fid': best_fid,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'scalerG': scalerG.state_dict(),
                    'scalerD': scalerD.state_dict(),

                }, False, config.save_path + '/ckpt_milestone')
            if(is_best):
                save_checkpoint({
                    'step': curr_iter - 1,
                    'state_dictG': netG.state_dict(),
                    'state_dictD': netD.state_dict(),
                    'best_fid': best_fid,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'scalerG': scalerG.state_dict(),
                    'scalerD': scalerD.state_dict(),

                }, is_best, config.save_path + '/ckpt')


        end = time.time()


def validate(netG):
    fids = []
    fid_value = 0

    lpips = calculate_lpips(netG, val_loader(config), config, 2048, use_z=True, use_msg=False, input_all=0)
    print('LPIPS: ', lpips)
    
    for _ in range(1):
        fid = calculate_fid(netG, val_loader(config), config, 2048, use_z=True, use_msg=False, input_all=0)
        print('FID: ', fid)
        fid_value += fid
        fids.append(fid)



    torch.cuda.empty_cache()
    return fid_value, np.var(fids), lpips

if __name__ == '__main__':
    main()
