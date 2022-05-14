import argparse
import os
import random
from re import I
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
from easydict import EasyDict


from data.eval import CreateDataLoader as val_loader
from utils import create_logger, save_checkpoint, load_state, get_scheduler, AverageMeter, calculate_fid, calculate_lpips

from models.standard_final_256 import *

parser = argparse.ArgumentParser(description='PyTorch Colorization Training')

parser.add_argument('--config', default='experiments_v17_256/origin/config.yaml')
parser.add_argument('--resume', default='experiments_v17_256/origin/ckpt_milestone.pth.tar', type=str, help='path to checkpoint')

def validate(netG):
    fids = []
    fid_value = 0

    lpips = calculate_lpips(netG, val_loader(config), config, 2048, use_z=True, use_msg=False, input_all=0, has_hint=True)
    print('LPIPS: ', lpips)
    
    for _ in range(1):
        fid = calculate_fid(netG, val_loader(config), config, 2048, use_z=True, use_msg=False, input_all=0, has_hint=True)
        print('FID: ', fid)
        fid_value += fid
        fids.append(fid)

    torch.cuda.empty_cache()
    return fid_value, np.var(fids), lpips

def mask_gen(nohint=True):
    maskS = config.image_size // 4

    if (nohint):
        mask = torch.zeros(config.batch_size, 4, maskS, maskS, device=config.device).float()
    else:
        mask = torch.cat( [torch.rand(1, 1, maskS, maskS, device=config.device).ge(X.rvs(1)[0]).float() for _ in range(config.batch_size)], 0)
    return mask

def get_z_random(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz)
    return z.detach().to(config.device)

def get_z_random_same_in_batch(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(1, nz, device=config.device) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(1, nz, device=config.device)

    z = torch.cat([z for _ in range(batch_size)], 0)

    return z.detach()
def main():
    global args, config, X


    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))

    config.save_path = os.path.dirname(args.config)

    config.batch_size = 4
    config.workers = 4


    ####### regular set up
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    config.device = device
    

    # random seed setup
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    ####### regular set up end
    # netG = torch.nn.DataParallel(NetG(ngf=config.ngf))
    # netD = torch.nn.DataParallel(NetD(ndf=config.ndf))
    netG = torch.nn.DataParallel(NetG(ngf=config.ngf, nz = config.nz))
    netD = torch.nn.DataParallel(NetD(ndf=config.ndf))
    # netT = torch.nn.DataParallel(CST())

    ####################
    netD = netD.to(device)
    netG = netG.to(device)
    # netT = netT.to(device)

    # setup optimizer

    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_schedulerG.base_lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr_scheduler.base_lr, betas=(0.5, 0.9))

    if args.resume:
        _ = load_state(args.resume, netG, netD, optimizerG, optimizerD)

    tb_logger = SummaryWriter(config.save_path + '/events/eval')
    logger = create_logger('global_logger', config.save_path + '/log.txt')
    logger.info(f'args: {pprint.pformat(args)}')
    logger.info(f'config: {pprint.pformat(config)}')

    i = 0
    # curr_iter = last_iter + 1

    print('start load data')

    dataloader = val_loader(config)
    data_iter = iter(dataloader)

    print('finished_load data')


    
    # z1 = get_z_random(config.batch_size, config.nz)
    # z2 = get_z_random(config.batch_size, config.nz)
    # z3 = get_z_random(config.batch_size, config.nz)
    # z4 = get_z_random(config.batch_size, config.nz)
    # z5 = get_z_random(config.batch_size, config.nz)
    # z6 = get_z_random(config.batch_size, config.nz)
    # z7 = get_z_random(config.batch_size, config.nz)
    # z8 = get_z_random(config.batch_size, config.nz)
    # z9 = get_z_random(config.batch_size, config.nz)


        
    z1_r = get_z_random(config.batch_size, config.nz)
    z2_r = get_z_random(config.batch_size, config.nz)
    z3_r = get_z_random(config.batch_size, config.nz)
    z4_r = get_z_random(config.batch_size, config.nz)
    z5_r = get_z_random(config.batch_size, config.nz)
    z6_r = get_z_random(config.batch_size, config.nz)
    z7_r = get_z_random(config.batch_size, config.nz)
    z8_r = get_z_random(config.batch_size, config.nz)
    z9_r = get_z_random(config.batch_size, config.nz)

    z1 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z2 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z3 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z4 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z5 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z6 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z7 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z8 = get_z_random_same_in_batch(config.batch_size, config.nz)
    z9 = get_z_random_same_in_batch(config.batch_size, config.nz)



    netG.eval()

    fid,_,lpips = validate(netG)
    tb_logger.add_scalar('fid_val_100000', fid, 100000)
    tb_logger.add_scalar('LPIPS_100000', lpips, 100000)
    logger.info(f'fid: {fid:.3f}')
    logger.info(f'LPIPS: {lpips:.3f} \t')
    # while i < 50:
    #     i += 1

    #     with torch.no_grad():
    #         start = time.time()

    #         data = data_iter.next()
    #         sketch = data
    #         sketch = sketch.to(device)
    #         hint = mask_gen()

    #         print(i)



    #         fake = netG(torch.cat([sketch,sketch,sketch],0), torch.cat([hint,hint,hint],0), torch.cat([z1,z2,z3],0), sketch_feat=None, skeleton_output=False, has_hint=True)


    #         fake2 = netG(torch.cat([sketch,sketch,sketch],0),  torch.cat([hint,hint,hint],0), torch.cat([z4,z5,z6],0), sketch_feat=None, skeleton_output=False, has_hint=True)
        
    #         fake3 = netG(torch.cat([sketch,sketch,sketch],0),  torch.cat([hint,hint,hint],0), torch.cat([z7,z8,z9],0), sketch_feat=None, skeleton_output=False, has_hint=True)


    #         tb_logger.add_image('eval imgs_nohint_samez_inbatch',
    #                             vutils.make_grid(torch.cat([fake.detach(),fake2.detach(),fake3.detach()],0).mul(0.5).add(0.5), nrow=config.batch_size),
    #                             i)

    #         fake = netG(torch.cat([sketch,sketch,sketch],0), torch.cat([hint,hint,hint],0), torch.cat([z1_r,z2_r,z3_r],0), sketch_feat=None, skeleton_output=False, has_hint=True)


    #         fake2 = netG(torch.cat([sketch,sketch,sketch],0),  torch.cat([hint,hint,hint],0), torch.cat([z4_r,z5_r,z6_r],0), sketch_feat=None, skeleton_output=False, has_hint=True)
        
    #         fake3 = netG(torch.cat([sketch,sketch,sketch],0),  torch.cat([hint,hint,hint],0), torch.cat([z7_r,z8_r,z9_r],0), sketch_feat=None, skeleton_output=False, has_hint=True)


    #         tb_logger.add_image('eval imgs_nohint_randz',
    #                             vutils.make_grid(torch.cat([fake.detach(),fake2.detach(),fake3.detach()],0).mul(0.5).add(0.5), nrow=config.batch_size),
    #                             i)


    #         #interpolation

    #         target1 = z1[0:1]
    #         target2 = z2[0:1]
    #         scale = [0.125,0.25,0.375,0.5,0.625,0.75,0.875]
    #         z121 = torch.lerp(target1,target2, scale[0])
    #         z122 = torch.lerp(target1,target2, scale[1])
    #         z123 = torch.lerp(target1,target2, scale[2])
    #         z124 = torch.lerp(target1,target2, scale[3])
    #         z125 = torch.lerp(target1,target2, scale[4])
    #         z126 = torch.lerp(target1,target2, scale[5])
    #         z127 = torch.lerp(target1,target2, scale[6])


    #         fake = netG(sketch[0:1].repeat(9,1,1,1), hint[0:1].repeat(9,1,1,1), torch.cat([target1,z121,z122,z123,z124,z125,z126,z127,target2],0), sketch_feat=None, skeleton_output=False, has_hint=True)

    #         print(fake.shape)
    #         b,c,w,h = fake.shape
    #         tb_logger.add_image('eval imgs_nohint_interpolate2',
    #                             vutils.make_grid(fake.detach().mul(0.5).add(0.5), nrow= len(scale)+2),
    #                             i)


    #         print('used time: ', time.time() - start)


if __name__ == '__main__':
    main()
