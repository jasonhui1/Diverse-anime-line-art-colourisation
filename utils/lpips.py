
import torch
import lpips

from torch.nn.functional import avg_pool2d
import numpy as np


@torch.no_grad()
def calculate_lpips(netG, dataloader, config, dims, use_z=False, use_msg=False, input_all=0, use_MN=False, netMN=None, use_hint=False):
    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex',version=0.1)
    loss_fn.cuda()
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_values = []
    # num_rand_outputs = len(group_of_images)
    num_rand_outputs = 20
    total_lpips = 0

    count = 0

    data_iter = iter(dataloader)
    for index, real_sim in enumerate(data_iter):

        if index < num_rand_outputs:
            real_sim = real_sim.cuda()
            zhint = torch.zeros(real_sim.shape[0], 4, config.image_size // 4, config.image_size // 4).float().cuda()

            if(input_all > 0):
                real_sims = [real_sim,real_sim] + [avg_pool2d(real_sim, int(np.power(2, i)))
                                for i in range(1, input_all)]

            fakes = []
            for i in range(num_rand_outputs):            
                if use_z:
                    z = torch.randn(real_sim.shape[0], config.nz, device=config.device)
                    if(use_MN):
                        z = netMN(z)

                    if(use_msg):
                        if(input_all > 0):
                            fake_cim = netG(real_sims, zhint,z, input_all=True)[-1].squeeze()
                        else:
                            fake_cim = netG(real_sim, zhint,z)[-1].squeeze()

                    else:
                        if(use_hint):
                            fake_cim = netG(real_sim, zhint,z, has_hint=True).squeeze()
                        else:
                            fake_cim = netG(real_sim, zhint,z).squeeze()


                else:
                    if(use_msg):
                        if(input_all > 0):
                            fake_cim = netG(real_sims, zhint, input_all=True)[-1].squeeze()
                        else:
                            fake_cim = netG(real_sim, zhint)[-1].squeeze()
                    else:
                        fake_cim = netG(real_sim, zhint).squeeze()

                fakes.append(fake_cim)


            # calculate the average of pairwise distances among all random outputs
            for i in range(num_rand_outputs-1):
                for j in range(i+1, num_rand_outputs):
                    lpips_values.append(loss_fn.forward(fakes[i], fakes[j]))
            lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
            total_lpips += lpips_value.item() 


        else:
            break



    return total_lpips/(num_rand_outputs)