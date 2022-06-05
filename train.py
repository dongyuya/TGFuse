# Training DenseFuse network
# auto-encoder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from os.path import join
# import sys
import time
import numpy as np
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm, trange
from time import sleep
import scipy.io as scio
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import utils
from net import net
from vit import VisionTransformer

from args_fusion import args
import pytorch_msssim
from torchvision import transforms
from loss import final_ssim, final_mse, dis_loss_func
from function import Vgg16
import torch.nn.functional as F
# from aloss import a_ssim
# from hloss import h_ssim
# device = torch.device("cuda:0")


def main():
    # original_imgs_path = utils.list_images(args.dataset)
    original_imgs_path2 = utils.list_images(args.dataset2)
    train_num = args.train_num
    # original_imgs_path = original_imgs_path[:train_num]
    original_imgs_path2 = original_imgs_path2[:train_num]
    random.shuffle(original_imgs_path2)
    # for i in range(5):
    i = 2
    train(i, original_imgs_path2)


def train(i, original_imgs_path):
    batch_size = args.batch_size

    in_c = 1  # 1 - gray; 3 - RGB
    if in_c == 1:
        img_model = 'L'
    else:
        img_model = 'RGB'
    # model = Generator()
    gen = net()
    dis1 = Vgg16()
    dis2 = Vgg16()
    # vgg = Vgg16()
    # pre_model = Pre()

    if args.trans_model_path is not None:
        pre_dict = torch.load(args.trans_model_path)['state_dict']

    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        gen.load_state_dict(torch.load(args.resume))
    print(gen)

    #optimizer = Adam(model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    L1_loss = nn.L1Loss()
    # ssim_loss = final_ssim
    ssim_loss = pytorch_msssim.ssim
    bce_loss = nn.BCEWithLogitsLoss()
    writer = SummaryWriter('./log')


    if args.cuda:
        gen.cuda()
        dis1.cuda()
        dis2.cuda()
        # vgg.cuda()

    # vgg.eval()
    # dis1.eval()

    tbar = trange(args.epochs, ncols=150)
    print('Start training.....')

    # creating save path
    temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)

    temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)


    # Loss_con = []
    Loss_gen = []
    Loss_all = []
    Loss_dis1 = []
    Loss_dis2 = []

    all_ssim_loss = 0
    all_gen_loss = 0.
    all_dis_loss1 = 0.
    all_dis_loss2 = 0.
    w_num = 0
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        image_set, batches = utils.load_dataset(original_imgs_path, batch_size)
        gen.train()
        count = 0

        # if e != 0:
        #     args.lr = args.lr * 0.5
        # if args.lr < 2e-6:
        #     args.lr = 2e-6

        for batch in range(batches):

            image_paths = image_set[batch * batch_size:(batch * batch_size + batch_size)]
            directory1 = "/data/Disk_B/KAIST-RGBIR/visible"
            directory2 = "/data/Disk_B/KAIST-RGBIR/lwir"
            paths1 = []
            paths2 = []
            for path in image_paths:
                paths1.append(join(directory1, path))
                paths2.append(join(directory2, path))
            # paths = []
            # for path in image_paths:
            #     paths.append(join(args.dataset, path))

            # img = utils.get_train_images_auto(paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
            img_vi = utils.get_train_images_auto(paths1, height=args.HEIGHT, width=args.WIDTH, mode=img_model)
            img_ir = utils.get_train_images_auto(paths2, height=args.HEIGHT, width=args.WIDTH, mode=img_model)


            count += 1

            optimizer_G = Adam(gen.parameters(), args.lr)
            optimizer_G.zero_grad()

            optimizer_D1 = Adam(dis1.parameters(), args.lr_d)
            optimizer_D1.zero_grad()

            optimizer_D2 = Adam(dis2.parameters(), args.lr_d)
            optimizer_D2.zero_grad()


            if args.cuda:
                # img = img.cuda()
                img_vi = img_vi.cuda()
                img_ir = img_ir.cuda()

            outputs = gen(img_vi, img_ir)
            # resolution loss
            # img = Variable(img.data.clone(), requires_grad=False)

            con_loss_value = 0
            ssim_loss_value = 0

            ssim_loss_temp = 1 - final_ssim(img_ir, img_vi, outputs)
            # con_loss_temp = final_mse(img_ir, img_vi, outputs)
            con_loss_temp = 0


            con_loss_value += con_loss_temp
            ssim_loss_value += ssim_loss_temp

            _, c, h, w = outputs.size()
            con_loss_value /= len(outputs)
            ssim_loss_value /= len(outputs)

            # total loss
            gen_loss = ssim_loss_value + con_loss_value
            gen_loss.backward()
            optimizer_G.step()
            # scheduler.step()

#-------------------------------------------------------------------------------------------------------------------
            vgg_out = dis1(outputs.detach())[0]
            vgg_vi = dis1(img_vi)[0]
            

            dis_loss1 = L1_loss(vgg_out, vgg_vi)

            dis_loss_value1 = 0
            dis_loss_temp1 = dis_loss1
            dis_loss_value1 += dis_loss_temp1

            dis_loss_value1 /= len(outputs)

            dis_loss_value1.backward()
            optimizer_D1.step()
# ----------------------------------------------------------------------------------------------------------------
            vgg_out = dis2(outputs.detach())[2]
            vgg_ir = dis2(img_ir)[2]
            dis_loss2 = L1_loss(vgg_out, vgg_ir)

            dis_loss_value2 = 0
            dis_loss_temp2 = dis_loss1
            dis_loss_value2 += dis_loss_temp2

            dis_loss_value2 /= len(outputs)

            dis_loss_value2.backward()
            optimizer_D2.step()

            # all_con_loss += con_loss_value.item()
            all_ssim_loss += ssim_loss_value.item()
            all_dis_loss1 += dis_loss_value1.item()
            all_dis_loss2 += dis_loss_value2.item()
            all_gen_loss = all_ssim_loss
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:[{}/{}] gen loss: {:.5f} dis_ir loss: {:.5f} dis_vi loss: {:.5f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_gen_loss / args.log_interval,
                                  all_dis_loss1 / args.log_interval,
                                  all_dis_loss2 / args.log_interval
                                  #(all_con_loss + all_ssim_loss) / args.log_interval
                )
                tbar.set_description(mesg)
                # tbar.close()

                # tqdm.write(mesg)

                # all_l = (all_con_loss + all_ssim_loss) / args.log_interval
                # Loss_con.append(all_con_loss / args.log_interval)
                # Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_gen.append(all_ssim_loss / args.log_interval)
                Loss_dis1.append(all_dis_loss1 / args.log_interval)
                Loss_dis2.append(all_dis_loss2 / args.log_interval)
                # Loss_all.append((all_con_loss + all_ssim_loss) / args.log_interval)
                writer.add_scalar('gen', all_gen_loss / args.log_interval, w_num)
                writer.add_scalar('dis_ir', all_dis_loss1 / args.log_interval, w_num)
                writer.add_scalar('dis_vi', all_dis_loss2 / args.log_interval, w_num)
                # writer.add_scalar('loss_ssim', all_ssim_loss / args.log_interval, w_num)
                w_num += 1

                all_con_loss = 0.
                all_ssim_loss = 0.

            if (batch + 1) % (args.train_num//args.batch_size) == 0:
                # save model
                gen.eval()
                gen.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(gen.state_dict(), save_model_path)
                gen.train()
                gen.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
            
    gen.eval()
    gen.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(gen.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    main()
