# test phase

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
from function import adaptive_instance_normalization

# from u2net_train import args
import utils

import numpy as np
import time

# from trans_net import ipt_base
from net import net

# normalize the predicted SOD probability map
def load_model(path):
    # fuse_net = net()
    # pre_dict = torch.load(path)
    # new_pre = {}
    # for k, v in pre_dict.items():
    #     name = k[7:]
    #     new_pre[name] = v
    #
    # fuse_net.load_state_dict(new_pre)

    fuse_net = net()
    fuse_net.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))

    fuse_net.eval()
    fuse_net.cuda()

    return fuse_net


def _generate_fusion_image(model, vi, ir):

    # vi_f, w_vi, vi_w = model.en_vi(vi)
    # ir_f, w_ir, ir_w = model.en_ir(ir)
    # fuse = model.fusion(vi_f, ir_f)
    # out = model.de(fuse)
    # vi_f = model.en_vi(vi)
    # ir_f = model.en_ir(ir)
    #
    # fusion = model.fusion(vi_f, ir_f)

    # en_img, ori_shape = model.en(vi)

    out = model(vi, ir)
    return out


def run_demo(model, vi_path, ir_path, output_path_root, index):
    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)

    out = utils.get_image(vi_path, height=None, width=None)
    # dim = img_ir.shape

    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)
    # dimension = con_img.size()

    img_fusion = _generate_fusion_image(model, vi_img, ir_img)
    ############################ multi outputs ##############################################
    file_name = 'fusion_' + str(index) + '.png'
    output_path = output_path_root + file_name
    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    img = img.astype('uint8')
    utils.save_images(output_path, img, out)
    # utils.save_images(output_path, img, out)
    print(output_path)


def main():
    vi_path = "images/Test_vi/"
    ir_path = "images/Test_ir/"
    # network_type = 'densefuse'

    output_path = './outputs/'
    # strategy_type = strategy_type_list[0]

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    in_c = 1
    out_c = in_c
    model_path = "./models/Epoch_19_iters_2500.model"
    #model_path = "./models/Final_epoch_50.model"

    with torch.no_grad():

        model = load_model(model_path)
        for i in range(10):
            index = i + 1
            visible_path = vi_path + str(index) + '.bmp'
            infrared_path = ir_path + str(index) + '.bmp'
            start = time.time()
            run_demo(model, visible_path, infrared_path, output_path, index)
            end = time.time()
            print('time:', end - start, 'S')
    print('Done......')


if __name__ == "__main__":
    main()
