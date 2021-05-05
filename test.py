import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import time
from scipy import misc
# from model.DIPONet_VGG import DIPONet_VGG
from model.DIPONet_ResNet import DIPONet_ResNet
from data import test_dataset

def test(test_loader, save_path):
    time_s = time.time()
    for i in range(test_loader.size):
        image, shape, name = test_loader.load_data()
        image = image.cuda()
        _, _, edge, res = model(image)
        res = F.upsample(res, size=shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name+'.png', res)
        edge = F.upsample(edge, size=shape, mode='bilinear', align_corners=False)
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        misc.imsave(save_path+name+'_edge.png', edge)


    time_e = time.time()
    print('Speed: %f FPS' % (test_loader.size / (time_e - time_s)))
    print('Test Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=448, help='testing size')
    opt = parser.parse_args()

    # model = DIPONet_VGG(pretrained=False)
    # model.load_state_dict(torch.load('./DIPONetByAuthors/vgg/DIPONet_VGG.pth'))
    model = DIPONet_ResNet(pretrained=False)
    model.load_state_dict(torch.load('./DIPONetByAuthors/resnet/DIPONet_ResNet.pth'))
    model.cuda()
    model.eval()

    dataset = 'HKU-IS'
    # save_path = './testresults/DIPONet_VGG/' + dataset + '/'
    save_path = './testresults/DIPONet_ResNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # test_root = 'E:/MyStudy/SOD_data/PASCALS/Imgs/'
    # test_root = 'E:/MyStudy/SOD_data/DUTOMRON/Imgs/'
    test_root = 'E:/MyStudy/SOD_data/HKU-IS/Imgs/'
    # test_root = 'E:/MyStudy/SOD_data/MSRA-B/Imgs/'
    # test_root = 'E:/MyStudy/SOD_data/DUTS/DUTS-TE/DUTS-TE-Image/'
    # test_root = 'E:/MyStudy/SOD_data/SOD/Images/'
    test_loader = test_dataset(test_root, opt.testsize)

    test(test_loader, save_path)
