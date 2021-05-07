import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os, argparse
from datetime import datetime

# from model.DIPONet_VGG import DIPONet_VGG
from model.DIPONet_ResNet import DIPONet_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr, bce2d_new, CEL, print_network
import torchvision.utils as vutils
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, data_batch in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts, gt_egs = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge']
        images = Variable(images)
        gts = Variable(gts)
        gt_egs = Variable(gt_egs)
        images = images.cuda()
        gts = gts.cuda() 
        gt_egs = gt_egs.cuda()

        spvedge1, globalsal1, spvedge2, globalsal2 = model(images)
        loss1 = bce2d_new(spvedge1, gt_egs)
        loss2 = 0.8*bce_loss(globalsal1, gts) + 0.2*CEL()(globalsal1, gts)
        loss3 = bce_loss(spvedge2, gt_egs)
        loss4 = 0.8*bce_loss(globalsal2, gts) + 0.2*CEL()(globalsal2, gts)

        loss = 0.6*loss1 + 0.6*loss2 + loss3 + loss4
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 50 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss1.data, loss2.data,
                    loss3.data, loss4.data))
        tmp_path = './tmp_path/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        if i % 100 == 0:
            vutils.save_image(torch.sigmoid(spvedge1.data), tmp_path +'iter%d-edge-1.jpg' % i, normalize=True, padding = 0)
            vutils.save_image(torch.sigmoid(globalsal1.data), tmp_path +'iter%d-sal-1.jpg' % i, normalize=True, padding = 0)
            vutils.save_image(torch.sigmoid(spvedge2.data), tmp_path +'iter%d-edge-2.jpg' % i, normalize=True, padding = 0)
            vutils.save_image(torch.sigmoid(globalsal2.data), tmp_path +'iter%d-sal-2.jpg' % i, normalize=True, padding = 0)
            vutils.save_image(images.data, tmp_path +'iter%d-sal-data.jpg' % i, padding = 0)
            vutils.save_image(gts.data, tmp_path +'iter%d-sal-target.jpg' % i, padding = 0)
            vutils.save_image(gt_egs.data, tmp_path +'iter%d-sal-edge.jpg' % i, padding = 0)

    # save_path = './trainresults/vgg/'
    save_path = './trainresults/resnet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '%d' % (epoch+1) + '.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=26, help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=5, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=list, default=[14, 22], help='every n epochs decay learning rate')
    opt = parser.parse_args()


    # model = DIPONet_VGG(pretrained=True)
    model = DIPONet_ResNet(pretrained=True)
    model.cuda() 
    print_network(model)
    params = model.parameters() 
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = 'E:/MyStudy/SOD_data/DUTS/DUTS-TR/DUTS-TR-Image/'
    gt_root = 'E:/MyStudy/SOD_data/DUTS/DUTS-TR/DUTS-TR-Mask/'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    bce_loss = torch.nn.BCEWithLogitsLoss() 

    for epoch in range(opt.epoch):
        adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
