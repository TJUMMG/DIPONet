import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') and not f.endswith('edge.png')]
        self.egs = [gt_root + f for f in os.listdir(gt_root) if f.endswith('edge.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.egs = sorted(self.egs)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([ 
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.eg_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index): 
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        eg = self.binary_loader(self.egs[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        eg = self.eg_transform(eg)
        return {'sal_image': image, 'sal_label': gt, 'sal_edge': eg}

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB') 

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 

    def __len__(self):
        return self.size 


def get_loader(image_root, gt_root, trainsize, batchsize, shuffle=True, num_workers=1, pin_memory=False, drop_last=True):

    dataset = SalObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=drop_last)
    return data_loader 

class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        shape = image.size
        shape = (shape[1], shape[0])
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.')[0]
        self.index += 1
        return image, shape, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

        