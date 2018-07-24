import scipy.io as sio
import glob, os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file
import random
from PIL import Image


class VoxelDataset(BaseDataset):
    def initialize(self, opt):
        self.voxel_dir = os.path.join(opt.dataroot, opt.phase + '_voxels')
        self.images_dir = os.path.join(opt.dataroot, opt.phase + '_imgs')
        self.voxel_paths = self.make_voxel_dataset()
        self.image_paths = self.make_image_dataset()

        self.voxel_paths = sorted(self.voxel_paths)
        self.image_paths = sorted(self.image_paths)
        self.voxel_size = len(self.voxel_paths)
        self.image_size = len(self.image_paths)
        print('called')

        self.transform = get_transform(opt)
    def name(self):
        return 'BaseDataset'

    def __len__(self):
        return max(self.image_size, self.voxel_size)

    def make_voxel_dataset(self):
        voxels = []
        print(self.voxel_dir)
        for dir in os.walk(self.voxel_dir):
            if "train_voxels/" in dir[0] and os.stat(dir[0] + '/model.mat').st_size is not 0:
                dir = (dir[0] + '/model.mat')
                voxels.append(dir)
                # mat = sio.loadmat(dir[0] + '/model.mat')
                # mat['input'][0]
        
        return voxels

    def __getitem__(self, index):
        image_path = self.image_paths[index % self.image_size]
        temp_voxel_path = self.voxel_dir +  image_path.split('/')[-2] + '/model.mat'
        if temp_voxel_path in self.voxel_paths:
            voxel_path = temp_voxel_path
        else:
            voxel_path = self.voxel_paths[random.randint(0, self.voxel_size - 1)]
        # Load Voxel
        voxel = sio.loadmat(voxel_path)
        voxel['input'][0]
        print(voxel)

        # Load Image
        img = Image.open(image_path).convert('RGB')
        A = self.transform(img)
        if self.opt.output_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            image = tmp.unsqueeze(0)

        return {'voxel': voxel, 'image': image, 'voxel_path': voxel_path, 'image_path' image_path}


    def make_image_dataset(self):
        images = []
        walked = os.walk(self.images_dir)
        for root, _, fnames in walked:
            for image in fnames:
                if is_image_file(image):
                    path = os.path.join(root, image)
                    images.append(path)
        
        return images



