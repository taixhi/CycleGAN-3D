import scipy.io as sio
import glob, os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file

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
    def name(self):
        return 'BaseDataset'

    def __len__(self):
        return max(self.image_size, self.voxel_size)

    def make_voxel_dataset(self):
        voxels = []
        for dir in os.walk(self.voxel_dir):
            if "./train_voxels/" in dir[0] and os.stat(dir[0] + '/model.mat').st_size is not 0:
                images.append(dir[0] + '/model.mat')
                # mat = sio.loadmat(dir[0] + '/model.mat')
                # mat['input'][0]
        
        return voxels

    def make_image_dataset(self):
        images = []
        for root, _, fnames in os.walk(self.images_dir):
            if is_image_file(fnames):
                path = os.path.join(root, fname)
                images.append(path)
        
        return images



