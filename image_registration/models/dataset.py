import numpy as np
import numpy.random as random
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class RayDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size

        fixed = sitk.ReadImage(args.fixed_vol_path) # fixed_vol_path is the path to the fixed volume
        self.props = [
            fixed.GetSize(),
            fixed.GetOrigin(),
            fixed.GetSpacing(),
            fixed.GetDirection(),
        ]
        fixed = sitk.GetArrayFromImage(fixed) # fixed is the fixed volume
        fixed_mask = sitk.ReadImage(
            args.fixed_vol_path.replace(".mha", "_mask.mha")
        ) 
        fixed_mask = sitk.GetArrayFromImage(fixed_mask) # fixed_mask is the mask of the fixed volume
        self.fixed = torch.from_numpy(
            fixed_mask * (fixed - args.min_val) / args.std_val
        ) # fixed is the fixed volume after normalization

        self.H, self.D, self.W = self.fixed.shape # H is the height, D is the depth, W is the width
        self.vol_size = self.H * self.D * self.W # vol_size is the total number of voxels in the fixed volume
        self.vol_shape = torch.tensor(([self.W, self.D, self.H])) # vol_shape is the shape of the fixed volume

        index_h = torch.linspace(-1.0, 1.0, self.H + 1)[:-1] + 1 / self.H # index_h is the height index
        index_d = torch.linspace(-1.0, 1.0, self.D + 1)[:-1] + 1 / self.D # index_d is the depth index
        index_w = torch.linspace(-1.0, 1.0, self.W + 1)[:-1] + 1 / self.W # index_w is the width index
        grid_h, grid_d, grid_w = torch.meshgrid(
            [index_h, index_d, index_w], indexing="ij"
        ) # grid_h, grid_d, grid_w are the height, depth, and width grids
        pts = torch.stack([grid_w, grid_d, grid_h], -1) # pts is the grid of the fixed volume
        self.pts = pts

        self.indexs = np.repeat(np.arange(self.vol_size), args.num_repeat)

        self.pts_vec = self.pts.view(-1, 3)
        self.vals_vec = self.fixed.view(-1, 1)

    def __len__(self):
        return self.vol_size // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        indexs = self.indexs[start: end]
        pts = self.pts_vec[indexs]
        vals = self.vals_vec[indexs]

        return dict(pts=pts, vals=vals)

    def shuffle(self):
        random.shuffle(self.indexs)

    def random_plane(self, view="axial"):
        if view == "axial":
            index = random.choice(range(self.H))
            pts = self.pts[index, :, :]
            vals = self.fixed[index, :, :]
        elif view == "sagital":
            index = random.choice(range(self.D))
            pts = self.pts[:, index, :]
            vals = self.fixed[:, index, :]
        elif view == "axial":
            index = random.choice(range(self.W))
            pts = self.pts[:, :, index]
            vals = self.fixed[:, :, index]
        return pts, vals

    def all_planes(self):
        return self.pts, self.fixed

    def tensor2vol(self, tensor, dvf=False):
        if dvf:
            tensor = tensor * self.vol_shape.to(tensor.device)
        else:
            tensor = tensor * self.args.std_val + self.args.min_val
        array = tensor.cpu().numpy()
        vol = sitk.GetImageFromArray(array)
        if dvf:
            vol = sitk.Cast(vol, sitk.sitkVectorFloat32)
        else:
            vol = sitk.Cast(vol, sitk.sitkInt16)
        vol.SetOrigin(self.props[1])
        vol.SetSpacing(self.props[2])
        vol.SetDirection(self.props[3])
        return vol
