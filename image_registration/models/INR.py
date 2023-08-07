import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedder import Embedder


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class NeRF(nn.Module):
    def __init__(self, args, D=8, W=256, input_ch=4, output_ch=4, skips=[4], multi_res=8):
        super(NeRF, self).__init__()
        self.skips = skips
        self.embedder = Embedder(input_ch, False, multi_res - 1, multi_res)
        input_ch = self.embedder.out_dim

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W)]
            + [
                SineLayer(W, W)
                if i not in self.skips
                else SineLayer(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.output_linear = nn.Linear(W, output_ch, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.output_linear.weight)

        moving = sitk.ReadImage(args.moving_vol_path)
        moving = sitk.GetArrayFromImage(moving)
        moving_mask = sitk.ReadImage(
            args.moving_vol_path.replace(".mha", "_mask.mha")
        )
        moving_mask = sitk.GetArrayFromImage(moving_mask)
        moving = torch.from_numpy(
            moving_mask * (moving - args.min_val) / args.std_val
        ).float()
        self.register_buffer('moving', moving)

    def forward(self, x):
        # print(x.shape)
        x = self.embedder(x)
        identity = x
        for i, linear in enumerate(self.pts_linears):
            x = linear(x)
            if i in self.skips:
                x = torch.cat([identity, x], -1)

        outputs = self.output_linear(x)

        return outputs

    def sample_moving(self, pts):
        for i in range(5 - len(pts.shape)):
            pts = pts.unsqueeze(0)
        vals = F.grid_sample(self.moving[None, None], pts, align_corners=False)
        return vals
