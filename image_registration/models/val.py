import torch
import torch.nn.functional as F
from torchvision.utils import flow_to_image

import wandb


def val_log(args, step, mae, movs, prds, gts, offs):
    print("==Test== MAE-ave: {:4f}".format(mae.item()))

    mov = torch.cat(movs, 1)
    prd = torch.cat(prds, 1)
    gt = torch.cat(gts, 1)
    visuals = torch.cat((mov, prd, gt), 0)

    diff_ori = gt - mov
    diff_new = gt - prd
    diffs = torch.cat((diff_ori, diff_new), 0)
    diff = torch.zeros(3, *diffs.shape)
    diff[0] = (diffs > 0) * diffs
    diff[1] = (diffs < 0) * diffs.abs()

    off = torch.cat(offs, 1)
    off = off.permute(2, 0, 1)[:2]
    flow = flow_to_image(off).float()

    wandb.log(
        {
            "plane": wandb.Image(visuals * 1020),
            "flow": wandb.Image(flow),
            "diff": wandb.Image(diff),
        },
        step,
    )
    print("Draw images at step: {}".format(step))



def val_log(args, step, mae, movs, prds, gts, offs):
    print("==Test== MAE-ave: {:4f}".format(mae.item()))

    mov = torch.cat(movs, 1)
    prd = torch.cat(prds, 1)
    gt = torch.cat(gts, 1)
    visuals = torch.cat((mov, prd, gt), 0)

    diff_ori = gt - mov
    diff_new = gt - prd
    diffs = torch.cat((diff_ori, diff_new), 0)
    diff = torch.zeros(3, *diffs.shape)
    diff[0] = (diffs > 0) * diffs 
    diff[1] = (diffs < 0) * diffs.abs() 

    off = torch.cat(offs, 1)
    off = off.permute(2, 0, 1)[:2]
    flow = flow_to_image(off).float()
    # print(diff.shape)

    wandb.log(
        {
            "plane": wandb.Image(visuals * 1020),
            "flow": wandb.Image(flow),
            "diff": wandb.Image(diff),
        },
        step,
    )
    print("Draw images at step: {}".format(step))


def val_step(args, dataset, model, step, device):
    maes, movs, prds, gts, offs = [], [], [], [], []
    for i in range(args.num_val_planes):
        pts, vals = dataset.random_plane()
        H, W, _ = pts.shape
        pts = pts.reshape(-1, 3).to(device)
        vals = vals.reshape(-1, 1).to(device)

        movs_i, prds_i, offs_i = [], [], []
        for j in range(0, H * W, args.chunk_size):
            pts_j = pts[j : j + args.chunk_size]
            vals_j = vals[j : j + args.chunk_size]
            # euler
            with torch.no_grad():
                time0 = torch.zeros((len(pts_j), 1)).to(device)
        
                off_time1 = model(torch.concatenate((pts_j, time0), dim=1))
                tmp1 = pts_j + 1 / 2 * off_time1[:, :3]
                time1 = 1 / 2 * torch.ones((len(pts_j), 1)).to(device)
        
                off_time2 = model(torch.concatenate((tmp1, time1), dim=1))
                tmp2 = tmp1 + 1 / 2 * off_time2[:, :3]
        
            new_pts_j = tmp2
            offs_j = new_pts_j - pts_j
            
            movs_j = model.sample_moving(pts_j).view(-1, 1)
            prds_j = model.sample_moving(new_pts_j).view(-1, 1)

            mae = F.l1_loss(prds_j, vals_j) * args.std_val
            maes.append(mae[None])
            movs_i.append(movs_j)
            prds_i.append(prds_j)
            offs_i.append(offs_j)

        movs_i = torch.cat(movs_i)
        prds_i = torch.cat(prds_i)
        offs_i = torch.cat(offs_i)

        movs.append(movs_i.view(H, W))
        prds.append(prds_i.view(H, W))
        gts.append(vals.view(H, W))
        offs.append(offs_i.view(H, W, 3))

    mae = torch.cat(maes).mean()
    val_log(args, step, mae, movs, prds, gts, offs)