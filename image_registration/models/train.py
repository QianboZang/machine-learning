import torch 
import torch.nn.functional as F

import wandb
import numpy as np
from models.regulizers import compute_jacobian_loss, compute_hyper_elastic_loss, compute_bending_energy 


def train_log(args, epoch, step, metrics):
    metrics_print = dict()
    for key, val in metrics.items():
        metrics_print[key] = np.nanmean(val)
        metrics[key] = []

    print(
        "==Train== epoch: {}, step: {}, ".format(epoch, step) + 
        ", ".join(["{}: {:4f}".format(key, val) for key, val in metrics_print.items()])
    )
    wandb.log(metrics_print, step)

    return metrics

def train_step(args, model, data, optimizer, metrics, device):
    pts = data["pts"].to(device)
    vals = data["vals"].to(device)
    ori_pts = pts.requires_grad_(True)

    # euler
    time0 = torch.zeros((len(pts), 1)).to(device)
    time1 = 1 / 2 * torch.ones((len(pts), 1)).to(device)
    time2 = torch.ones((len(pts), 1)).to(device)

    off_time1 = model(torch.concatenate((pts, time0), dim=1))
    tmp1 = pts + 1 / 2 * off_time1[:, :3]

    off_time2 = model(torch.concatenate((tmp1, time1), dim=1))
    tmp2 = tmp1 + 1 / 2 * off_time2[:, :3]
    
    new_pts = tmp2
    rel = new_pts - ori_pts
    # reg for velocity
    velocity_diff = off_time2[:, :3] - off_time1[:, :3]

    prds = model.sample_moving(new_pts).view(-1, 1)
    
    losses = train_backward(args, prds, vals, ori_pts, rel)
    losses["lr"] = optimizer.param_groups[0]["lr"]

    for key, val in losses.items():
        if key in metrics:
            metrics[key].append(val)
        else:
            metrics[key] = [val]
    return metrics


def train_backward(args, prds, vals, ori_pts, rel):
    # Compute loss
    loss = F.l1_loss(prds, vals) * args.std_val
    losses = dict(MAE=loss.item())
    reg_jacob = compute_jacobian_loss(ori_pts, rel, batch_size=args.batch_size) * 256
    reg_hyper = compute_hyper_elastic_loss(ori_pts, rel, batch_size=args.batch_size) * 256
    reg_bending = compute_bending_energy(ori_pts, rel, batch_size=args.batch_size) * 256

    loss += reg_jacob * args.lambda_jacob 

    losses["reg_jacob"] = reg_jacob.item()
    loss = loss / args.num_step_opt
    loss.backward()

    offs_abs = rel.abs() * 256
    losses.update(
        dict(
            off_mean=offs_abs.mean().item(),
            off_max=offs_abs.max().item(),
            off_std=offs_abs.std().item(),
        )
    )
    return losses