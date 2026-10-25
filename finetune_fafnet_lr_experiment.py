"""
Stockade -- FaFNet LR-schedule diagnostic (Experiment A from the review plan).

The original 100-epoch lowerbound run used a scheduler built for 200 epochs
(MultiStepLR milestones=[50,100,150,200]), so it only ever passed the first
decay step. This resumes from that run's final weights with a FRESH
optimizer and a schedule sized for a short follow-up run, to test whether
the epoch-33 plateau was an LR artifact or genuine convergence.

Controlled-experiment discipline (per the plan): only the LR schedule
changes. Same model (FaFNet/lowerbound), same data/split, same batch size,
same optimizer type -- nothing else.

- Loads ONLY model_state_dict from the epoch-100 checkpoint (not optimizer/
  scheduler state -- those get a clean slate for the new schedule).
- New schedule: Adam lr=0.0005 (continuing from where epoch 100 left off),
  MultiStepLR milestones=(15, 30), gamma=0.5 -- compressed to fit a 40-epoch
  budget instead of the original's 50/100/150/200.
- Checkpoints go to a SEPARATE logs dir (logs_lr_experiment/) so the
  original run's checkpoints are never touched, and the two are directly
  comparable side by side.

Usage:
    python finetune_fafnet_lr_experiment.py --nepoch 40
"""

import argparse
import os
import sys

sys.path.insert(0, "reference_coperception")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from coperception.models.det import FaFNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/V2X-Sim-det-001/V2X-Sim-det/train")
    parser.add_argument("--init-checkpoint", default="logs/lowerbound/no_rsu/epoch_100.pth")
    parser.add_argument("--logpath", default="logs_lr_experiment")
    parser.add_argument("--nepoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nworker", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0005, help="Starting LR -- continues from epoch 100's LR, not reset to the original 0.001.")
    parser.add_argument("--milestones", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"LR schedule: start={args.lr}, milestones={args.milestones}, gamma={args.gamma}, nepoch={args.nepoch}")

    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    config.flag = "lowerbound"

    num_agent = 6
    agent_idx_range = range(1, num_agent)  # no RSU, matches the original run

    dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        bound="lowerbound",
        kd_flag=0,
        rsu=False,
    )
    print(f"Training dataset size: {len(dataset)}")
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.nworker)

    model = FaFNet(config, layer=3, kd_flag=0, num_agent=num_agent - 1, compress_level=0)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # The checkpoint was saved from an already-DataParallel-wrapped model
    # (train_codet.py wraps before saving), so its keys carry the "module."
    # prefix -- load AFTER wrapping here too, or this mismatches and raises.
    init_ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_state_dict(init_ckpt["model_state_dict"])
    print(f"Loaded model weights from {args.init_checkpoint} (was epoch {init_ckpt['epoch']})")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    faf_module = FaFModule(
        model, model, config, optimizer, criterion, kd_flag=0,
        lr_milestones=tuple(args.milestones), lr_gamma=args.gamma,
    )

    save_dir = os.path.join(args.logpath, "lowerbound", "no_rsu")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")
    log_file = open(log_path, "w")
    log_file.write(f"LR experiment: start={args.lr}, milestones={args.milestones}, gamma={args.gamma}\n")
    log_file.write(f"Initialized from {args.init_checkpoint}\n\n")
    log_file.flush()

    for epoch in range(1, args.nepoch + 1):
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}, learning rate {lr}")

        running_loss = 0.0
        running_cls = 0.0
        running_loc = 0.0
        n_batches = 0

        faf_module.model.train()
        t = tqdm(loader)
        for sample in t:
            (
                padded_voxel_point_list,
                padded_voxel_points_teacher_list,
                label_one_hot_list,
                reg_target_list,
                reg_loss_mask_list,
                anchors_map_list,
                vis_maps_list,
                target_agent_id_list,
                num_agent_list,
                trans_matrices_list,
            ) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1) - 1  # no rsu

            padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)
            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {
                "bev_seq": padded_voxel_point.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "vis_maps": vis_maps.to(device),
                "target_agent_ids": target_agent_id.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices": trans_matrices,
            }

            loss, cls_loss, loc_loss = faf_module.step(data, args.batch_size, num_agent=num_agent - 1)
            running_loss += loss
            running_cls += cls_loss
            running_loc += loc_loss
            n_batches += 1
            t.set_description(f"Epoch {epoch}, lr {lr}")
            t.set_postfix(cls_loss=running_cls / n_batches, loc_loss=running_loc / n_batches)

        faf_module.scheduler.step()

        log_file.write(
            f"epoch: {epoch}, lr: {lr}\tTotal loss {running_loss / n_batches:.6f}\t"
            f"classification Loss {running_cls / n_batches:.6f}\t"
            f"Localization Loss {running_loc / n_batches:.6f}\n"
        )
        log_file.flush()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": faf_module.model.state_dict(),
                "optimizer_state_dict": faf_module.optimizer.state_dict(),
                "scheduler_state_dict": faf_module.scheduler.state_dict(),
                "loss": running_loss / n_batches,
            },
            os.path.join(save_dir, f"epoch_{epoch}.pth"),
        )

    log_file.close()
    print(f"\nDone. Checkpoints and log under {save_dir}")


if __name__ == "__main__":
    main()
