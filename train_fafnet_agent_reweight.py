"""
Stockade -- agent-5/4 data-scarcity follow-up experiment.

Resumes from the LR-experiment's best checkpoint (epoch 40, 65.52%/61.71%
mAP) and oversamples scene-frames where agent4/agent5 have real
(non-placeholder) data, via build_agent_sample_weights.py's
WeightedRandomSampler weights -- testing whether more training exposure
to their sparse real signal closes the per-agent mAP gap.

Only the sampler changes relative to the LR-experiment run. Same model,
same batch size, continues the same LR trajectory (0.000125, no further
scheduled decay within this run's modest epoch budget) -- isolating the
sampling-strategy variable.

Usage:
    python train_fafnet_agent_reweight.py --nepoch 20
"""

import argparse
import os
import sys

sys.path.insert(0, "reference_coperception")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from coperception.models.det import FaFNet

from build_agent_sample_weights import compute_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/V2X-Sim-det-001/V2X-Sim-det/train")
    parser.add_argument("--init-checkpoint", default="logs_lr_experiment/lowerbound/no_rsu/epoch_40.pth")
    parser.add_argument("--logpath", default="logs_agent_reweight")
    parser.add_argument("--nepoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nworker", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.000125, help="Continues from where the LR experiment left off.")
    parser.add_argument("--weights-file", default="agent_sample_weights.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    config.flag = "lowerbound"

    num_agent = 6
    agent_idx_range = range(1, num_agent)

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

    if os.path.exists(args.weights_file):
        weights = np.load(args.weights_file)
    else:
        weights = compute_weights(args.data)
        np.save(args.weights_file, weights)
    assert len(weights) == len(dataset), f"weight count {len(weights)} != dataset size {len(dataset)}"

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.nworker)

    model = FaFNet(config, layer=3, kd_flag=0, num_agent=num_agent - 1, compress_level=0)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    init_ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_state_dict(init_ckpt["model_state_dict"])
    print(f"Loaded model weights from {args.init_checkpoint} (was epoch {init_ckpt['epoch']})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    # No further decay within this modest budget -- isolating the sampler as the only new variable.
    faf_module = FaFModule(model, model, config, optimizer, criterion, kd_flag=0, lr_milestones=(1000,), lr_gamma=0.5)

    save_dir = os.path.join(args.logpath, "lowerbound", "no_rsu")
    os.makedirs(save_dir, exist_ok=True)
    log_file = open(os.path.join(save_dir, "log.txt"), "w")
    log_file.write(f"Agent reweight experiment: lr={args.lr} (fixed), oversampling agent4/5 real scenes\n")
    log_file.write(f"Initialized from {args.init_checkpoint}\n\n")
    log_file.flush()

    for epoch in range(1, args.nepoch + 1):
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}, learning rate {lr}")

        running_loss = running_cls = running_loc = 0.0
        n_batches = 0

        faf_module.model.train()
        t = tqdm(loader)
        for sample in t:
            (
                padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list,
                reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list,
                target_agent_id_list, num_agent_list, trans_matrices_list,
            ) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1) - 1

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
            f"epoch: {epoch}, lr: {lr}\tTotal loss {running_loss/n_batches:.6f}\t"
            f"classification Loss {running_cls/n_batches:.6f}\tLocalization Loss {running_loc/n_batches:.6f}\n"
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
