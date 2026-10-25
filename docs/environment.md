# Environment, checkpoint provenance, and exact commands behind every reported metric

**Note:** the commands below were originally run with default output paths
(`risk_dataset.csv` etc. at the project root). Those files have since been
moved into `experiments/` as the preserved record of this run. Re-running
the commands as documented will regenerate fresh copies at the root —
diff or move them into `experiments/` under a new name if you want to
keep both as distinct experiment records.

## Environment

- Python 3.11.9, Windows-native venv at `stockade-env/` (NOT `.venv/` — that
  folder is a dead leftover, a Linux-layout venv with nothing installed;
  safe to ignore/delete).
- Key packages: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`
  (CUDA 12.8 build, required for the RTX 5060 Ti), `ultralytics`,
  `opencv-python`, `scikit-learn==1.9.1`, `pandas`, `numpy`, `scipy`,
  `numba`/`llvmlite`, `nuscenes-devkit==1.0.9` (installed but its `Box`
  class is shadowed by our local stub — see `PATCHES.md`).
- `reference_coperception/` is a patched local clone of
  https://github.com/coperception/coperception — see `PATCHES.md` for the
  three changes and why each was necessary.
- GPU: NVIDIA RTX 5060 Ti (16GB), 16 CPU cores.

## Dataset provenance

- `data/DAD/` — Dashcam Accident Dataset, 1,750 clips
  (`videos/{training,testing}/{positive,negative}/*.mp4`), 1280x720 @
  25fps, 100 frames/clip. **Only clip-level crash/normal labels exist in
  this copy** (from folder name) — no frame-level time-of-accident
  annotations were found anywhere in the project (searched exhaustively,
  see conversation log). `000828.mp4` is corrupted ("moov atom not
  found") and permanently unprocessable.
- `data/V2X-Sim-det-001/` — preprocessed V2X-Sim detection task data,
  8,000 train / 1,000 val / 1,000 test scene-frames, 6 agents each
  (agent0 = RSU, agent1-5 = vehicles), extracted from
  `V2X-Sim-det-001.zip`.
  - **Known duplicate:** a second, pre-existing extraction of the same
    zip sits at the top-level `V2X-Sim-det-001/` (not under `data/`,
    ~73GB, predates this project's work on it). Only `data/V2X-Sim-det-001`
    is referenced by any script. The top-level copy is dead weight
    pending a decision to delete it.
- `data/CCD-Normal/`, `data/nuscenes-mini/` — extracted, not yet used by
  any script.

## Exact commands behind every reported metric

### Phase 1 — tracking (full DAD set)
```
python track_vehicles.py --input data/DAD --output outputs_dad_full
```
Run twice: first attempt crashed at clip 1,293/1,750 on the corrupted
`000828.mp4` (script had no per-clip error handling at the time); the
script was then hardened (skip-if-output-exists + catch-and-continue per
clip) and re-run, which skipped the 1,128 already-done clips and finished
the remaining ~457. Final result: 1,749/1,750 clips processed
(`outputs_dad_full/**/*_tracks.csv`).

### Phase 2 — rule-based scorer (v3), full-scale evaluation
```
python score_risk.py --tracks-dir outputs_dad_full --videos-dir data/DAD > outputs/phase2_results_full.txt
```
Result (620 crash / 1,129 normal clips):
| threshold | recall | precision | false-alarm rate |
|---|---|---|---|
| 0.3 | 56% | 43% | 42% |
| 0.5 | 38% | 44% | 27% |
| 0.7 | 12% | 45% | 8% |

85% of false positives at threshold 0.5 trace to the "closing fast" rule
specifically (264/309 false positives), vs. 15% from "weaving" — computed
by parsing the per-clip `reason` column in
`outputs/phase2_results_full.txt`.

### Phase 2 — learned classifier dataset + training
```
python build_risk_dataset.py --tracks-dir outputs_dad_full --videos-dir data/DAD --output risk_dataset.csv
python build_risk_dataset_v2.py --tracks-dir outputs_dad_full --videos-dir data/DAD --output risk_dataset_v2.csv
# merged on [clip, split, label] (NOT clip alone -- clip numbers repeat across
# split/label folders) -> risk_dataset_combined.csv
python train_risk_classifier.py --data risk_dataset_combined.csv --feature-set v1
python train_risk_classifier.py --data risk_dataset_combined.csv --feature-set v2
python train_risk_classifier.py --data risk_dataset_combined.csv --feature-set combined
```
1,690 clips total (1,233 train / 457 test, split exactly along DAD's own
training/testing folders). Result:
| feature set | LogReg PR-AUC | GBT PR-AUC |
|---|---|---|
| v1 (geometric) | 0.392 | 0.418 |
| v2 (ego-compensated real-TTC) | 0.419 | 0.363 |
| combined | 0.411 | 0.394 |

No variant beats the rule-based scorer's precision/false-alarm rate.
No-skill PR-AUC floor for this class balance (~36% crash prevalence in
test) is ~0.36.

### FaFNet (`lowerbound`) training
```
PYTHONPATH=reference_coperception ./stockade-env/Scripts/python.exe reference_coperception/tools/det/train_codet.py \
  -d data/V2X-Sim-det-001/V2X-Sim-det/train \
  --com lowerbound --nepoch 100 --nworker 6 --batch_size 4 --log --logpath logs
```
Checkpoints: `logs/lowerbound/no_rsu/epoch_1.pth` .. `epoch_100.pth`
(~90MB each, ~9GB total, NOT tracked in git — paths recorded here instead).
Ran Oct 7 22:12 -> Oct 9 10:10 (~36 hours), steady ~21-22 min/epoch, no
interruption or resume. Scheduler: `MultiStepLR`, milestones=[50,100,150,200],
gamma=0.5 — only epoch-50 decay was reached; the run stopped exactly at
the epoch-100 milestone without passing it.

### FaFNet evaluation (epoch 33 and epoch 100)
```
PYTHONPATH=reference_coperception ./stockade-env/Scripts/python.exe reference_coperception/tools/det/test_codet.py \
  -d data/V2X-Sim-det-001/V2X-Sim-det/val \
  --com lowerbound --resume logs/lowerbound/no_rsu/epoch_33.pth --nworker 2 --visualization 0
# (repeated with epoch_100.pth --nworker 4)
```
Evaluated on the 1,000-scene held-out val split (not the test split —
test is still untouched, per the plan's instruction to keep it isolated
until model selection is complete).

| checkpoint | mAP@0.5 | mAP@0.7 |
|---|---|---|
| epoch 33 | 63.2% | 60.6% |
| epoch 100 | 64.1% | 60.9% |

Per-agent (epoch 100, mAP@0.5): agent1 61.5%, agent2 63.5%, agent3 69.5%,
agent4 64.4%, agent5 58.8% — agent 5 consistently weakest at both
checkpoints; cause not yet diagnosed.

### DiscoNet — blocked, not trained
Smoke-tested only. `DiscoNet.forward()` with `kd_flag=0` returns
`(result, save_agent_weight_list)`, but `CoDetModule.FaFModule.step()`
only unpacks that shape when `kd_flag=1` — confirmed from source
(`coperception/models/det/DiscoNet.py` return statement vs.
`coperception/utils/CoDetModule.py::step`). DiscoNet requires
`kd_flag=1`, which requires a pretrained `upperbound` teacher checkpoint
via `--resume_teacher` that does not exist yet. Not started.
