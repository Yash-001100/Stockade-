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
checkpoints.

### LR-schedule experiment (Experiment A)
The epoch-100 run's scheduler (`MultiStepLR`, milestones=[50,100,150,200])
was sized for 200 epochs and only ever passed one decay step by epoch 100 —
tests whether the epoch-33 plateau was an LR artifact. Resumed from
`logs/lowerbound/no_rsu/epoch_100.pth` (model weights only) with a fresh
optimizer, Adam lr=0.0005, milestones=(15,30), gamma=0.5, 40 epochs, via
`finetune_fafnet_lr_experiment.py` -> `logs_lr_experiment/`.

| checkpoint | mAP@0.5 | mAP@0.7 |
|---|---|---|
| epoch 10 | 64.6% | 61.2% |
| epoch 20 | 65.0% | 61.5% |
| epoch 30 | 64.9% | 61.3% |
| epoch 40 | 65.5% | 61.7% |

Per-agent (epoch 40, mAP@0.5/mAP@0.7): agent1 62.7%/58.1%, agent2
65.4%/62.5%, agent3 70.8%/66.9%, agent4 65.8%/61.9%, agent5 59.9%/55.9%.
Modest but real gain over epoch 100 (64.1% -> 65.5% mAP@0.5) — the
original schedule was undertrained for its own milestones, not fully
converged. Agent5 remains weakest by a consistent ~5-10pp margin across
every checkpoint and every experiment in this project.

### Agent-5 diagnosis (`diagnose_agent5.py`)
Ran directly on preprocessed ground truth (no model inference) across the
full val split (1000 scene-frames/agent) and train split (8000/agent) to
test two hypotheses: (a) agent5's scenes are structurally harder
(farther/sparser objects), or (b) agent5 just has less real training
data. Measured per agent: objects/frame, mean/median distance from ego,
%objects >20m, occupied-area fraction, and `trans_matrices` orthonormality
(coordinate-pipeline sanity check).

Result: distance/density statistics are essentially identical across all
5 agents once measured on the full dataset (an earlier small-sample run,
max_samples=400, misleadingly suggested agent5 had farther/harder scenes
— that didn't replicate at full scale). `trans_matrices` are clean
(no orthonormality violations). The real driver is data scarcity: only
21.2% of agent5's training scene-frames have real (non-placeholder)
sensor data, vs. 48.8% for agent4 and ~100% for agents 1-3. Agent5 is a
genuinely vehicle-sparse position in the V2X-Sim scenes, not a pipeline
bug or an inherently harder viewpoint.

### Agent-4/5 oversampling follow-up — negative result
Hypothesis: if agent5's weakness is data scarcity, oversampling the
scene-frames where it has real data (via `WeightedRandomSampler`, weights
3.0x agent5-real / 1.5x agent4-real / 1.0x otherwise, built by
`build_agent_sample_weights.py`) should let the model extract more signal
per epoch. Resumed from the LR experiment's epoch 40 checkpoint, same LR
(0.000125, no further decay — isolating the sampler as the only new
variable), 20 epochs, via `train_fafnet_agent_reweight.py` ->
`logs_agent_reweight/`.

| agent | baseline (LR-exp ep.40) | reweight ep.10 | reweight ep.20 |
|---|---|---|---|
| agent1 | 62.7% / 58.1% | 62.9% / 58.2% | 62.8% / 58.0% |
| agent2 | 65.4% / 62.5% | 65.4% / 62.6% | 65.2% / 62.5% |
| agent3 | 70.8% / 66.9% | 70.8% / 66.9% | 70.7% / 66.8% |
| agent4 | 65.8% / 61.9% | 65.8% / 62.0% | 65.6% / 61.8% |
| agent5 | 59.9% / 55.9% | 59.9% / 56.0% | 59.6% / 55.9% |
| avg | 65.5% / 61.7% | 65.6% / 61.8% | 65.4% / 61.6% |

No improvement anywhere — every agent flat to within ~0.3pp, with a
slight overall dip by epoch 20. Oversampling increases how often the
model *revisits* the same ~1,696 unique real agent5 scenes per epoch; it
doesn't add new unique data, and the model had already extracted what
those scenes teach it by epoch 40. This reinforces the diagnosis above:
agent5's ceiling is a genuine unique-data-volume limit, not a
sampling-frequency problem. A real fix needs either more unique real
agent5 data (not available in this dataset) or a model that lets agent5
borrow signal from neighboring agents in the same scene — which is what
cooperative fusion (DiscoNet) is designed to do, unlike `lowerbound`'s
per-agent-isolated training.

### DiscoNet — blocked, not trained
Smoke-tested only. `DiscoNet.forward()` with `kd_flag=0` returns
`(result, save_agent_weight_list)`, but `CoDetModule.FaFModule.step()`
only unpacks that shape when `kd_flag=1` — confirmed from source
(`coperception/models/det/DiscoNet.py` return statement vs.
`coperception/utils/CoDetModule.py::step`). DiscoNet requires
`kd_flag=1`, which requires a pretrained `upperbound` teacher checkpoint
via `--resume_teacher` that does not exist yet. Not started.
