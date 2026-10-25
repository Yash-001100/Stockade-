# Stockade — Phase 1 setup

Quick note first: I couldn't run any of this for you directly on your PC this time —
the remote shell Claude uses to work on your machine is currently broken because of a
Windows update from Sept 8 that Anthropic is already tracking. Nothing wrong on your
end, just a temporary limitation. Everything below is written so you can run it
yourself in a normal PowerShell/Command Prompt window; once the fix lands I'll be able
to run this directly for you.

## 1. Extract your datasets

Right now your `Stockade` folder has these still zipped:

- `Dachcam_dataset.zip` and `Normal-002.zip` — almost certainly your CCD (Car Crash
  Dataset) download (Crash-1500 + Normal clips). Extract both.
- `v1.0-mini.tgz` — the nuScenes mini split. Extract it (7-Zip handles .tgz; Windows'
  built-in "Extract All" may not).
- `MoonBlvd Detection-of-Traffic-Anomaly master dataset.zip` — this is just the DoTA
  GitHub repo's code, not the actual dataset (as we found earlier) — you can leave this
  zipped or skip it for now.
- `example-cooperative-vehicle-infrastructure.zip`, `V2X-Seq-SPD-Example.zip`,
  `V2X-Seq-TFD-Example.zip`, `V2X-Sim-det-001.zip` — these are all Phase 8 (networked
  version) material, not needed for anything you're doing right now. They're also
  eating ~11GB of disk space combined — fine to leave zipped, or even delete and
  re-download later if you're tight on space.

Extract the CCD and nuScenes ones into a clear structure, e.g.:

```
Stockade\
  data\
    CCD\            <- contents of Dachcam_dataset.zip + Normal-002.zip
    nuscenes-mini\  <- contents of v1.0-mini.tgz
  track_vehicles.py
  requirements.txt
```

It doesn't matter exactly how the CCD zips' internal folders are named — the script
below searches recursively for `.mp4` files, so as long as the clips end up somewhere
under `Stockade\data\CCD`, it'll find them.

## 2. Set up Python

Open PowerShell in your `Stockade` folder (Shift+Right-click the folder → "Open
PowerShell window here", or `cd` into it) and run:

```powershell
python -m venv stockade-env
stockade-env\Scripts\activate
pip install -r requirements.txt
```

If `python` isn't recognized, you'll need to install Python 3.10+ from python.org
first (check "Add python.exe to PATH" during install).

## 3. Run Phase 1

Start small — process just 3 clips first to make sure everything works before running
it over hundreds:

```powershell
python track_vehicles.py --input data\CCD --output outputs --limit 3
```

The first run will also download the small YOLOv8 model weights (~6MB) automatically.

Check the `outputs` folder for:
- An annotated video per clip, with tracked boxes and IDs drawn on — watch a couple of
  these to see whether IDs stay stable across the clip or switch/flicker.
- A `<clip-name>_tracks.csv` per clip — this is the trajectory data Phase 2 will use.

Once that looks right, drop `--limit 3` and let it run across everything:

```powershell
python track_vehicles.py --input data\CCD --output outputs
```

This will take a while over the full CCD set (thousands of clips) — that's expected
and fine to leave running in the background.

## 4. What "done" looks like for Phase 1

Per the roadmap: a handful of annotated clips where tracked vehicles keep a stable ID
across the whole clip, and CSVs with clean per-frame trajectories. Once you have that,
you're ready for Phase 2 — turning those trajectories into risk-score features
(lateral position, time-to-collision, following-distance volatility).
