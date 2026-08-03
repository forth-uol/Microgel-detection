# LivSURF Microgel Detection Project | Handover Guide

Last verified: 2 August 2026  
Project directory: `E:\Livesurf\Total`  

## 1. Start Here

This project uses Ultralytics YOLO26m to detect microgels in microscopy images. The aim is to produce detection boxes that can support particle counting and, later, diameter measurement.

The model used in the current presentation is **Exp16**:

- Weights: `Exp16/weights/best.pt`
- Dataset: `microgel_dataset_clean/data_new.yaml`
- Image size: 1920
- Thresholds used in the presentation: `conf=0.50`, `NMS IoU=0.80`
- Validation metrics: Precision 0.9580, Recall 0.9180, F1 0.9376, mAP50 0.9766, mAP50-95 0.8732
- Prediction examples: `runs/detect/Exp16_prediction`
- Scope of selection: Exp16 is the current preferred model **within the Exp11–Exp16 stage**. It is not the model with the highest mAP50-95 across all historical single-class experiments.

Exp5 and Exp6 achieved higher metrics in the historical single-class route. Their weights are under `E:\Livesurf\Now_model`, outside this directory:

| Model | Weight location | conf | IoU | mAP50-95 | Notes |
|---|---|---:|---:|---:|---|
| Exp5 | `E:\Livesurf\Now_model\Exp5\weights\best.pt` | 0.50 | 0.80 | 0.9080 | Strong historical single-class baseline |
| Exp6 | `E:\Livesurf\Now_model\Exp6\weights\best.pt` | 0.45 | 0.80 | 0.9077 | Strong baseline without mosaic or flipping |
| Exp16 | `Exp16\weights\best.pt` | 0.50 | 0.80 | 0.8732 | Current presentation model; balanced-fitness route |

Do not choose the final model from one metric alone. The practical objective is “one suitable box per particle”, so missed detections, reflection-related false positives, duplicate boxes, edge fit, counting error and diameter error must also be checked.

## 2. Recommended Handover Order

1. Read this README first.
2. Open `EXP_yolo_latest.xlsx`, focusing on `Sheet1` and `Rename Map`.
3. Inspect `Exp16/weights/best.pt` and `runs/detect/Exp16_prediction`.
4. Read `threshold_search_results/best_thresholds_summary.csv`, then inspect individual model CSV files as needed.
5. Confirm the Python, PyTorch, CUDA and Ultralytics versions before running any training or threshold search.
6. Reproduce the Exp16 validation once before attempting to retrain all models.
7. Prioritise count-error and diameter-error validation on an independent test set.
8. Complete Section 11 after the HPC/HTC environment has been inspected directly.

## 3. Directory Map

| Path | Contents | Importance |
|---|---|---|
| `EXP_yolo_latest.xlsx` | Experiment index, metrics, code fragments and rename records for Exp1–Exp19 | Critical experiment register |
| `Exp1`–`Exp4`, `Exp9`, `Exp10` | Standard training outputs: arguments, curves, result CSVs, validation images and weights | Important historical experiments |
| `Exp13`–`Exp16` | Hyperparameter-tuning outputs: best parameters, tuning records, plots and weights | Important later experiments |
| `microgel_dataset_clean` | Current 1920×1200 single-class main dataset | Critical |
| `microgel_dataset_clean_640` | Derived 640×640 dataset sliced from the main dataset | Derived and rebuildable |
| `threshold_search_results` | Per-model confidence/NMS-IoU searches and combined summary | Critical |
| `runs/detect` | Prediction images and prediction TXT files from different models | Important visual-validation evidence |
| `images`, `labels` | Earlier 59/10-image data version | Historical; not the current default dataset |
| `microgel_dataset_clean_1920` | Older training run not mapped into the current Exp numbering | Historical; do not delete yet |
| `.codex_exp16_update` | Inspection, preview and QA intermediates created while updating the presentation | Not a training input; rebuildable |
| `LivSURF_Microgel_Detection_Presentation_10min_Exp16_updated.pptx` | Current 10-minute project presentation | Important presentation material |
| `LivSURF_Microgel_Detection_Presentation_10min_Exp16_updated.pdf` | PDF version of the presentation | Important for distribution |

The `Total` directory currently contains approximately 2,338 files and occupies about 1.43 GB. Most of the space is used by model weights, datasets and prediction images.

## 4. Datasets

### 4.1 Current Main Dataset

Configuration: `microgel_dataset_clean/data_new.yaml`

| Split | Images | Label files | Microgel boxes | Pairing status |
|---|---:|---:|---:|---|
| train | 63 | 63 | 10,161 | Complete |
| val | 18 | 18 | 2,511 | Complete |
| Total | 81 | 81 | 12,672 | Complete |

The dataset contains only class `0: microgel`.

Important: `microgel_dataset_clean/dataset_summary.txt` reports 14 validation images and 2,075 validation boxes. Those figures are outdated. The current directory contains **18 validation images and 2,511 validation boxes**. Future work should use the actual files or a fresh recount as the source of truth, and the summary should be updated whenever the dataset changes.

The source note states that the original annotations came from JSON files under `E:\Livesurf\TEST_yolo_bright\images\test`, and that YOLO TXT labels were generated from the JSON rectangle coordinates. Matching JSON files are also retained in `microgel_dataset_clean/images`; training uses the JPG files and the TXT labels under `labels`.

### 4.2 Derived 640 Dataset

Configuration: `microgel_dataset_clean_640/data_new.yaml`

| Split | 640 tiles | Label files | Written boxes |
|---|---:|---:|---:|
| train | 378 | 378 | 8,435 |
| val | 108 | 108 | 1,870 |

Generation script: `slice_yolo_640.py`

Slicing rules:

- A box is kept only if it lies fully inside a tile and does not touch the source-image boundary.
- A particle crossing a tile boundary may remain visible in the image but will not be labelled.
- Output is lossless PNG with no resizing or enhancement.
- The nominal tile size and stride are both 640. Because the source image height is 1200, the final row starts at `y=560`, creating an actual vertical overlap of 80 px.
- In the train split, 854 eligible source boxes are not fully covered by any tile; the equivalent number for val is 295. The 640 dataset is therefore not a lossless equivalent of the full-resolution dataset.

Use a new output directory when rebuilding. The script intentionally stops if the target already exists:

```powershell
python slice_yolo_640.py --source microgel_dataset_clean --output microgel_dataset_clean_640_v2
```

## 5. How to Read the Experiment Records

### 5.1 Master Register

`EXP_yolo_latest.xlsx` is the most complete experiment index currently available:

- `Sheet1`: parameters, metrics, paths, code fragments and notes for Exp1–Exp19.
- `Rename Map`: mapping from historical directory names to Exp numbers.
- Exp17, Exp18 and Exp19 are still planned experiments. They have no completed local results and must not be reported as findings.

After a directory has been renamed, do not infer its experiment number from an old name. Check `Rename Map` first.

### 5.2 Types of Completed Experiments

- Exp1–Exp4, Exp9 and Exp10 are primarily full training runs. Their per-epoch results are in `results.csv`.
- Exp13–Exp16 are primarily `model.tune()` runs. Their metrics and selected parameters are in `best_hyperparameters.yaml` and `tune_results.ndjson`; they do not contain the standard training `results.csv`.
- Exp7 and Exp8 are threshold/prediction checks using the Exp5 and Exp6 weights, not new training weights.
- Exp11 has local prediction output, but its model weights were not copied into the current workspace.
- The Exp12 model and two-class dataset are under `E:\Livesurf\2_labels`, outside `Total`.

### 5.3 Missing and Historical Directories Requiring Care

- `Exp1/weights` and `Exp2/weights` are empty; only the training records and plots remain.
- The local Exp11 model directory is missing; the threshold CSV retains its cluster path.
- `microgel_dataset_clean_1920` is an old training directory that has not been mapped to the current Exp register. Its best mAP50-95 is approximately 0.9014. Do not delete or rename it until its identity has been confirmed.
- Exp5 and Exp6 weights are under `E:\Livesurf\Now_model`.
- The older `EXP_yolo.xlsx` remains useful as historical reference, but new conclusions should be recorded in `EXP_yolo_latest.xlsx`.

## 6. Exp16: Current Presentation Model

### 6.1 Why It Was Selected

Exp16 uses balanced fitness:

```text
fitness = 0.15 × Precision
        + 0.30 × Recall
        + 0.20 × mAP50
        + 0.35 × mAP50-95
```

It produced the strongest current result in the later Exp11–Exp16 two-class/weighted-objective series and was visually reviewed across several validation-image conditions. The final presentation describes it as the current front-runner while explicitly retaining the limitation that an independent test set has not yet been completed.

The Exp16 tuning record and the later threshold evaluation are different evaluations; their figures must not be mixed:

- In `Exp16/best_hyperparameters.yaml`, the best tuning fitness occurred at iteration 80, with P 0.9366, R 0.7873, mAP50 0.8238 and mAP50-95 0.7270.
- The final threshold evaluation in `threshold_search_results/Exp16_threshold_search.csv` and the combined summary reports P 0.9580, R 0.9180, mAP50 0.9766 and mAP50-95 0.8732.

### 6.2 Threshold Selection

The presentation and combined summary use:

```text
conf = 0.50
NMS IoU = 0.80
```

The default threshold-search logic first selects the NMS IoU by mAP50-95, then uses the one-box score, F1 and duplicate rate to choose a deployable confidence threshold.

If the application objective changes to prioritise F1, the Exp16 detail CSV shows that `conf=0.45, IoU=0.50` gives an F1 of approximately 0.9485 and a duplicate-particle rate of 0 for that row. This is not the threshold reported in the current presentation. Count and diameter errors must be compared again on an independent test set before adopting it.

### 6.3 Rechecking Predictions

`predict.py` is currently hard-coded for **Exp9**, not Exp16. Before rechecking Exp16, change at least:

```python
MODEL_PATH = Path(r"E:\Livesurf\Total\Exp16\weights\best.pt")
CONF = 0.50
NMS_IOU = 0.80
```

Also change the output `name` to a new value such as `Exp16_recheck_YYYYMMDD` so that the existing results are not confused with the recheck. Then run:

```powershell
python predict.py
```

The review should include at least:

- whether each box follows the particle boundary;
- whether visible particles are missed;
- whether reflections cause false positives;
- whether a particle receives duplicate boxes;
- whether edge particles and overlapping particles are handled consistently;
- whether the boxes are suitable for later count and diameter estimation.

## 7. Threshold Search

Main files:

- `find_best_yolo_thresholds_one_box.py`: single-model search.
- `find_best_yolo_thresholds_all_models.py`: recursively discovers multiple `*/weights/best.pt` files, searches each model and writes a combined summary.
- `run_threshold_search_all.slurm`: cluster batch entry point.

The search calculates:

- Precision, Recall and F1;
- mAP50 and mAP50-95;
- one-box rate;
- duplicate-particle rate and duplicate-box rate;
- `selection_score = F1 - 0.25 × duplicate_particle_rate`.

First list the models that will be discovered:

```powershell
python find_best_yolo_thresholds_all_models.py --model-root . --data microgel_dataset_clean/data_new.yaml --list-models
```

To recheck only Exp16 and write to a new result directory:

```powershell
python find_best_yolo_thresholds_all_models.py `
  --model-root . `
  --model Exp16 `
  --data microgel_dataset_clean/data_new.yaml `
  --split val `
  --imgsz 1920 `
  --conf 0.05:0.80:0.05 `
  --iou 0.30:0.90:0.05 `
  --match-iou 0.50 `
  --map-conf 0.001 `
  --max-det 1000 `
  --device 0 `
  --batch 1 `
  --selection-metric map50_95 `
  --duplicate-weight 0.25 `
  --output-dir threshold_search_results_recheck
```

Important notes:

- YOLO26 prediction must use the one-to-many head. The scripts set `end2end=False`; otherwise the conventional NMS IoU may not take effect.
- `--resume` skips a run only when the settings match exactly, the summary status is `completed`, and the detail CSV path still exists. Many current summary rows contain absolute cluster paths, so a local run may not match the resume condition.
- Do not overwrite `threshold_search_results` directly. Write rechecks to a new directory, compare the results, and update the official summary only after verification.
- A model may appear under similar entries because of old directory names, renamed directories and external directories. Compare `model_path` as well as the model name.

Run on the cluster with:

```bash
sbatch run_threshold_search_all.slurm
```

Before submission, correct the working directory, virtual environment, data path, partition and GPU configuration at the top of the script.

## 8. Training and Tuning Scripts

| File | Purpose | Current status |
|---|---|---|
| `YOLO_train.py` | Dual tune/train mode for the 1920 reflection-background route | Inspect before use |
| `YOLO_train_640.py` | Tune/train on the 640 dataset | Inspect before use |
| `YOLO_train_mosaic_flip0.py` | Tune/train with mosaic and flips disabled | Useful reference for the Exp6 route |
| `yolo_total.slurm` | Runs three augmentation experiments sequentially | Cluster paths are hard-coded |
| `yolo_train.slurm` | Calls `YOLO_train.py` and checks the custom-fitness patch | Required dependency is missing locally |

Known issues:

- `YOLO_train.py` contains `hsc_h`, which is probably intended to be `hsv_h`; the script also passes `hsv_h=0.0`. Confirm and correct this before submitting a long job.
- `yolo_train.slurm` explicitly requires `sitecustomize.py`, but that file is not present in the `Total` root.
- `YOLO_train.py` and `YOLO_train_640.py` require `yolo26m.pt` in the root directory, which is not present locally.
- `yolo_total.slurm` uses `yolo26m.pt` / `yolo26m.yaml`; these base files are also absent from the local `Total` root.
- The complete Exp16 custom-fitness code is currently stored mainly in the Exp16 row of `EXP_yolo_latest.xlsx`, rather than as an independent `.py` file. It should be extracted into a versioned script and its dependency versions recorded.
- `txt_josn.py` is an older one-file converter from YOLO TXT to X-AnyLabeling JSON. Its input path, image dimensions and file name are hard-coded and must be edited before use. Its own file name also contains a spelling mistake.

## 9. Local Environment Check

This directory has no `requirements.txt`, `environment.yml` or complete version-lock file. Do not upgrade Ultralytics, PyTorch or CUDA before recording the current versions; YOLO26 parameter behaviour and the `DetMetrics.fitness` patch may change between versions.

Run these checks first:

```powershell
python --version
python -c "import torch, ultralytics, PIL, yaml; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
```

Environment information that still needs to be recorded by the next person:

- Python version;
- Ultralytics version;
- PyTorch and CUDA versions;
- GPU model;
- `pip freeze` or a reduced requirements file;
- full environment-creation commands;
- whether local and cluster environments use the same versions.

## 10. Current Scientific Conclusions and Next Steps

Conclusions currently supported by the evidence:

- The single-class YOLO route can achieve strong detection metrics on the current validation set.
- Exp16 is the current preferred model in the later balanced-fitness series and has been visually checked on several real validation-image conditions.
- The main problem in the early two-class experiments was extreme class imbalance: approximately 10,711 microgel boxes versus 56 notmicrogel boxes. This does not demonstrate that the second-class concept is invalid.
- The Pix2Pix-Turbo synthetic-data route can transfer overall geometry and appearance, but small and overlapping particles can still resemble repeated concentric-ring templates. It has not yet been shown to improve YOLO performance.

Priorities for the next stage:

1. Build a test set completely independent of the current validation set.
2. Measure particle-count error and diameter error directly, rather than relying only on detection mAP.
3. Collect substantially more reflection/notmicrogel labels to reduce the extreme two-class imbalance.
4. Complete the planned Exp17–Exp19 comparisons, with a clear hypothesis, one controlled variable and a stopping criterion for each experiment.
5. Validate synthetic-image quality and label correctness independently before mixing synthetic images into YOLO training.
6. Extract the Exp16 custom-fitness code from Excel into an independent, versioned script.
7. Compare Exp5, Exp6 and Exp16 on one fixed independent test set instead of comparing figures across different experiment stages.

## 11. HPC/HTC Environment (Complete After Direct Inspection)

Existing files indicate that the project uses the Liverpool Barkla cluster:

- Login nodes: `barklalogin1.liv.ac.uk`, `barklalogin2.liv.ac.uk`
- Visualisation nodes: `barklaviz1.liv.ac.uk`, `barklaviz2.liv.ac.uk`
- Common working directory: `/mnt/fastscratch/users/sgzjia25/yolo_train`
- Common virtual environment: `/mnt/fastscratch/users/sgzjia25/yolo_env`
- Training scripts have used `gpu-a-lowsmall`
- The threshold-search script has used `cpu-l40s-low` while requesting one GPU
- Example CUDA module: `cuda/12.8.0-gcc14.2.0`

These details come from existing scripts and should be treated only as leads. The following must be confirmed during the cluster review:

- the actual login method, MFA/VPN requirements and account permissions;
- currently available partitions, GPU types, time limits and memory limits;
- exact Python and CUDA module versions;
- whether the virtual environment still exists and can be activated;
- which files exist under `yolo_train` but are missing locally;
- the real locations of the datasets, base weights, `sitecustomize.py` and result directories;
- standard procedures for `sbatch`, `squeue`, `sacct`, cancelling jobs and copying results;
- how Windows paths and the experiment register should be updated after results are synchronised from the cluster.

## 12. Recording Standards for New Experiments

Every new experiment should:

1. Use a unique new directory and never overwrite an old result.
2. Save the run script, Slurm file, data YAML, full arguments and environment versions.
3. Record how the model was initialised: pretrained, from scratch, or fine-tuned from a specific checkpoint.
4. Record the dataset version and the train/val/test file lists or hashes.
5. Preserve `best.pt`, `last.pt`, training curves and representative predictions.
6. Run the standard threshold search and record the selection metric.
7. Perform visual checking on a fixed image set.
8. Add a new row to `EXP_yolo_latest.xlsx` without reusing an old Exp number.
9. If a directory is renamed, update `Rename Map`, script paths, the threshold summary and the README together.
10. State clearly whether the outcome is “completed”, “failed”, “uncertain” or “planned”.

Suggested naming pattern:

```text
ExpNN_<short-purpose>_<YYYYMMDD>
```

## 13. Backup and Cleanup

Prioritise backups of:

- `EXP_yolo_latest.xlsx`
- `Exp*/weights/best.pt`
- `Exp*/best_hyperparameters.yaml`, `args.yaml`, `results.csv`, `tune_results.ndjson`
- `microgel_dataset_clean`
- `threshold_search_results`
- `runs/detect/Exp16_prediction`
- final PPTX/PDF files
- the environment records and independent test set added later

Do not delete these items before their origin and purpose have been confirmed:

- the unmapped `microgel_dataset_clean_1920` directory;
- Exp1/Exp2 training records, even though their weights are missing;
- Exp5, Exp6 and Exp12 assets referenced by `Rename Map` but located outside `Total`;
- existing threshold CSVs, because they retain historical model paths and evaluation settings.

Rebuildable content generally includes previews, layout-inspection files and some presentation intermediates under `.codex_exp16_update`. Even so, make a complete directory backup before cleanup.



