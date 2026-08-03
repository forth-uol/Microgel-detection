# Complete Guide: Windows → Barkla HPC → YOLO Training

This guide covers the complete workflow:

1. Install WinSCP and WindTerm.
2. Connect to the University of Liverpool Barkla HPC.
3. Upload your YOLO dataset.
4. Create a Conda environment.
5. Install PyTorch and Ultralytics.
6. Write a simple YOLO training script.
7. Write a SLURM job script.
8. Submit the SLURM job using WindTerm.
9. Check GPU usage using `node-usage.sh`.
10. Download the trained model using WinSCP.

Your university username is assumed to be:

```text
sgzjia25
```

The main project directory used in this guide is:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project
```

Your notes list the following Barkla servers:

```text
barklalogin1.liv.ac.uk
barklalogin2.liv.ac.uk
barklaviz1.liv.ac.uk
barklaviz2.liv.ac.uk
```

Use `barklalogin2.liv.ac.uk` for normal SSH login and file transfers. The `barklaviz` servers are mainly used for visualisation-related tasks.

---

# Step 1: Install WinSCP and WindTerm on Windows

## 1.1 Install WinSCP

WinSCP is used to:

* Upload datasets from Windows to Barkla.
* Upload Python and SLURM scripts.
* Download trained model files such as `best.pt`.
* Browse files on Barkla using a graphical interface.

Download WinSCP from its official website and install it using the default settings.
https://winscp.net/eng/download.php

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/14b9c3de-3741-4af3-adf7-3b36c3204274" />

## 1.2 Install WindTerm

WindTerm is used as the SSH terminal for connecting to Barkla.

Download the Windows x86-64 portable ZIP package from the official WindTerm GitHub Releases page.

https://github.com/kingToolbox/WindTerm/releases/tag/2.7.0

The file name will usually look similar to:

```text
WindTerm_x.x.x_Windows_Portable_x86_64.zip
```

Then:

1. Extract the ZIP file.
2. Open the extracted folder.
3. Double-click `WindTerm.exe`.

WindTerm is portable, so it normally does not require installation.

---

# Step 2: Connect to Barkla Using WindTerm

Open WindTerm.

Select:

```text
Session
→ New Session
→ SSH
```

Enter the following details:

```text
Host: barklalogin2.liv.ac.uk
Port: 22
```
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/2b30c784-1543-46e9-82b7-6fa95ba90c43" />

Enter school account name and password

Enter your University of Liverpool MWS password.

If the connection times out while you are outside the university network, connect to the University of Liverpool VPN first and try again.

## Check the connection

After logging in,

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/a4b48d5e-115c-4832-acf8-7211b8df7a9d" />

---

# Step 3: Create the Project Directories

In WindTerm, run:

```text
ls
cd fastscratch
mkdir example
ls
```

Using `fastscratch` is preferable for large datasets, model checkpoints and training outputs.

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/60bd31b8-8f2a-4fe0-965d-7e84dcdd87fe" />


# Step 4: Connect to Barkla Using WinSCP

Open WinSCP 

<img width="808" height="516" alt="image" src="https://github.com/user-attachments/assets/07ab27e5-7149-4f82-be5c-dc9b7eb2a9df" />


Enter:

```text
File protocol: SFTP
Host name: barklalogin2.liv.ac.uk
Port number: 22
User name: Your university name
Password: Your university password
```

Click:

```text
Save
Ok
Login
```

Accept the server fingerprint if prompted.

<img width="806" height="518" alt="image" src="https://github.com/user-attachments/assets/a2b3cfb3-faea-4e53-80d9-ab5608bea83a" />

After connecting:

* The left-hand side shows files on your Windows computer.
* The right-hand side shows files on Barkla.

You can now drag files between Windows and Barkla.

<img width="809" height="521" alt="image" src="https://github.com/user-attachments/assets/f2c6d699-a3a6-45e0-b384-36aaa14f3c7e" />

---

# Step 5: Prepare and Upload the YOLO Dataset

A standard YOLO object-detection dataset should use the following structure:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Each image should normally have a corresponding YOLO label file.

For example:

```text
images/train/image001.jpg
labels/train/image001.txt
```

The image and label must have the same base filename.

## Upload the dataset

Use WinSCP to upload your dataset

<img width="809" height="521" alt="image" src="https://github.com/user-attachments/assets/ee2b1b0d-15ff-44f3-9c24-dcf913ad68ef" />

## Create `data.yaml`

In WindTerm, run:

```bash
cd /mnt/fastscratch/users/sgzjia25/example/dataset

nano data.yalm
path: /mnt/fastscratch/users/sgzjia25/example/dataset

train: images/train
val: images/val

names:
  0: microgel
YAML
```
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/e852dc3b-128e-49b8-bc1a-dcf9362fd216" />

Press Ctrl+O（save), Enter, Ctrl+X(exit)

Display the file:

```bash
cat data.yaml
```
<img width="376" height="142" alt="image" src="https://github.com/user-attachments/assets/0119ebe0-d1f7-4a9f-99d5-b54ca38f04d0" />


## Step 6: Load the Conda Module on Barkla

The previous commands:

```bash
module load python/3.11.9-gcc14.2.0
source ~/yolo_project/yolo_env/bin/activate
```

activate a standard Python virtual environment rather than a Conda environment.

Do not load the old Python module or activate the old `yolo_env` virtual environment when using the new Conda environment.

First, search for the available Conda-related modules:

```bash
module --ignore-cache spider 2>&1 | grep -iE "anaconda|miniconda|miniforge|conda|python"
module spider miniforge3
```

On Barkla, the available Miniforge module includes:

```text
miniforge3/25.3.0-python3.12.10-dynamic
```

Load this module:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
```

Barkla may display an informational message explaining that this is a dynamic Miniforge installation. This is not an error.

Check that Conda has loaded correctly:

```bash
which conda
conda --version
```

The expected output is similar to:

```text
/opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10_dynamic/bin/conda
conda 25.7.0
```

If a valid Conda path and version are displayed, the Miniforge module has loaded successfully.

Do not run the following old commands:

```bash
module load python/3.11.9-gcc14.2.0
source ~/yolo_project/yolo_env/bin/activate
```

They belong to the old Python virtual environment and should not be mixed with the new Conda environment.


# Step 7: Create the Conda Environment

Create a directory for Conda environments:

```bash
mkdir -p /mnt/fastscratch/users/$USER/conda_envs
```

Initialise Conda in the current shell:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

Create the YOLO environment:

```bash
conda create \
    -p /mnt/fastscratch/users/$USER/conda_envs/yolo \
    python=3.11 \
    pip \
    -y
```

Activate it:

```bash
conda activate /mnt/fastscratch/users/$USER/conda_envs/yolo
```

Check the environment:

```bash
which python
python --version
which pip
```

The Python path should look similar to:

```text
/mnt/fastscratch/users/sgzjia25/conda_envs/yolo/bin/python
```

It should not be:

```text
/usr/bin/python
```

If it still shows `/usr/bin/python`, the Conda environment has not been activated correctly.

---

# Step 8: Install PyTorch and Ultralytics

Your previous Barkla setup used:

```bash
module load cuda/12.8.0-gcc14.2.0
```

Check whether that module is still available:

```bash
module spider cuda/12.8.0-gcc14.2.0
```

Load it:

```bash
module load cuda/12.8.0-gcc14.2.0
```

Initialise and activate Conda:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/fastscratch/users/$USER/conda_envs/yolo
```

Upgrade the installation tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA-enabled PyTorch build compatible with the CUDA environment used on Barkla.

For example:

```bash
python -m pip install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

Then install Ultralytics:

```bash
python -m pip install ultralytics
```

Check PyTorch:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda)"
```

Check Ultralytics:

```bash
python -c "from ultralytics import YOLO; print('Ultralytics import successful')"
```

## `CUDA available: False` on a login node

If you run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

on `barklalogin1` or `barklalogin2`, it may return:

```text
False
```

This is usually normal.

The login node has not allocated a training GPU to your session. PyTorch should only report:

```text
True
```

inside a SLURM job that has been allocated a GPU.

Do not run full YOLO training directly on the login node.

---

# Step 9: Write a Simple YOLO Training Script

Create the Python training script:

```bash
cd /mnt/fastscratch/users/sgzjia25/yolo_project

cat > train.py <<'PY'
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Cannot find data.yaml: {DATA_YAML}")

    model = YOLO("yolo26m.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1920,
        batch=4,
        device=0,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        project=str(RUNS_DIR),
        name="microgel_yolo26m",
        pretrained=True,
        val=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
PY
```

Display the script:

```bash
sed -n '1,200p' train.py
```

Check the Python syntax:

```bash
python -m py_compile train.py
```

If the command produces no output, the Python syntax is valid.

The model filename must exist in your installed Ultralytics version. If `yolo26m.pt` is not recognised, use a supported model or your own checkpoint, for example:

```python
model = YOLO("yolo11m.pt")
```

or:

```python
model = YOLO("/path/to/your/best.pt")
```

## Recommended first test

Before starting a long training run, test the entire workflow using a small model and one epoch.

Temporarily change:

```python
model = YOLO("yolo26n.pt")
```

and:

```python
epochs=1,
imgsz=640,
batch=2,
```

Once the complete job runs correctly, change the settings back to your intended values, such as:

```python
model = YOLO("yolo26m.pt")
epochs=100
imgsz=1920
batch=4
```

---

# Step 10: Check the Available GPU Partitions

Run:

```bash
sinfo -o "%20P %15a %20G %15l" | grep -i gpu
```

Your previous jobs used the partition:

```text
gpu-a-lowsmall
```

If that partition still appears in the output, you can use it in the SLURM script.

If it does not appear, use one of the currently available GPU partitions shown by `sinfo`.

---

# Step 11: Write the SLURM Script

Create the job script:

```bash
cd /mnt/fastscratch/users/sgzjia25/yolo_project

cat > train.slurm <<'SLURM'
#!/bin/bash -l

#SBATCH --job-name=yolo_microgel
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

#SBATCH --chdir=/mnt/fastscratch/users/sgzjia25/yolo_project
#SBATCH --output=/mnt/fastscratch/users/sgzjia25/yolo_project/logs/%x.%N.%j.out
#SBATCH --error=/mnt/fastscratch/users/sgzjia25/yolo_project/logs/%x.%N.%j.err

set -euo pipefail

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job name:     ${SLURM_JOB_NAME}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "========================================"

module purge

# Replace this with the exact Anaconda module available on Barkla.
module load Anaconda3

module load cuda/12.8.0-gcc14.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/fastscratch/users/sgzjia25/conda_envs/yolo

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Python executable:"
which python

echo "Python version:"
python --version

echo "PyTorch and GPU information:"
python -c "import torch; print('PyTorch:', torch.__version__); print('Built CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo "NVIDIA status:"
nvidia-smi

echo "Starting YOLO training..."
python -u train.py

echo "========================================"
echo "Finish time: $(date)"
echo "Training completed"
echo "========================================"
SLURM
```

Important: replace:

```bash
module load Anaconda3
```

with the exact Anaconda or Miniconda module name that worked in Step 6.

Check the SLURM script:

```bash
sed -n '1,240p' train.slurm
```

## Fix Windows line endings

If you edited or uploaded the scripts from Windows using WinSCP, run:

```bash
sed -i 's/\r$//' train.slurm train.py
```

This removes Windows CRLF line endings.

Otherwise, you may see an error such as:

```text
/bin/bash^M: bad interpreter
```

---

# Step 12: Submit the SLURM Job

Move into the project directory:

```bash
cd /mnt/fastscratch/users/sgzjia25/yolo_project
```

Submit the job:

```bash
sbatch train.slurm
```

A successful submission returns something similar to:

```text
Submitted batch job 123456
```

Here:

```text
123456
```

is the SLURM Job ID.

You can also save the Job ID automatically:

```bash
JOBID=$(sbatch --parsable train.slurm)
echo "Submitted job: $JOBID"
```

---

# Step 13: Check the Job Status

Run:

```bash
squeue -u $USER
```

Example output:

```text
JOBID  PARTITION       NAME             USER      ST  TIME  NODES  NODELIST(REASON)
123456 gpu-a-lowsmall  yolo_microgel    sgzjia25 PD  0:00  1      (Resources)
```

Common states are:

```text
PD = Pending
R  = Running
CG = Completing
```

For more detailed information:

```bash
squeue -j 123456 -o "%.18i %.18P %.25j %.8u %.8T %.10M %.10l %.20R"
```

Replace `123456` with your Job ID.

If the state is `PD`, the final column may show:

```text
(Resources)
```

or:

```text
(Priority)
```

This normally means that your job is waiting for a suitable GPU. It does not necessarily mean that the script has failed.

---

# Step 14: Find the GPU Node

You can only inspect GPU usage after the job state changes to:

```text
R
```

Run:

```bash
squeue -u $USER -t R -o "%.18i %.20j %.8T %.20N"
```

Example:

```text
JOBID              NAME                 STATE    NODELIST
123456             yolo_microgel        RUNNING  gpu133
```

The allocated node in this example is:

```text
gpu133
```

You can automatically retrieve the node name:

```bash
NODE=$(squeue -h -u "$USER" -t R -o "%N" | head -n 1)

echo "Running node: $NODE"
```

---

# Step 15: Check GPU Usage with `node-usage.sh`

Run:

```bash
node-usage.sh "$NODE"
```

For example:

```bash
node-usage.sh gpu133
```

Some versions of `node-usage.sh` may require only the numerical part of the node name.

If the previous command gives an argument error, run:

```bash
NODE_NUMBER="${NODE//[!0-9]/}"

echo "$NODE_NUMBER"

node-usage.sh "$NODE_NUMBER"
```

For example:

```bash
node-usage.sh 133
```

Your notes include node numbers such as `18`, `19`, `20`, `21`, `26`, `27`, `31`, `32`, `33`, `38`, `39`, `40`, `44`, `51`, `52`, `56`, `57`, `58`, `129`, `133`, `169`, `257`, `279` and `297`. However, you should not manually choose one of these nodes. Always use the node actually allocated to your job by SLURM.

To refresh the GPU information every five seconds:

```bash
watch -n 5 "node-usage.sh $NODE"
```

Press:

```text
Ctrl + C
```

to stop the monitoring command.

Important fields include:

```text
GPU utilisation
GPU memory usage
CPU utilisation
System memory usage
```

At the beginning of training, GPU utilisation may temporarily remain low while YOLO scans the dataset, creates cache files or initialises the model.

Once each epoch begins, GPU utilisation should normally increase.

---

# Step 16: View the Training Logs

List the log files:

```bash
cd /mnt/fastscratch/users/sgzjia25/yolo_project

ls -lh logs
```

The filenames may look similar to:

```text
yolo_microgel.gpu133.123456.out
yolo_microgel.gpu133.123456.err
```

View the standard output in real time:

```bash
tail -f logs/yolo_microgel.*.123456.out
```

Replace `123456` with your Job ID.

View the error log:

```bash
tail -f logs/yolo_microgel.*.123456.err
```

Press:

```text
Ctrl + C
```

to exit `tail -f`.

Search the logs for common errors:

```bash
grep -iE "error|exception|traceback|cuda out of memory" logs/*123456*
```

---

# Step 17: Confirm That PyTorch Is Using the GPU

At the beginning of the job log, you should see something similar to:

```text
CUDA available: True
GPU: NVIDIA ...
```

If the SLURM job shows:

```text
CUDA available: False
```

then PyTorch is not using the allocated GPU.

Check the following parts of the SLURM script:

```bash
module load cuda/12.8.0-gcc14.2.0
```

Check the active Python environment:

```bash
which python
```

It should point to:

```text
/mnt/fastscratch/users/sgzjia25/conda_envs/yolo/bin/python
```

Check PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Inside a correctly allocated GPU job, the final value should be:

```text
True
```

---

# Step 18: Stop a Training Job

To cancel one job:

```bash
scancel 123456
```

Replace `123456` with the actual Job ID.

Then check:

```bash
squeue -u $USER
```

To cancel all your running and pending jobs:

```bash
scancel -u $USER
```

Be careful with this command because it cancels every job owned by your account.

---

# Step 19: Find the YOLO Training Results

The training output should be stored in:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/runs/microgel_yolo26m
```

List the output files:

```bash
find runs/microgel_yolo26m -maxdepth 2 -type f | sort
```

The trained weights should be located at:

```text
runs/microgel_yolo26m/weights/best.pt
runs/microgel_yolo26m/weights/last.pt
```

Check the files:

```bash
ls -lh runs/microgel_yolo26m/weights
```

The difference is:

```text
best.pt   Model with the best validation performance
last.pt   Model saved after the final training epoch
```

Other useful output files may include:

```text
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
```

---

# Step 20: Download `best.pt` Using WinSCP

Open WinSCP and connect to Barkla.

On the right-hand Barkla side, open:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/runs/microgel_yolo26m/weights
```

Drag:

```text
best.pt
```

from the Barkla side to a folder on the Windows side.

You may also download:

```text
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
```

---

# Commands You Will Use Most Often

## Enter the project directory

```bash
cd /mnt/fastscratch/users/sgzjia25/yolo_project
```

## Load the environment

```bash
module purge

module load Anaconda3
module load cuda/12.8.0-gcc14.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/fastscratch/users/sgzjia25/conda_envs/yolo
```

Replace `module load Anaconda3` with the actual module available on Barkla.

## Submit training

```bash
sbatch train.slurm
```

## Check your jobs

```bash
squeue -u $USER
```

## Check the allocated node

```bash
squeue -u $USER -t R -o "%.18i %.20j %.8T %.20N"
```

## Check GPU usage

```bash
NODE=$(squeue -h -u "$USER" -t R -o "%N" | head -n 1)
node-usage.sh "$NODE"
```

## View logs

```bash
ls -lh logs
tail -f logs/*.out
```

## Cancel a job

```bash
scancel JOB_ID
```

## Check the trained model

```bash
ls -lh runs/microgel_yolo26m/weights
```

---

# Important Rules

Do not run full training directly on:

```text
barklalogin1
barklalogin2
```

The login nodes should only be used for:

* Uploading and organising files.
* Creating environments.
* Installing packages.
* Editing scripts.
* Submitting jobs.
* Checking job status and logs.

Start the actual GPU training through SLURM:

```bash
sbatch train.slurm
```

Do not start the full training with:

```bash
python train.py
```

directly on the login node.

The normal workflow is:

```text
WindTerm login
→ upload dataset using WinSCP
→ create Conda environment
→ write train.py
→ write train.slurm
→ submit with sbatch
→ check with squeue
→ inspect the allocated GPU using node-usage.sh
→ monitor the logs
→ download best.pt using WinSCP
```



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



