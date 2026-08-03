# Complete Guide: Windows → Barkla HPC → YOLO Training

Last checked: 3 August 2026

This guide covers the complete workflow:

1. Install WinSCP and WindTerm on Windows.
2. Connect to the University of Liverpool Barkla HPC.
3. Create a project directory on `fastscratch`.
4. Upload a YOLO object-detection dataset.
5. Create a Conda environment.
6. Install PyTorch and Ultralytics.
7. Download the pretrained YOLO26m weights before requesting a GPU.
8. Create the Python training program and Slurm job script.
9. Run a short test job, then submit production training.
10. Monitor the job, inspect GPU usage and logs, resume interrupted training, and download the results.

The examples assume the following University username and project directory:

```text
Username: sgzjia25
Project:  /mnt/fastscratch/users/sgzjia25/yolo_project
```

Commands that use `$USER` automatically substitute the currently logged-in username. The `#SBATCH` paths in the Slurm script use the full username because Slurm does not expand shell variables in `#SBATCH` directives.

The Barkla hostnames listed in the existing project notes are:

```text
barklalogin1.liv.ac.uk
barklalogin2.liv.ac.uk
barklaviz1.liv.ac.uk
barklaviz2.liv.ac.uk
```

This guide uses `barklalogin2.liv.ac.uk` as the example login and SFTP host. If it is unavailable, use the other login node according to the current University guidance. The `barklaviz` nodes are intended mainly for visualisation work.

> **Important:** Partition names, module versions, GPU types, resource limits and login requirements are controlled by Barkla administrators and can change. Commands later in this guide show how to check the live configuration before submitting a job.

---

## Step 1: Install WinSCP and WindTerm on Windows

### 1.1 Install WinSCP

WinSCP provides a graphical interface for:

- uploading datasets, Python files and Slurm scripts;
- downloading trained weights such as `best.pt`;
- browsing files on Barkla.

Download and install WinSCP from the [official WinSCP download page](https://winscp.net/eng/download.php). The default installation settings are normally suitable.

![WinSCP download page](https://github.com/user-attachments/assets/14b9c3de-3741-4af3-adf7-3b36c3204274)

### 1.2 Install WindTerm

WindTerm is used as the SSH terminal for Barkla.

Download the latest stable Windows x86-64 portable ZIP package from the [official WindTerm releases page](https://github.com/kingToolbox/WindTerm/releases). Its filename is normally similar to:

```text
WindTerm_x.x.x_Windows_Portable_x86_64.zip
```

Then:

1. Extract the ZIP archive.
2. Open the extracted directory.
3. Double-click `WindTerm.exe`.

WindTerm is portable and normally does not require a separate installation.

---

## Step 2: Connect to Barkla Using WindTerm

Open WindTerm and select:

```text
Session → New Session → SSH
```

Enter:

```text
Host: barklalogin2.liv.ac.uk
Port: 22
```

![Creating a Barkla SSH session in WindTerm](https://github.com/user-attachments/assets/2b30c784-1543-46e9-82b7-6fa95ba90c43)

Enter your University username and MWS password when prompted. Complete any current University MFA requirement.

If the connection times out outside the University network, connect to the University of Liverpool VPN and try again.

The first connection may display a server-host-key prompt. Check the fingerprint against current University guidance before accepting it.

After login, the terminal should show a Barkla shell prompt:

![Successful Barkla login](https://github.com/user-attachments/assets/a4b48d5e-115c-4832-acf8-7211b8df7a9d)

Run these lightweight checks:

```bash
whoami
hostname
pwd
```

Do not run model training directly on a login node. Login nodes are for file management, environment setup, lightweight checks and Slurm job submission.

---

## Step 3: Create the Project Directories

In WindTerm, define the project paths and create the required directories:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

mkdir -p "$PROJECT/dataset"
mkdir -p "$PROJECT/logs"
mkdir -p "$PROJECT/weights"

cd "$PROJECT"
pwd
```

For the example account, `pwd` should display:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project
```

![Creating a fastscratch project directory](https://github.com/user-attachments/assets/60bd31b8-8f2a-4fe0-965d-7e84dcdd87fe)

`fastscratch` is suitable for large datasets, checkpoints and training outputs, but it must not be treated as the only long-term backup location.

---

## Step 4: Connect to Barkla Using WinSCP

Open WinSCP:

![WinSCP start screen](https://github.com/user-attachments/assets/07ab27e5-7149-4f82-be5c-dc9b7eb2a9df)

Enter:

```text
File protocol: SFTP
Host name:     barklalogin2.liv.ac.uk
Port number:   22
User name:     your University username
Password:      your University password
```

Select `Save` if you want to retain the connection profile, then select `Login`. Avoid storing the password on a shared computer.

If WinSCP displays a host-key prompt, verify the fingerprint before accepting it.

![WinSCP login settings](https://github.com/user-attachments/assets/a2b3cfb3-faea-4e53-80d9-ab5608bea83a)

After connecting:

- the left pane shows files on Windows;
- the right pane shows files on Barkla.

Files can now be dragged between the two systems.

![WinSCP local and remote panes](https://github.com/user-attachments/assets/f2c6d699-a3a6-45e0-b384-36aaa14f3c7e)

On the Barkla side, open:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project
```

---

## Step 5: Prepare and Upload the YOLO Dataset

### 5.1 Required directory structure

A YOLO object-detection dataset should use this structure:

```text
yolo_project/
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

An image containing objects must have a label file with the same base filename:

```text
images/train/image001.jpg
labels/train/image001.txt
```

Each line in a detection label file must use:

```text
class_id x_center y_center width height
```

Class IDs start at `0`. The four box coordinates must use normalised `xywh` values in the range `0–1`. An image containing no labelled objects may have no label file or an empty label file.

### 5.2 Upload the dataset

Use WinSCP to upload the four `train`/`val` directories into:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/dataset
```

![Uploading the dataset with WinSCP](https://github.com/user-attachments/assets/ee2b1b0d-15ff-44f3-9c24-dcf913ad68ef)

### 5.3 Create `data.yaml`

In WindTerm, run:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"

cat > "$PROJECT/dataset/data.yaml" <<YAML
path: $PROJECT/dataset
train: images/train
val: images/val

names:
  0: microgel
YAML
```

The closing `YAML` line terminates the shell input and is not written into the file.

![Creating data.yaml](https://github.com/user-attachments/assets/e852dc3b-128e-49b8-bc1a-dcf9362fd216)

Display the file:

```bash
cat "$PROJECT/dataset/data.yaml"
```

![Checking data.yaml](https://github.com/user-attachments/assets/0119ebe0-d1f7-4a9f-99d5-b54ca38f04d0)

The filename must be exactly `data.yaml`, not `data.yalm`.

---

## Step 6: Load the Miniforge Conda Module

Search the current module catalogue rather than assuming that an old module name still exists:

```bash
module --ignore-cache spider 2>&1 | grep -iE "anaconda|miniconda|miniforge|conda|python"
module spider miniforge3
```

The module recorded when this guide was prepared was:

```text
miniforge3/25.3.0-python3.12.10-dynamic
```

If it is still available, load it:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
```

Barkla may print an informational message describing the dynamic Miniforge build. That message is not an error.

Confirm that Conda is available:

```bash
which conda
conda --version
```

The exact version may change. A valid path under `/opt/apps/` and a Conda version number indicate that the module loaded successfully.

If the recorded module is no longer present, use the exact current name returned by `module spider miniforge3` in this guide's later commands and in `train.slurm`.

---

## Step 7: Create the YOLO Conda Environment

Load Miniforge and initialise Conda in the current Bash shell:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
```

Create a directory for environments:

```bash
mkdir -p "/mnt/fastscratch/users/$USER/conda_envs"
```

Create a Python 3.11 environment:

```bash
conda create \
    -p "/mnt/fastscratch/users/$USER/conda_envs/yolo" \
    python=3.11 \
    pip \
    -y
```

![Creating the YOLO Conda environment](https://github.com/user-attachments/assets/7868c249-9981-4f45-b240-b7b53a6b7a12)

Activate it:

```bash
conda activate "/mnt/fastscratch/users/$USER/conda_envs/yolo"
```

Check the active Python installation:

```bash
which python
python --version
python -m pip --version
```

The output should be similar to:

```text
/mnt/fastscratch/users/sgzjia25/conda_envs/yolo/bin/python
Python 3.11.x
pip ... from /mnt/fastscratch/users/sgzjia25/conda_envs/yolo/lib/python3.11/site-packages/pip
```

![Checking the active Conda environment](https://github.com/user-attachments/assets/2b87eca6-04a8-4ec3-a9f6-dd8a5695785d)

If `which python` displays `/usr/bin/python`, stop and activate the Conda environment correctly before installing anything.

Create the environment only once. In later login sessions, use:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "/mnt/fastscratch/users/$USER/conda_envs/yolo"
```

---

## Step 8: Install PyTorch and Ultralytics

### 8.1 Set paths and activate the environment

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

mkdir -p "$PROJECT/dataset" "$PROJECT/logs" "$PROJECT/weights"
cd "$PROJECT"

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

which python
python --version
python -m pip --version
```

Do not continue if `which python` points to `/usr/bin/python`.

### 8.2 Install PyTorch with CUDA 12.8 support

```bash
python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

PyTorch officially publishes this version combination. The CUDA 12.8 wheel contains the CUDA runtime libraries needed by PyTorch; it still requires a sufficiently recent NVIDIA driver on the allocated GPU node.

### 8.3 Install a fixed Ultralytics version

```bash
python -m pip install ultralytics==8.4.102
```

Pinning the version makes the environment reproducible. Test any later upgrade in a separate environment before using it for production results.

### 8.4 Verify the installation

```bash
python -m pip check

python -c "import torch; print('PyTorch:', torch.__version__); print('Built CUDA:', torch.version.cuda)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "from ultralytics import YOLO; print('YOLO import successful')"
```

Expected versions:

```text
PyTorch: 2.8.0+cu128
Built CUDA: 12.8
Ultralytics: 8.4.102
YOLO import successful
```

On a login node, this command may print `False`:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

That is normal because a login shell has not been allocated a GPU. The decisive test is performed inside the Slurm GPU job.

### 8.5 Record the environment

```bash
python -m pip freeze > "$PROJECT/yolo_requirements.txt"
grep -iE "torch|torchvision|ultralytics" "$PROJECT/yolo_requirements.txt"
```

Keep this file with the training scripts and results.

---

## Step 9: Check the Dataset and Download the Model Before Training

### 9.1 Check the project structure

The final structure should be:

```text
yolo_project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── logs/
├── weights/
├── train.py
├── train.slurm
└── yolo_requirements.txt
```

Set the path and check the dataset:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

test -f dataset/data.yaml && echo "data.yaml exists"
test -d dataset/images/train && echo "training images directory exists"
test -d dataset/images/val && echo "validation images directory exists"
test -d dataset/labels/train && echo "training labels directory exists"
test -d dataset/labels/val && echo "validation labels directory exists"

cat dataset/data.yaml
```

### 9.2 Count images and labels

```bash
find dataset/images/train -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
find dataset/labels/train -type f -name '*.txt' | wc -l

find dataset/images/val -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
find dataset/labels/val -type f -name '*.txt' | wc -l
```

Different counts are not automatically an error because images with no objects may have no label file. Images that contain objects must have a matching `.txt` filename.

Inspect several label files and confirm that every row contains five values, the class IDs are valid, and the four coordinates are between `0` and `1`.

### 9.3 Download YOLO26m before requesting a GPU

Do not rely on a compute node having internet access. Download the weights from the login node before submitting the job:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

cd "$PROJECT/weights"
python -c "from ultralytics import YOLO; YOLO('yolo26m.pt'); print('YOLO26m is ready')"
ls -lh yolo26m.pt
```

This file must exist and be non-empty:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/weights/yolo26m.pt
```

If the login node cannot access the download URL, download `yolo26m.pt` on Windows from an official Ultralytics source and upload it to the `weights` directory with WinSCP.

---

## Step 10: Create the Training Program (`train.py`)

Enter the project directory:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"
```

Create `train.py`:

```bash
cat > train.py <<'PY'
import argparse
import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
MODEL_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26m.pt"
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26m on the microgel dataset"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Use -1 to let Ultralytics estimate a safe batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing last.pt checkpoint",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {description}: {path}")


def main() -> None:
    args = parse_args()

    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        require_file(checkpoint, "resume checkpoint")

        print(f"Resuming training from: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return

    require_file(DATA_YAML, "dataset configuration")
    require_file(MODEL_WEIGHTS, "pretrained model")

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        run_name = f"microgel_yolo26m_{job_id}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"microgel_yolo26m_local_{timestamp}"

    print(f"Dataset: {DATA_YAML}")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Run name: {run_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Workers: {args.workers}")

    model = YOLO(str(MODEL_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        optimizer="auto",
        patience=20,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        amp=True,
        cache=False,
        seed=0,
        deterministic=True,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
PY
```

The pretrained model path is fixed to:

```text
weights/yolo26m.pt
```

Each new Slurm job writes to a separate directory:

```text
runs/microgel_yolo26m_JOBID
```

For example, job `123456` writes to:

```text
runs/microgel_yolo26m_123456
```

The `--resume` path is intended for an interrupted run. It restores the saved epoch, optimiser, learning-rate scheduler and other training state; it is not the same as loading weights to start a new experiment.

---

## Step 11: Check Barkla GPU Partitions and Modules

Inspect the live GPU partitions:

```bash
sinfo -o "%20P %15a %20G %15l" | grep -i gpu
```

The existing project scripts use:

```text
gpu-a-lowsmall
```

Confirm that it still exists and inspect its limits:

```bash
sinfo -p gpu-a-lowsmall
scontrol show partition gpu-a-lowsmall
```

Confirm that Miniforge still exists:

```bash
module spider miniforge3/25.3.0-python3.12.10-dynamic
```

If Barkla administrators require a CUDA toolkit module, inspect the current CUDA modules:

```bash
module spider cuda
module spider cuda/12.8.0-gcc14.2.0
```

The PyTorch `cu128` wheel already includes its CUDA runtime libraries. A separate CUDA toolkit module is normally unnecessary for standard training and can introduce library conflicts. The Slurm script below therefore does not load one by default. Add the current CUDA module only if Barkla's documented configuration requires it or if compilation of custom CUDA extensions is needed.

If a partition or module no longer exists, do not guess its replacement. Use the exact live names returned by `sinfo` and `module spider`, then update `train.slurm`.

---

## Step 12: Create the Slurm Job Script (`train.slurm`)

Ensure the log directory exists and enter the project directory:

```bash
mkdir -p /mnt/fastscratch/users/sgzjia25/yolo_project/logs
cd /mnt/fastscratch/users/sgzjia25/yolo_project
```

Create `train.slurm`:

```bash
cat > train.slurm <<'SLURM'
#!/bin/bash -l

#SBATCH --job-name=yolo_microgel
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/fastscratch/users/sgzjia25/yolo_project
#SBATCH --output=/mnt/fastscratch/users/sgzjia25/yolo_project/logs/%x.%N.%j.out
#SBATCH --error=/mnt/fastscratch/users/sgzjia25/yolo_project/logs/%x.%N.%j.err

set -euo pipefail

PROJECT="/mnt/fastscratch/users/sgzjia25/yolo_project"
YOLO_ENV="/mnt/fastscratch/users/sgzjia25/conda_envs/yolo"

EPOCHS="${YOLO_EPOCHS:-100}"
IMGSZ="${YOLO_IMGSZ:-1920}"
BATCH="${YOLO_BATCH:--1}"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job name:     ${SLURM_JOB_NAME}"
echo "Partition:    ${SLURM_JOB_PARTITION}"
echo "Node list:    ${SLURM_JOB_NODELIST}"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "========================================"

module purge
module load miniforge3/25.3.0-python3.12.10-dynamic

eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$WORKERS"
export MPLBACKEND=Agg

echo "Python executable: $(which python)"
python --version

echo "PyTorch and allocated GPU:"
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Compiled architectures:", torch.cuda.get_arch_list())

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: Slurm allocated the job, but PyTorch cannot access a GPU"
    )

print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
PY

echo "NVIDIA status:"
nvidia-smi

cd "$PROJECT"

if [[ -n "${YOLO_RESUME:-}" ]]; then
    echo "Resuming from: $YOLO_RESUME"
    srun python -u train.py --resume "$YOLO_RESUME"
else
    echo "Starting a new training run"
    echo "Epochs:  $EPOCHS"
    echo "Image:   $IMGSZ"
    echo "Batch:   $BATCH"
    echo "Workers: $WORKERS"

    srun python -u train.py \
        --epochs "$EPOCHS" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        --workers "$WORKERS"
fi

echo "========================================"
echo "Finish time: $(date)"
echo "Training command completed"
echo "========================================"
SLURM
```

The script requests one GPU, four CPU cores, 32 GB of RAM and 24 hours. These are starting values, not guaranteed valid limits. Adjust them only after checking the live partition configuration.

If the live site requires a CUDA module, add its verified current name immediately after the Miniforge module line.

---

## Step 13: Run Pre-submission Checks

Set the paths, load the environment and enter the project directory:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

cd "$PROJECT"
```

If the files were created or edited on Windows, remove CRLF line endings:

```bash
sed -i 's/\r$//' train.py train.slurm
```

Check Python and shell syntax:

```bash
python -m py_compile train.py
bash -n train.slurm
```

Confirm all required files and directories:

```bash
test -f train.py && echo "train.py OK"
test -f train.slurm && echo "train.slurm OK"
test -f dataset/data.yaml && echo "data.yaml OK"
test -s weights/yolo26m.pt && echo "model OK"
test -d logs && echo "logs directory OK"
```

Inspect the final files if needed:

```bash
sed -n '1,240p' train.py
sed -n '1,260p' train.slurm
```

Syntax checks cannot verify the partition, modules, dataset contents, NVIDIA driver or available GPU memory. A short Slurm test job is still required.

---

## Step 14: Submit a Short One-epoch Test Job

Do not begin with 100 epochs at `imgsz=1920`. Test the complete workflow with one epoch, `imgsz=640` and batch size `2`:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(sbatch --parsable \
    --export=ALL,YOLO_EPOCHS=1,YOLO_IMGSZ=640,YOLO_BATCH=2 \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted test job: $JOBID"
```

Check its queue state:

```bash
squeue -j "$JOBID" -o "%.18i %.18P %.25j %.8T %.10M %.10l %.30R"
```

After the job ends, inspect the accounting record:

```bash
sacct -j "$JOBID" \
    --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

The main job row should report:

```text
COMPLETED
```

The test should also produce:

```text
runs/microgel_yolo26m_JOBID/weights/best.pt
runs/microgel_yolo26m_JOBID/weights/last.pt
```

Proceed to production only after confirming the accounting state, logs and output files.

---

## Step 15: Submit the Production Training Job

The default settings are:

```text
epochs = 100
imgsz  = 1920
batch  = -1 (automatic estimation)
```

Submit with the defaults:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(sbatch --parsable train.slurm)
JOBID="${JOBID%%;*}"

printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted production job: $JOBID"
```

For a more conservative starting resolution of `1280`:

```bash
JOBID=$(sbatch --parsable \
    --export=ALL,YOLO_EPOCHS=100,YOLO_IMGSZ=1280,YOLO_BATCH=-1 \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted production job: $JOBID"
```

The `--export` overrides avoid repeatedly editing the Python or Slurm files.

---

## Step 16: Check the Job Status, Node and GPU

After a later login, restore the most recently saved job ID:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(cat .last_job_id)
echo "$JOBID"
```

View the job:

```bash
squeue -j "$JOBID" -o "%.18i %.18P %.25j %.8T %.10M %.10l %.30R"
```

Common live states are:

```text
PD  Pending
R   Running
CG  Completing
```

When a job is pending, reasons such as `(Resources)` or `(Priority)` normally mean it is waiting in the queue, not that the Python program has failed.

After the state changes to `R`, get the node assigned to this job:

```bash
NODE=$(squeue -h -j "$JOBID" -t R -o "%N")
echo "Running node: $NODE"
```

Always identify the node by job ID. An account may have several jobs running at once.

If Barkla's `node-usage.sh` helper is available, its recorded modes are:

```text
all  all nodes
cpu  CPU-only nodes
gpu  GPU nodes
```

Show the GPU table:

```bash
node-usage.sh gpu
```

Show only the node assigned to the current job while retaining the table header:

```bash
node-usage.sh gpu | grep -E "^NODE|^----|^${NODE}[[:space:]]"
```

Refresh every five seconds:

```bash
watch -n 5 "node-usage.sh gpu | grep -E '^NODE|^----|^${NODE}[[:space:]]'"
```

Press `Ctrl+C` to stop refreshing. This does not cancel the Slurm job.

---

## Step 17: View the Logs

Once the job has started:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(cat .last_job_id)
ls -lh logs/*"$JOBID"*
```

Follow standard output:

```bash
tail -f logs/yolo_microgel.*."$JOBID".out
```

Follow standard error:

```bash
tail -f logs/yolo_microgel.*."$JOBID".err
```

Press `Ctrl+C` to stop following a file. The training job continues.

Search both logs for common failures:

```bash
grep -iE \
    "error|exception|traceback|out of memory|killed|failed" \
    logs/*"$JOBID"*
```

The beginning of a healthy GPU-job log should include output similar to:

```text
CUDA available: True
GPU: NVIDIA ...
```

If the Slurm log says `CUDA available: False`, the script exits instead of silently training on the CPU.

---

## Step 18: Check Completion, Resume or Cancel

### 18.1 Check the final state

```bash
sacct -j "$JOBID" \
    --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

Important states include:

```text
COMPLETED      Finished successfully
FAILED         Program failed; inspect the logs
OUT_OF_MEMORY  Insufficient CPU or GPU memory; inspect the logs
TIMEOUT        Reached the Slurm time limit
CANCELLED      The job was cancelled
```

### 18.2 Resume an interrupted run from `last.pt`

Assume the interrupted job ID was `123456`:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

OLD_JOBID=123456
LAST_PT="$PROJECT/runs/microgel_yolo26m_${OLD_JOBID}/weights/last.pt"

test -s "$LAST_PT" && echo "Checkpoint found: $LAST_PT"
```

Submit the resume job:

```bash
JOBID=$(sbatch --parsable \
    --export=ALL,YOLO_RESUME="$LAST_PT" \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted resume job: $JOBID"
```

Use `last.pt`, not `best.pt`, to resume an interrupted training state. `resume=True` restores the saved epoch, optimiser and scheduler. It is not intended to extend an already completed experiment beyond its originally configured epoch count.

### 18.3 Cancel a specific job

```bash
scancel "$JOBID"
squeue -j "$JOBID"
```

This command cancels every running and queued job owned by the current account:

```bash
scancel -u "$USER"
```

Do not use it casually.

---

## Step 19: Locate and Verify the Training Results

After the production job finishes:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(cat .last_job_id)
RUN_DIR="$PROJECT/runs/microgel_yolo26m_${JOBID}"

find "$RUN_DIR" -maxdepth 2 -type f | sort
ls -lh "$RUN_DIR/weights"
```

Main weights:

```text
best.pt  Checkpoint with the best validation fitness during the run
last.pt  Checkpoint from the latest completed epoch
```

Common result files include:

```text
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
args.yaml
```

Do not judge success only by the presence of `best.pt`. Confirm all of the following:

- the main `sacct` state is `COMPLETED`;
- the logs contain no traceback, out-of-memory error or silent early termination;
- `results.csv` contains plausible training and validation metrics;
- `best.pt` is not empty;
- representative validation predictions are visually sensible for the scientific task.

Check the weight file:

```bash
test -s "$RUN_DIR/weights/best.pt" && echo "best.pt exists and is not empty"
```

---

## Step 20: Download Results with WinSCP

Connect to:

```text
barklalogin2.liv.ac.uk
```

Open the result directory for the relevant job ID:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/runs/microgel_yolo26m_JOBID
```

Download at least:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
args.yaml
```

Also retain the matching files from the project root:

```text
train.py
train.slurm
dataset/data.yaml
yolo_requirements.txt
```

`fastscratch` must not be the only copy. Download important results promptly or move them to a University-approved long-term storage location.

---

## Quick Reference: Commands Used After Login

### Load the environment

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

cd "$PROJECT"
```

### Submit production training

```bash
JOBID=$(sbatch --parsable train.slurm)
JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "$JOBID"
```

### Restore the last saved job ID

```bash
cd "/mnt/fastscratch/users/$USER/yolo_project"
JOBID=$(cat .last_job_id)
echo "$JOBID"
```

### View status and assigned node

```bash
squeue -j "$JOBID" -o "%.18i %.18P %.25j %.8T %.10M %.10l %.30R"
NODE=$(squeue -h -j "$JOBID" -t R -o "%N")
echo "$NODE"
```

### Follow the standard-output log

```bash
tail -f logs/yolo_microgel.*."$JOBID".out
```

### View the final accounting state

```bash
sacct -j "$JOBID" \
    --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

### Cancel the selected job

```bash
scancel "$JOBID"
```

---

## Troubleshooting

### `Cannot find dataset configuration`

Check:

```bash
ls -lh "/mnt/fastscratch/users/$USER/yolo_project/dataset/data.yaml"
cat "/mnt/fastscratch/users/$USER/yolo_project/dataset/data.yaml"
```

Confirm that the filename is `data.yaml`, not `data.yalm`, and that the `path` entry points to the uploaded dataset.

### `Cannot find pretrained model`

Check:

```bash
ls -lh "/mnt/fastscratch/users/$USER/yolo_project/weights/yolo26m.pt"
```

The file must be non-empty and located exactly where `train.py` expects it.

### `CUDA available: False`

On a login node, this is normally expected. In a Slurm GPU-job log, check the allocation:

```bash
scontrol show job "$JOBID"
```

Confirm that the job requested a GPU:

```text
#SBATCH --gres=gpu:1
```

Also inspect the log's `nvidia-smi`, PyTorch version, built CUDA version and Python executable. `which python` must point to the intended Conda environment.

### CUDA out of memory

First reduce the image size while retaining automatic batch estimation:

```bash
sbatch --export=ALL,YOLO_EPOCHS=100,YOLO_IMGSZ=1280,YOLO_BATCH=-1 train.slurm
```

If memory is still insufficient, use a smaller image size and explicit batch size:

```bash
sbatch --export=ALL,YOLO_EPOCHS=100,YOLO_IMGSZ=960,YOLO_BATCH=1 train.slurm
```

Record any changed parameters with the experiment results.

### `Invalid CUDA 'device=0' requested`

The job may not have received a GPU, PyTorch may be a CPU-only build, or the GPU may not be exposed correctly. Inspect `scontrol show job`, `nvidia-smi` and the PyTorch diagnostics at the start of the log.

### The job remains in `PD`

```bash
squeue -j "$JOBID" -o "%.18i %.8T %.40R"
```

`Resources` or `Priority` usually means the job is waiting. A partition, QOS, account or invalid-resource reason requires correction of the request or advice from Barkla support.

### The job state is `TIMEOUT`

Resume from the run's `last.pt` as described in Step 18. Check the partition's maximum allowed time before changing:

```text
#SBATCH --time=24:00:00
```

### The log file is not present yet

The output filename contains the compute-node name (`%N`) and is normally created when the job starts. A pending job may therefore have no training log yet. Check `squeue` first.

---

## Important Operating Rules

1. Do not run full training directly on `barklalogin1` or `barklalogin2`.
2. Run production GPU training through `sbatch train.slurm`.
3. Begin with a one-epoch, `imgsz=640` test job.
4. Track status, node, logs and results by job ID.
5. Check live partition and module names before submission.
6. Download pretrained weights before requesting a compute node.
7. Use `last.pt` to resume an interrupted run.
8. Do not use `fastscratch` as the only backup.
9. Do not assume that training succeeded merely because `best.pt` exists; check `sacct`, logs, metrics and predictions.
10. Preserve the scripts, dataset YAML, package versions and training arguments with every result.

---

## References

- [University of Liverpool: High-performance computing](https://www.liverpool.ac.uk/research-it/high-performance-computing/)
- [Ultralytics YOLO26 models](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics training mode and resume training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics object-detection dataset format](https://docs.ultralytics.com/datasets/detect/)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [Slurm `sbatch`](https://slurm.schedmd.com/sbatch.html)
- [Slurm `squeue`](https://slurm.schedmd.com/squeue.html)
- [Slurm `sacct`](https://slurm.schedmd.com/sacct.html)
- [Slurm `scancel`](https://slurm.schedmd.com/scancel.html)
