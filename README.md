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


## Step 6: Load the Miniforge Conda Module on Barkla

First, search for the available Conda-related modules:

```bash
module --ignore-cache spider 2>&1 | grep -iE "anaconda|miniconda|miniforge|conda|python"
module spider miniforge3
```

On Barkla, the required Miniforge module is:

```text
miniforge3/25.3.0-python3.12.10-dynamic
```

Load the module:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
```

Barkla may display the following informational message:

```text
This module is based on Miniforge3-25.3.0-3-Linux-x86_64.sh and includes some frequently used packages.
It's called 'dynamic' because more light packages may be added later and conda may be updated to newer version.
```

This is an informational message, not an error.

Check that Conda has loaded correctly:

```bash
which conda
conda --version
```

The expected output is:

```text
/opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10_dynamic/bin/conda
conda 25.7.0
```

If a valid Conda path and version are displayed, the Miniforge module has loaded successfully.


## Step 7: Create the YOLO Conda Environment

Make sure the Miniforge module has already been loaded:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
```

Create a directory for the Conda environments:

```bash
mkdir -p /mnt/fastscratch/users/$USER/conda_envs
```

Initialise Conda in the current Bash shell:

```bash
eval "$(conda shell.bash hook)"
```

After this command, the shell prompt may begin with:

```text
(base)
```

This indicates that Conda has been initialised successfully.

Create a new Conda environment containing Python 3.11 and pip:

```bash
conda create \
    -p /mnt/fastscratch/users/$USER/conda_envs/yolo \
    python=3.11 \
    pip \
    -y
```
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/7868c249-9981-4f45-b240-b7b53a6b7a12" />

Activate the environment:

```bash
conda activate /mnt/fastscratch/users/$USER/conda_envs/yolo
```

Check the active Python environment:

```bash
which python
python --version
python -m pip --version
```

The expected output is similar to:

```text
/mnt/fastscratch/users/sgzjia25/conda_envs/yolo/bin/python
Python 3.11.15
pip 26.2 from /mnt/fastscratch/users/sgzjia25/conda_envs/yolo/lib/python3.11/site-packages/pip
```

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/2b87eca6-04a8-4ec3-a9f6-dd8a5695785d" />

The shell prompt should also begin with the environment path:

```text
(/mnt/fastscratch/users/sgzjia25/conda_envs/yolo)
```

The Python path must not be:

```text
/usr/bin/python
```

If `/usr/bin/python` is shown, the Conda environment has not been activated correctly.

Do not run `conda create` again after the environment has been created. For future sessions, only load Miniforge, initialise Conda and activate the existing environment:

```bash
module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate /mnt/fastscratch/users/$USER/conda_envs/yolo
```

---

Step 8: Install PyTorch and Ultralytics

8.1 Set the paths for the current terminal session

Run the following commands in WindTerm:

export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

mkdir -p "$PROJECT/dataset"
mkdir -p "$PROJECT/logs"
mkdir -p "$PROJECT/weights"

cd "$PROJECT"
pwd

pwd should display:

/mnt/fastscratch/users/sgzjia25/yolo_project

8.2 Load and activate the Conda environment

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

Check the current Python installation:

which python
python --version
python -m pip --version

which python should display something similar to:

/mnt/fastscratch/users/sgzjia25/conda_envs/yolo/bin/python

If it displays /usr/bin/python, do not continue with the installation. Reload Miniforge and activate the Conda environment first.

8.3 Install PyTorch with CUDA 12.8 support

python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

This is an officially supported PyTorch version combination.

8.4 Install a fixed version of Ultralytics

To make the environment reproducible, this guide uses the following fixed version:

python -m pip install ultralytics==8.4.102

This is the version used when this guide was written. If you need to upgrade it later, verify the new version with a short test job first. Do not upgrade immediately before a production training run.

8.5 Verify the installation

python -m pip check

python -c "import torch; print('PyTorch:', torch.__version__); print('Built CUDA:', torch.version.cuda)"

python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"

python -c "from ultralytics import YOLO; print('YOLO import successful')"

The expected output should look similar to:

PyTorch: 2.8.0+cu128
Built CUDA: 12.8
Ultralytics: 8.4.102
YOLO import successful

When you run the following command on a login node, the result may be False:

python -c "import torch; print(torch.cuda.is_available())"

This is normal because no GPU has been allocated to the current terminal session on the login node. The final GPU check must be performed inside a Slurm GPU job.

8.6 Save the environment versions

python -m pip freeze > "$PROJECT/yolo_requirements.txt"

Check the file:

grep -iE "torch|torchvision|ultralytics" "$PROJECT/yolo_requirements.txt"

Step 9: Check the dataset and download the model in advance

9.1 Confirm the dataset path

The final dataset structure should be:

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
└── train.slurm

Check the actual directories:

cd "$PROJECT"

test -f dataset/data.yaml && echo "data.yaml exists"
test -d dataset/images/train && echo "training images directory exists"
test -d dataset/images/val && echo "validation images directory exists"
test -d dataset/labels/train && echo "training labels directory exists"
test -d dataset/labels/val && echo "validation labels directory exists"

Display data.yaml:

cat dataset/data.yaml

Its contents should be:

path: /mnt/fastscratch/users/sgzjia25/yolo_project/dataset
train: images/train
val: images/val

names:
  0: microgel

Do not write the heredoc closing marker YAML into data.yaml.

9.2 Check the numbers of images and labels

find dataset/images/train -type f | wc -l
find dataset/labels/train -type f -name '*.txt' | wc -l

find dataset/images/val -type f | wc -l
find dataset/labels/val -type f -name '*.txt' | wc -l

Images that contain no objects may have no label file. Images that contain objects should have a .txt file with the same base name. For example:

images/train/image001.jpg
labels/train/image001.txt

Each detection-label line should use the following format:

class_id x_center y_center width height

All four coordinate values must be normalized to the range 0–1.

9.3 Download the YOLO26 weights in advance

Do not rely on a GPU compute node having internet access. Download the model before submitting the job.

cd "$PROJECT/weights"

python -c "from ultralytics import YOLO; YOLO('yolo26m.pt'); print('YOLO26m is ready')"

ls -lh yolo26m.pt

The following file must exist:

/mnt/fastscratch/users/sgzjia25/yolo_project/weights/yolo26m.pt

If the Barkla login node cannot access the model download URL, you can download the weights on Windows and then upload them to the weights directory using WinSCP.

Step 10: Create the training program train.py

Enter the project directory:

cd "$PROJECT"

Create the file:

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
    parser = argparse.ArgumentParser(description="Train YOLO26 on the microgel dataset")
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
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
PY

The model path is fixed to the following location inside the project:

weights/yolo26m.pt

Therefore, the training job will not attempt to download the model from the internet.

Each Slurm job uses a separate results directory:

runs/microgel_yolo26m_JOBID

For example, if the Job ID is 123456, the results directory will be:

runs/microgel_yolo26m_123456

Step 11: Check Barkla GPU partitions and modules

First, view the GPU partitions:

sinfo -o "%20P %15a %20G %15l" | grep -i gpu

This guide uses:

gpu-a-lowsmall

Confirm that this partition currently exists:

sinfo -p gpu-a-lowsmall
scontrol show partition gpu-a-lowsmall

Confirm that the required modules still exist:

module spider miniforge3/25.3.0-python3.12.10-dynamic
module spider cuda/12.8.0-gcc14.2.0

If a partition or module does not exist, do not guess its name. Use the current names shown by sinfo and module spider to update the Slurm file in the next step.

The PyTorch cu128 wheel already includes the required CUDA runtime libraries. Loading the CUDA module mainly provides compilation tools or satisfies Barkla's software-environment requirements. If the CUDA module conflicts with PyTorch libraries, remove the CUDA-module line from the Slurm file only after confirming this with the administrators.

Step 12: Create the Slurm job file train.slurm

First, make sure that the log directory exists:

mkdir -p /mnt/fastscratch/users/sgzjia25/yolo_project/logs
cd /mnt/fastscratch/users/sgzjia25/yolo_project

Create the job file:

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
module load cuda/12.8.0-gcc14.2.0

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
    raise SystemExit("ERROR: Slurm allocated the job, but PyTorch cannot access a GPU")

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

Note: #SBATCH lines do not expand the $USER variable, so these lines use the full path containing the username.

Step 13: Pre-submission checks

If the files were uploaded from Windows, first remove CRLF line endings:

cd "$PROJECT"
sed -i 's/\r$//' train.py train.slurm

Check the Python syntax:

python -m py_compile train.py

Check the shell syntax:

bash -n train.slurm

Confirm the key files:

test -f train.py && echo "train.py OK"
test -f train.slurm && echo "train.slurm OK"
test -f dataset/data.yaml && echo "data.yaml OK"
test -f weights/yolo26m.pt && echo "model OK"
test -d logs && echo "logs directory OK"

View the final files:

sed -n '1,240p' train.py
sed -n '1,260p' train.slurm

py_compile and bash -n check syntax only. They cannot verify the GPU partition, modules, dataset contents, or whether the available GPU memory is sufficient. You must therefore still run a short test job first.

Step 14: Submit a short one-epoch test job first

Do not start with 100 epochs at a resolution of 1920 on the first attempt.

Test the complete workflow with 1 epoch, a resolution of 640, and batch size 2:

cd "$PROJECT"

JOBID=$(sbatch --parsable \
    --export=ALL,YOLO_EPOCHS=1,YOLO_IMGSZ=640,YOLO_BATCH=2 \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id

echo "Submitted test job: $JOBID"

Check the status:

squeue -j "$JOBID" -o "%.18i %.18P %.25j %.8T %.10M %.10l %.30R"

After the test job has finished, check it with:

sacct -j "$JOBID" \
    --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES,ExitCode

The desired state is:

COMPLETED

The following files should also be generated:

runs/microgel_yolo26m_JOBID/weights/best.pt
runs/microgel_yolo26m_JOBID/weights/last.pt

Proceed to production training only after the short test has succeeded.

Step 15: Submit the production training job

The default parameters in train.slurm are:

epochs = 100
imgsz = 1920
batch = -1 (automatic estimation)

Submit the production training job:

cd "$PROJECT"

JOBID=$(sbatch --parsable train.slurm)
JOBID="${JOBID%%;*}"

printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted production job: $JOBID"

To start with the more conservative resolution of 1280, run:

JOBID=$(sbatch --parsable \
    --export=ALL,YOLO_EPOCHS=100,YOLO_IMGSZ=1280,YOLO_BATCH=-1 \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted production job: $JOBID"

This uses --export to override the training parameters, so you do not need to edit train.py or train.slurm repeatedly.

Step 16: Check the status, node, and GPU

After logging in to WindTerm again, you can restore the most recently saved Job ID:

export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"
JOBID=$(cat .last_job_id)
echo "$JOBID"

View the job:

squeue -j "$JOBID" -o "%.18i %.18P %.25j %.8T %.10M %.10l %.30R"

Common states:

PD = Pending, waiting for resources
R  = Running
CG = Completing

When the job is in PD, the final column may show:

(Resources)
(Priority)

This usually means the job is waiting in the queue and does not indicate that the program has failed.

After the job changes to R, obtain its node using the Job ID:

NODE=$(squeue -h -j "$JOBID" -t R -o "%N")
echo "Running node: $NODE"

Do not use the node of the "first running job under the current account," because the account may have several different jobs running at the same time.

Barkla's node-usage.sh accepts one optional mode argument:

all = show all nodes
cpu = show CPU-only nodes
gpu = show GPU nodes

It does not accept a node name as its argument. To show all GPU nodes, run:

node-usage.sh gpu

To display only the GPU node running this specific job while preserving the table header, run:

node-usage.sh gpu | grep -E "^NODE|^----|^${NODE}[[:space:]]"

Refresh the selected node information every five seconds:

watch -n 5 "node-usage.sh gpu | grep -E '^NODE|^----|^${NODE}[[:space:]]'"

Press Ctrl+C to stop refreshing. This does not stop the Slurm training job.

Step 17: View the logs

After the job starts running:

cd "$PROJECT"
ls -lh logs/*"$JOBID"*

View standard output:

tail -f logs/yolo_microgel.*."$JOBID".out

View standard error:

tail -f logs/yolo_microgel.*."$JOBID".err

Press Ctrl+C to stop following the log. This does not stop the training job.

Search for common errors:

grep -iE \
    "error|exception|traceback|out of memory|killed|failed" \
    logs/*"$JOBID"*

The beginning of the job log should contain output similar to:

CUDA available: True
GPU: NVIDIA ...

If the job displays CUDA available: False, the Slurm job should exit immediately instead of continuing to train on the CPU.

Step 18: Check completion status, resume training, and cancel jobs

18.1 Check the status after the job ends

sacct -j "$JOBID" \
    --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES,ExitCode

Main states:

COMPLETED      Finished successfully
FAILED         Program failed; check the logs
OUT_OF_MEMORY  Insufficient memory
TIMEOUT        Reached the Slurm time limit
CANCELLED      Job was cancelled

18.2 Resume training from last.pt

Assume that the original Job ID is 123456:

OLD_JOBID=123456
LAST_PT="$PROJECT/runs/microgel_yolo26m_${OLD_JOBID}/weights/last.pt"

test -f "$LAST_PT" && echo "Checkpoint found: $LAST_PT"

Submit the resume job:

JOBID=$(sbatch --parsable \
    --export=ALL,YOLO_RESUME="$LAST_PT" \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id

echo "Submitted resume job: $JOBID"

resume=True restores the epoch, optimizer, learning-rate scheduler, and other training states. It is not equivalent to loading only the model weights and starting a new experiment.

18.3 Cancel a job

Cancel a specific job:

scancel "$JOBID"

Then check:

squeue -j "$JOBID"

The command for cancelling all jobs under the current account is:

scancel -u "$USER"

Do not use this command casually, because it cancels every running and queued job under the account.

Step 19: Locate the training results

After the production job has finished:

RUN_DIR="$PROJECT/runs/microgel_yolo26m_${JOBID}"

find "$RUN_DIR" -maxdepth 2 -type f | sort

The model weights are located in:

ls -lh "$RUN_DIR/weights"

Main files:

best.pt  The model with the best validation-set performance
last.pt  The checkpoint from the final completed epoch

Other common result files include:

results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
args.yaml

Do not judge training success only by the existence of files. Also confirm all of the following:

The sacct state is COMPLETED.

The logs contain no traceback or out-of-memory error.

results.csv contains reasonable training and validation metrics.

The size of best.pt is not 0 bytes.

Check the file:

test -s "$RUN_DIR/weights/best.pt" && echo "best.pt exists and is not empty"

Step 20: Download the results using WinSCP

Connect in WinSCP using:

barklalogin2.liv.ac.uk

Open the results directory for the corresponding Job ID:

/mnt/fastscratch/users/sgzjia25/yolo_project/runs/microgel_yolo26m_JOBID/weights

Download:

best.pt
last.pt

It is also recommended to download:

results.csv
results.png
confusion_matrix.png
PR_curve.png
F1_curve.png
args.yaml

fastscratch should not be treated as the only long-term backup location. After training finishes, download important weights and results to Windows as soon as possible, or copy them to a university-approved long-term storage location.

Most frequently used commands after each login

Load the environment

export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

cd "$PROJECT"

Submit production training

JOBID=$(sbatch --parsable train.slurm)
JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "$JOBID"

Restore the Job ID

cd /mnt/fastscratch/users/$USER/yolo_project
JOBID=$(cat .last_job_id)
echo "$JOBID"

View the status

squeue -j "$JOBID" -o "%.18i %.18P %.25j %.8T %.10M %.10l %.30R"

View the node

NODE=$(squeue -h -j "$JOBID" -t R -o "%N")
echo "$NODE"

View the log

tail -f logs/yolo_microgel.*."$JOBID".out

View the final status

sacct -j "$JOBID" \
    --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,AllocTRES,ExitCode

Cancel a specific job

scancel "$JOBID"

Common errors

Cannot find dataset configuration

Check:

ls -lh /mnt/fastscratch/users/$USER/yolo_project/dataset/data.yaml

Make sure that the file has not been incorrectly named data.yalm.

Cannot find pretrained model

Check:

ls -lh /mnt/fastscratch/users/$USER/yolo_project/weights/yolo26m.pt

CUDA available: False

If this appears on a login node, it is usually normal.

If this appears in a Slurm GPU-job log, check:

scontrol show job "$JOBID"

Also confirm that the job actually requested:

#SBATCH --gres=gpu:1

You should also check that which python points to the correct Conda environment.

CUDA out of memory

First, reduce the image size. For example:

sbatch --export=ALL,YOLO_EPOCHS=100,YOLO_IMGSZ=1280,YOLO_BATCH=-1 train.slurm

If memory is still insufficient, use:

sbatch --export=ALL,YOLO_EPOCHS=100,YOLO_IMGSZ=960,YOLO_BATCH=1 train.slurm

Invalid CUDA 'device=0' requested

This usually means that the job did not receive a GPU, PyTorch was installed as a CPU-only build, or the GPU was not exposed correctly to the job. Check nvidia-smi and the PyTorch diagnostic output at the beginning of the Slurm log.

The Slurm job remains in PD

Run:

squeue -j "$JOBID" -o "%.18i %.8T %.40R"

Resources or Priority usually means that the job is simply waiting for resources. If the output reports a partition, QOS, account, or resource-request error, modify train.slurm or contact the Barkla administrators.

The job state is TIMEOUT

Use last.pt in the corresponding run directory to resume training as described in Step 18. You can also confirm the maximum time allowed by the partition and then adjust the following setting as appropriate:

#SBATCH --time=24:00:00

Important rules

Do not run the full python train.py command directly on barklalogin1 or barklalogin2.

Login nodes should be used only for file management, lightweight checks, environment installation, script preparation, and job submission.

Production GPU training must use sbatch train.slurm.

The first training attempt must begin with a short test using 1 epoch and a resolution of 640.

Query the status, node, logs, and results by Job ID. Do not simply select the first running job under the account.

Do not rely on compute nodes to download pretrained models automatically.

Do not use fastscratch as the only backup location.

Do not assume training succeeded merely because best.pt exists; also check the sacct state and the logs.

References

Ultralytics YOLO26：https://docs.ultralytics.com/models/yolo26/

Ultralytics training parameters and resume training: https://docs.ultralytics.com/modes/train/

Ultralytics detection dataset format: https://docs.ultralytics.com/datasets/detect/

Installing previous PyTorch versions: https://pytorch.org/get-started/previous-versions/

Slurm sbatch：https://slurm.schedmd.com/sbatch.html

Slurm squeue：https://slurm.schedmd.com/squeue.html



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



