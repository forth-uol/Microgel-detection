# Barkla YOLO Training README - Steps 8 to 20

Last checked: 4 August 2026

This section continues the same style as Steps 1-7: each step has the command to run, the screenshot to compare against, and a short safety note.

Current audited state:

- The environment, dataset, scripts and model weights were checked on Barkla.
- A first one-epoch test job, `10117422`, failed because the old submit command used `--export=ALL` and carried stale `SLURM_*` variables into the job.
- The submit command below is corrected. It uses a clean export and does not carry old Slurm variables.
- The corrected one-epoch test job is `10117602`. It was still `PENDING` at the final check on 4 August 2026 at 12:36.
- No production training job has been submitted. Submit production only after the one-epoch test job is `COMPLETED`.

Do not run YOLO training directly on a login node. Login nodes are only for file management, lightweight checks, environment setup and Slurm submission.

---

## Step 8: Install PyTorch and Ultralytics

### 8.1 Activate the YOLO environment

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"

module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

which python
python --version
python -m pip --version
```

Do not continue if `which python` points to `/usr/bin/python`.

### 8.2 Install and verify the packages

```bash
python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install ultralytics==8.4.102

python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('Built CUDA:', torch.version.cuda)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "from ultralytics import YOLO; print('YOLO import successful')"
```

![Step 8 package verification](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_08_verify_success.png)

### 8.3 Record the environment

```bash
cd "$PROJECT"
python -m pip freeze > yolo_requirements.txt
grep -iE "torch|torchvision|ultralytics" yolo_requirements.txt
ls -lh yolo_requirements.txt
```

![Step 8 requirements file](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_08_record_requirements.png)

Keep `yolo_requirements.txt` with the final training results.

---

## Step 9: Check the Dataset and Download the Model Before Training

The final project dataset should be under:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/dataset
```

Run:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

test -f dataset/data.yaml && echo "data.yaml OK"
test -d dataset/images/train && echo "training images OK"
test -d dataset/images/val && echo "validation images OK"
test -d dataset/labels/train && echo "training labels OK"
test -d dataset/labels/val && echo "validation labels OK"

echo -n "train images: "
find dataset/images/train -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l

echo -n "train labels: "
find dataset/labels/train -type f -name '*.txt' | wc -l

echo -n "val images: "
find dataset/images/val -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l

echo -n "val labels: "
find dataset/labels/val -type f -name '*.txt' | wc -l
```

The checked dataset contains:

```text
train images: 63
train labels: 63
val images:   18
val labels:   18
```

![Step 9 dataset copied and counted](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_13_fix_dataset_location_singleline.png)

Download the model before requesting a GPU:

```bash
export YOLO_ENV="/mnt/fastscratch/users/$USER/conda_envs/yolo"
module load miniforge3/25.3.0-python3.12.10-dynamic
eval "$(conda shell.bash hook)"
conda activate "$YOLO_ENV"

cd "$PROJECT/weights"
python -c "from ultralytics import YOLO; YOLO('yolo26m.pt'); print('YOLO26m is ready')"
ls -lh yolo26m.pt
```

The checked file is:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project/weights/yolo26m.pt
```

---

## Step 10: Create the Training Program (`train.py`)

Enter the project directory:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"
```

Create `train.py` using the checked training program. The local checked copy is:

```text
C:\Users\benef\Documents\Codex\2026-08-03\ban\work\remote_train.py
```

Important paths inside the script:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
MODEL_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26m.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
```

Check syntax after creating the file:

```bash
python -m py_compile train.py
ls -lh train.py
```

![Step 10 train.py syntax check](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_10_train_py_status.png)

The script writes each Slurm run to:

```text
runs/microgel_yolo26m_JOBID
```

---

## Step 11: Check Barkla GPU Partitions and Modules

Check the live GPU partitions:

```bash
sinfo -o "%20P %15a %20G %15l" | grep -i gpu
```

Confirm the selected partition:

```bash
sinfo -p gpu-a-lowsmall
scontrol show partition gpu-a-lowsmall
```

Confirm the Miniforge module:

```bash
module spider miniforge3/25.3.0-python3.12.10-dynamic
```

![Step 11 GPU partition and module check](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_11_partition_modules.png)

If a partition or module no longer exists, stop and use the exact live names returned by `sinfo` and `module spider`.

---

## Step 12: Create the Slurm Job Script (`train.slurm`)

Create the log directory and enter the project:

```bash
mkdir -p /mnt/fastscratch/users/sgzjia25/yolo_project/logs
cd /mnt/fastscratch/users/sgzjia25/yolo_project
```

Create `train.slurm` using the checked Slurm script. The local checked copy is:

```text
C:\Users\benef\Documents\Codex\2026-08-03\ban\work\remote_train.slurm
```

The checked script requests:

```text
partition: gpu-a-lowsmall
GPU:       1
CPUs:      4
memory:    32G
time:      24:00:00
```

Check the script:

```bash
bash -n train.slurm
grep -E "^#SBATCH --partition|^#SBATCH --gres|^#SBATCH --mem|^#SBATCH --time" train.slurm
```

![Step 12 train.slurm syntax check](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_12_train_slurm_status.png)

Do not add unverified CUDA modules. The PyTorch `cu128` wheel includes the CUDA runtime used by PyTorch.

---

## Step 13: Run Pre-submission Checks

Run these checks from the project directory:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

python -m py_compile train.py && echo "train.py OK"
bash -n train.slurm && echo "train.slurm OK"
test -f dataset/data.yaml && echo "data.yaml OK"
test -s weights/yolo26m.pt && echo "model OK"
test -d dataset/images/train && echo "training images OK"
test -d dataset/images/val && echo "validation images OK"
test -d dataset/labels/train && echo "training labels OK"
test -d dataset/labels/val && echo "validation labels OK"
```

![Step 13 final pre-submission checks](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_13_final_presubmit_clean.png)

These checks do not train the model. They only confirm files, syntax, dataset folders and weights before Slurm submission.

---

## Step 14: Submit a Short One-epoch Test Job

Use a small test before production:

```text
epochs = 1
imgsz  = 640
batch  = 2
```

Submit with a clean export:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

JOBID=$(sbatch --parsable \
    --export=YOLO_EPOCHS=1,YOLO_IMGSZ=640,YOLO_BATCH=2 \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted clean test job: $JOBID"

squeue -j "$JOBID" -o "%.12i %.14P %.16j %.10T %.8M %.12R"
```

![Step 14 clean one-epoch test submit](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_14_submit_clean_test_job.png)

Do not use `--export=ALL` here. It can carry stale `SLURM_*` variables from the current shell and make `srun` fail.

---

## Step 15: Submit the Production Training Job

Do not run this step until the one-epoch test job is `COMPLETED`, the logs are clean, and the result files exist.

The current clean test job is still pending:

![Step 15 current queue reason before production](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_16_queue_reason.png)

After the test passes, submit production with:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

JOBID=$(sbatch --parsable train.slurm)
JOBID="${JOBID%%;*}"

printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted production job: $JOBID"
```

For a conservative first production run:

```bash
JOBID=$(sbatch --parsable \
    --export=YOLO_EPOCHS=100,YOLO_IMGSZ=1280,YOLO_BATCH=-1 \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted production job: $JOBID"
```

---

## Step 16: Check the Job Status, Node and GPU

Restore the latest job ID:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(cat .last_job_id)
echo "JOBID=$JOBID"
```

Check Slurm:

```bash
squeue -j "$JOBID" -o "%.12i %.14P %.16j %.10T %.8M %.12R"
sacct -j "$JOBID" --format=JobID,JobName,State,Elapsed,ExitCode
```

![Step 16 latest test job status](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_16_final_status_check.png)

To show only the reason:

```bash
squeue -j "$JOBID" -h -o "State=%T  Time=%M  Limit=%l  Reason=%R"
```

![Step 16 queue reason](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_16_queue_reason.png)

If the job is `PENDING` with `(Priority)` or `(Resources)`, wait. Do not change resource requests unless Barkla policy and `sinfo` show that the change is appropriate.

When the job is running, identify the assigned node:

```bash
NODE=$(squeue -h -j "$JOBID" -t R -o "%N")
echo "Running node: $NODE"
```

If `node-usage.sh` is available:

```bash
node-usage.sh gpu | grep -E "^NODE|^----|^${NODE}[[:space:]]"
```

---

## Step 17: View the Logs

After the job starts:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(cat .last_job_id)
ls -lh logs/*"$JOBID"*
```

View output:

```bash
tail -f logs/yolo_microgel.*."$JOBID".out
```

View errors:

```bash
tail -f logs/yolo_microgel.*."$JOBID".err
```

Press `Ctrl+C` to stop following a log file. This does not cancel the Slurm job.

The screenshot below shows the log-checking method. It is from the first failed test job, `10117422`; the corrected test job `10117602` had not started yet when checked.

![Step 17 log view example](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_17_test_job_logs.png)

The failure diagnosis showed the old `--export=ALL` problem:

![Step 17 failure diagnosis](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_17b_failure_diagnosis.png)

For the corrected job, the beginning of a healthy log should include:

```text
CUDA available: True
GPU: NVIDIA ...
Starting a new training run
```

---

## Step 18: Check Completion, Resume or Cancel

Check the final state:

```bash
JOBID=$(cat .last_job_id)
sacct -j "$JOBID" --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,ExitCode
```

![Step 18 completion check](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_18_completion_check_pending.png)

Only proceed if the main job row says:

```text
COMPLETED
```

To resume an interrupted run:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

OLD_JOBID=123456
LAST_PT="$PROJECT/runs/microgel_yolo26m_${OLD_JOBID}/weights/last.pt"
test -s "$LAST_PT" && echo "Checkpoint found: $LAST_PT"

JOBID=$(sbatch --parsable \
    --export=YOLO_RESUME="$LAST_PT" \
    train.slurm)

JOBID="${JOBID%%;*}"
printf '%s\n' "$JOBID" > .last_job_id
echo "Submitted resume job: $JOBID"
```

Use `last.pt`, not `best.pt`, when resuming an interrupted training state.

Cancel only a specific job when needed:

```bash
scancel "$JOBID"
squeue -j "$JOBID"
```

Do not use `scancel -u "$USER"` unless you truly intend to cancel every queued and running job owned by the account.

---

## Step 19: Locate and Verify the Training Results

After the job is complete:

```bash
export PROJECT="/mnt/fastscratch/users/$USER/yolo_project"
cd "$PROJECT"

JOBID=$(cat .last_job_id)
RUN_DIR="$PROJECT/runs/microgel_yolo26m_${JOBID}"

find "$RUN_DIR" -maxdepth 2 -type f | sort
ls -lh "$RUN_DIR/weights"
```

Current checked state: the corrected test job was still pending, so the run directory had not been created yet.

![Step 19 results check while pending](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_19_results_check_pending.png)

After a successful run, verify:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
args.yaml
```

Also confirm that:

- `sacct` reports `COMPLETED`;
- logs contain no traceback, out-of-memory message or silent early termination;
- `best.pt` is not empty;
- validation predictions make sense for the scientific task.

---

## Step 20: Download Results with WinSCP

Open WinSCP and go to the project:

```text
/mnt/fastscratch/users/sgzjia25/yolo_project
```

![Step 20 WinSCP project view](C:/Users/benef/Documents/Codex/2026-08-03/ban/outputs/screenshots_steps/step_20_winscp_project_view_final.png)

After the completed run creates a result directory, open:

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

Also keep the matching setup files:

```text
train.py
train.slurm
dataset/data.yaml
yolo_requirements.txt
```

`fastscratch` should not be the only copy of important results. Download results promptly or move them to University-approved long-term storage.
