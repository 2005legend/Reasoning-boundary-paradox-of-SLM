# HEG-GRPO on AWS (single GPU)

Standalone training package for the HEG-GRPO ablation, built from the code the Kaggle notebook
actually launched (cell 45) and restructured for one AWS GPU. Only this folder goes to AWS.

```
notebooks/control.ipynb  START HERE once JupyterLab is open: runs every phase from cells
notebooks/monitor.ipynb  live training curves, ETA and cost per run
PREREGISTRATION.md       hypotheses, primary contrasts and analysis, fixed BEFORE the main runs
train.py                 train one (gate, seed, size); resumes automatically; optional final eval
evaluate.py              Pass@k for the base model or any finished run
analyze.py               paper tables: per-condition Pass@k + the pre-registered contrasts (seed-paired,
                         seed x problem bootstrap, Holm), boundary entry/exit, plot
pilot_report.py          go/no-go checks + budget projection from a pilot run (measured eval cost)
rlvr/                    config, rewards, data, clustering, gates, grpo_core, model_utils, ...
scripts/                 setup_instance.sh, start_jupyter.sh, run_queue.sh, prefetch.py, calibrate_meg.py,
                         human_eval.py (blinded annotation sheets + scoring; HUMAN_EVAL_GUIDE.md)
queues/                  pilot.txt, main_0.5B.txt (2x3 factorial x 3 seeds), extras_0.5B.txt, main_1.5B.txt
                         extension_seeds_0.5B.txt (to 5 seeds), extension_240_0.5B.txt (240 steps, trajectory); registered, not yet run
tests/                   CPU tests; setup runs them automatically
```

Conditions (training on MATH, macro topics = its 7 subjects): `vanilla`; `cb_grpo` (topic budget);
`o_self` (SELF: skip prompts whose greedy answer is right); `meg` (rarity credit redistribution over
solution modes); `h_cb_grpo` = CB x SELF; `heg_grpo` = CB x MEG (headline). Extras: `heg_grpo` with
random partitions (control) and `bbg` (problem-level gate from arXiv:2606.15455).

Overview: **Part A** (AWS account, once, $0) -> **B** (launch the GPU machine) -> **C** (connect from
your laptop) -> **D** (install, ~10 min) -> **E** (JupyterLab) -> **F** (pilot, ~$1) -> **G** (main runs,
unattended) -> **H** (coming back) -> **I** (results + cleanup).

---

## Part A. AWS account setup (once, free)

Sign in at https://console.aws.amazon.com. In the top-right region menu pick
**US East (N. Virginia) us-east-1** and keep it there for every step below.

### A1. Check the credits
Search bar -> **Billing and Cost Management** -> left menu **Credits**. Note the remaining amount,
the expiry date, and that the applicable services include EC2 (and ideally S3).

### A2. Request GPU quota (do this first: approval takes hours to a few days)
Search bar -> **Service Quotas** -> **AWS services** -> **Amazon Elastic Compute Cloud (Amazon EC2)**
-> search **Running On-Demand G and VT instances** -> select it -> **Request increase at account level**
-> new value **8** -> Request. (The quota counts vCPUs; a g5.xlarge uses 4.)
You get an email when it is approved; the quota page then shows 8.

### A3. Budget alarm
Billing and Cost Management -> **Budgets** -> **Create budget** -> **Customize (advanced)** ->
**Cost budget** -> Next.
- Period **Monthly**, budgeted amount **200**.
- Budget scope: All AWS services. Under **Advanced options**, untick **Credits** so the budget
  tracks real usage instead of the post-credit bill (which would stay at $0).
- Alerts: add thresholds at **25%, 50%, 75%, 90%** of *Actual* cost, with your email.

### A4. Key pair (your SSH login key)
Search bar -> **EC2** -> left menu **Network & Security -> Key Pairs** -> **Create key pair**:
name `heg`, type **RSA**, format **.pem** -> Create. The browser downloads `heg.pem`.
In **PowerShell** on your laptop:

```powershell
mkdir $env:USERPROFILE\.ssh -Force
Move-Item $env:USERPROFILE\Downloads\heg.pem $env:USERPROFILE\.ssh\heg.pem
icacls $env:USERPROFILE\.ssh\heg.pem /inheritance:r /grant:r "$($env:USERNAME):R"
```

(The `icacls` line is required: Windows OpenSSH refuses keys other users can read.)

### A5. S3 bucket (where every checkpoint and result is backed up)
Search bar -> **S3** -> **Create bucket**: name must be globally unique, e.g.
`heg-grpo-sidaarth-2026`, region us-east-1, leave **Block all public access** ON -> Create.

### A6. IAM role (lets the machine write to S3 without storing any AWS keys on it)
Search bar -> **IAM** -> **Roles** -> **Create role** -> Trusted entity **AWS service**, use case
**EC2** -> Next -> (attach nothing) Next -> name `heg-grpo-ec2` -> Create role.
Open the new role -> **Add permissions -> Create inline policy** -> **JSON** tab, paste (replace the
bucket name twice) -> Next -> name `heg-grpo-s3` -> Create:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket"],
     "Resource": "arn:aws:s3:::heg-grpo-sidaarth-2026"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::heg-grpo-sidaarth-2026/*"}
  ]
}
```

---

## Part B. Launch the GPU machine (after the quota email)

EC2 -> **Instances** -> **Launch instances**:

| Field | Value |
|---|---|
| Name | `heg-grpo` |
| Application and OS Images | search `Deep Learning Base AMI with Single CUDA`, press Enter, pick **Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)**, 64-bit **x86**, newest date. Not the ARM64 one, not the Ubuntu 26.04 "GPU PyTorch" one, not Neuron. (It replaced the older "Base OSS Nvidia Driver" AMI. torch 2.13 needs NVIDIA driver >= 580; the 20260721 release ships 595, and setup prints the driver version first.) |
| Instance type | `g5.xlarge` |
| Key pair | `heg` |
| Network settings | **Edit** -> Create security group -> allow **SSH**, source **My IP**. Do **not** open port 8888 |
| Configure storage | **100** GiB, **gp3** |
| Advanced details -> IAM instance profile | `heg-grpo-ec2` |
| Advanced details -> Shutdown behavior | **Stop** |

Click **Launch instance**, then open the instance and copy its **Public IPv4 address**.
Billing starts now (~$1/hour) and stops when the instance is *stopped* or *terminated*.

---

## Part C. Connect from your laptop (PowerShell)

### C1. SSH shortcut (one-time)
```powershell
notepad $env:USERPROFILE\.ssh\config
```
Paste the block below, put in your instance's IP, and save. In the Save dialog choose
**Save as type: All files** so Notepad does not rename it to `config.txt`.

```
Host heg
    HostName <PUBLIC-IP>
    User ubuntu
    IdentityFile ~/.ssh/heg.pem
    LocalForward 8888 localhost:8888
    ServerAliveInterval 60
```

`ssh heg` now logs you in **and** opens the JupyterLab tunnel at the same time.

### C2. Copy the code up
From PowerShell, in the folder that contains `heg_grpo_aws`
(`C:\Users\USER\sidaarth\reasoning boundry paradox of SLM`):

```powershell
scp -r heg_grpo_aws heg:~/
```

The first connection asks "Are you sure you want to continue connecting?": type `yes`.

---

## Part D. Install on the machine (~10 min, once)

```powershell
ssh heg
```

then on the machine:

```bash
cd ~/heg_grpo_aws
bash scripts/setup_instance.sh
```

It must end with: the A10G listed with `bf16=True`, `prefetch complete`, all tests passed, and an
`arn:aws:sts::...:assumed-role/heg-grpo-ec2/...` line (that proves S3 access works).

---

## Part E. JupyterLab

On the machine (same SSH window):

```bash
bash ~/heg_grpo_aws/scripts/start_jupyter.sh
```

Copy the printed `http://127.0.0.1:8888/?token=...` URL into your laptop's browser. It works as
long as the `ssh heg` window stays open (that window is the tunnel).

In JupyterLab's left file browser open **heg_grpo_aws/notebooks/control.ipynb**:
1. In the settings cell set `S3_URI = 's3://heg-grpo-sidaarth-2026/heg-grpo'` (your bucket).
2. Run the settings cell and the environment-check cell (Shift+Enter). Every line should look healthy.

Rule: **training never runs inside a notebook cell.** The control notebook launches it into a
background `tmux` session, so closing the browser, the laptop lid or the SSH window does not stop it.

---

## Part F. Pilot (~1-1.5 h, ~$1-2)

In `control.ipynb`:
1. Run **launch pilot**. In order: `calibrate` (picks MEG's mode partitioner on base-model rollouts
   with the pre-registered rule, writes `~/artifacts/calibration/0.5B/calibration.json`), two 30-step
   `heg_grpo` runs (lr 1e-6 and 1e-5, each timing a 16-problem eval probe), and a 10-step `o_self` run.
   The machine powers itself off 4 h after this launch, so a forgotten pilot cannot keep billing
   (`sudo shutdown -c` in a terminal cancels that if you need longer).
2. Re-run **status** every few minutes, or open `monitor.ipynb` for curves.
3. When all pilot lines say `DONE`, run **pilot report**.

The report checks:
- **reward is alive**: correctness leaves 0. If not, answer parsing is broken: stop.
- **mode diagnostics are logged** and the partition finds **more than one mode** in a fair share of
  groups (otherwise MEG is a no-op).
- **SELF filter is active but not total** (o_self run): share of prompts whose greedy answer fails.
- **policy is moving**: KL to the base model rises. Flat at ~0 means that lr is too low to learn.
- **completions fit** in `max_new_tokens`, and **groups carry signal** (if most groups are all-right
  or all-wrong, train on easier levels: add `--math_levels 1,2,3` to the train args).
- **budget**: measured seconds/step and measured eval time -> $ per run -> whether the 18 core runs
  (and the 6 extras) fit in 85% of $200, or how many steps would.

Then fill the `[pilot]` values in `PREREGISTRATION.md` (lr, steps, levels, partitioner) before
starting the main queue. If anything says FAIL, or you are unsure, paste the reports to Claude.

---

## Part G. Main runs (unattended, possibly several days)

In `control.ipynb`, set `LR`, `STEPS` and `MAX_HOURS` from the reports, then run **launch main**.

- `main_0.5B.txt` runs the base-model eval, then 6 conditions x 3 seeds, seed-major (if the budget
  runs out you still have complete seed-0 and seed-1 comparisons). `extras_0.5B.txt` (random-partition
  control + BBG) goes last, only if the pilot report says it fits.
- Each queue line trains one (condition, seed), checkpoints every 25 steps (copied to S3), then runs
  the final MATH-500 and GSM8K evals.
- When the queue ends, the machine **powers itself off** (compute billing stops).
  `MAX_HOURS` is a dead-man switch that powers it off regardless, in case something hangs.
- Two failures in a row stop the queue, so a systematic bug cannot burn the budget overnight.
- Check progress any time: reconnect, re-run **status** or open `monitor.ipynb`.

---

## Part H. Coming back (after auto-shutdown, or any interruption)

1. EC2 -> Instances -> select `heg-grpo` -> **Instance state -> Start**. Wait for "Running".
2. The **public IP changes** on every start: copy the new one into `HostName` in `~/.ssh/config`.
3. `ssh heg`, then `bash ~/heg_grpo_aws/scripts/start_jupyter.sh`, then open the new URL.
4. If the queue had not finished, run the same **launch main** cell again: finished runs are
   skipped and an interrupted run resumes from its last checkpoint (at most 25 steps lost).

---

## Part I. Results and cleanup

- In `control.ipynb` run **results**: it writes `~/artifacts/analysis/<benchmark>_<size>.md/.png`
  and displays the plots. Claims follow `PREREGISTRATION.md`: P1/P2 need a Holm-adjusted p < 0.05
  and a seeds x problems CI that excludes 0; interactions are reported as bounds.
- **Human evaluation, before terminating.** Run the "Human evaluation" cell in `control.ipynb`.
  - It builds blinded sheets from the saved completions: `~/artifacts/human_eval/annotator_A|B`.
  - Give each folder to one annotator together with `HUMAN_EVAL_GUIDE.md`. That takes about 4–5 h per person
    and needs no GPU.
  - Keep `key.json` to yourself.
  - When both are back, run `python scripts/human_eval.py score`. It works on the laptop too, with
    `--artifact_root`.
- Download: right-click any file in JupyterLab's file browser -> **Download**, or copy the whole
  results folder from PowerShell: `scp -r heg:~/artifacts .\aws_artifacts`. Everything is also in
  the S3 bucket (S3 console -> bucket -> select files -> Download).
- When fully done: EC2 -> **Instance state -> Terminate**, then EC2 -> **Volumes** and delete any
  leftover volume. A *stopped* machine costs nothing for compute but its 100 GB disk still costs
  about $8/month. Keep the S3 bucket (cents per month).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Launch fails with `VcpuLimitExceeded` | quota (A2) not approved yet |
| Launch fails with `InsufficientInstanceCapacity` | Network settings -> pick a different subnet/availability zone, or retry later |
| `ssh heg` hangs, then times out | your home IP changed: EC2 -> Security Groups -> inbound SSH rule -> source **My IP** -> Save |
| `UNPROTECTED PRIVATE KEY FILE` | re-run the `icacls` line from A4 |
| `Permission denied (publickey)` | wrong key, or user not `ubuntu`; check `~/.ssh/config` |
| setup: "torch cannot see the GPU" | the driver_version printed at the top must be >= 580: relaunch with the newest Deep Learning Base AMI with Single CUDA (Ubuntu 24.04), x86 |
| setup sits at `Fetching 7 files` with no progress for >5 min | Ctrl-C and re-run `bash scripts/setup_instance.sh` (safe to repeat; downloads resume). prefetch already avoids the hf-xet client that stalled in the rehearsal |
| pilot report: every check FAIL, `cor=0.00`, `trunc` near 1 | answers are cut off before `\boxed{}`: check the train args did not lower `--max_new_tokens` (default 1024) |
| env check: `aws sts` fails | EC2 -> instance -> Actions -> Security -> **Modify IAM role** -> `heg-grpo-ec2` |
| JupyterLab page won't load | the `ssh heg` window was closed (reopen it), or JupyterLab is not running (run `start_jupyter.sh`) |
| laptop says port 8888 is in use | change the tunnel line to `LocalForward 8889 localhost:8888` and use `127.0.0.1:8889` |
| "already holds a run with a different config" | you changed settings for an existing run: add a new `--tag`, or delete that run folder |
| CUDA out of memory (unlikely at 0.5B) | add `--score_chunk_size 2` to the train args, with a new `--tag` |
| queue log: "offline mode" / "couldn't find ... in cache" | the queue runs offline; download first: `python scripts/prefetch.py --sizes 0.5B,1.5B`, or launch once with `HF_ONLINE=1` |

## Command-line equivalents (no notebook)

```bash
tmux new -s train
bash scripts/run_queue.sh queues/pilot.txt --s3 s3://BUCKET/heg-grpo
python pilot_report.py --run heg_grpo__pilot_lr1e-5 --budget 200 --steps 800
python pilot_report.py --run o_self__pilot_self --steps 800
bash scripts/run_queue.sh queues/main_0.5B.txt --s3 s3://BUCKET/heg-grpo \
    --train-args "--steps N --lr LR" --usd-per-hour 1.006 --max-hours H --shutdown-when-done
python analyze.py --model_size 0.5B --benchmark math500
python analyze.py --model_size 0.5B --benchmark gsm8k
```

## What changed on 2026-10-03 (after the third novelty check; see PREREGISTRATION.md)

| Area | Before | Now | Why |
|---|---|---|---|
| Training data | GSM8K | MATH train (7 subjects), MATH-500 primary eval, GSM8K out-of-distribution | 0.7% of GSM8K problems admit >1 approach (arXiv:2606.29985), so MEG had nothing to act on |
| Macro topics | KMeans, K=16 on embeddings | the 7 MATH subjects (KMeans kept as an ablation) | no arbitrary K |
| `o_self` | EMA solve rate (alpha 0.1, tau 0.7) | SELF greedy-failure selection each step (2510.02230); greedy answer from the same generate call | the EMA never fired (~2 visits per prompt), so h_cb_grpo was identical to cb_grpo |
| 0.5B rule | o_self / h_cb_grpo forbidden at 0.5B | allowed | scoping choice, not memory; it blocked the primary contrast |
| `meg` | down-weight dominant modes (lambda formula) | mean-preserving rarity credit redistribution (Cue-GRPO rule, alpha 0.8), random-partition control | removes the "lower learning rate" confound; matches the published family |
| Mode partitioner | MiniLM cosine, tau 0.85 | embedding or word-bigram, chosen by a pre-registered calibration on base rollouts | embedding cosine tracks phrasing more than strategy |
| Reward / prompt | format + correctness, custom XML prompt | correctness only (Math-Verify), standard Qwen math prompt | the format reward was identically 0; comparable with the literature |
| Diagnostics | mode entropy only for MEG runs | mode entropy, positive-weight mean, SELF selection rate for every condition; group dumps | shows vanilla's collapse; exposes gate-scale confounds |
| Analysis | each condition vs vanilla | pre-registered P1/P2 (Holm) + interaction bounds, seed x problem bootstrap, boundary entry/exit | N4 is underpowered as a verdict at 3 seeds |

## What changed vs the Kaggle notebook (state this in the paper's setup section)

| Area | Kaggle notebook | This package | Why |
|---|---|---|---|
| Precision | 4-bit NF4 QLoRA | bf16 base + fp32 LoRA adapters | 0.5B fits easily on 24 GB; 4-bit generation is the slow path |
| Hardware | 2x T4, accelerate DDP | 1 GPU, plain PyTorch loop | removes the DDP grad-accum/scheduler bug class entirely |
| Batch / update | 32 prompts x 4 rollouts | 16 prompts x 8 rollouts (same 128 completions) | MEG needs >4 rollouts to resolve modes |
| On-policy update | dropout 0.05 made old != new log-probs | dropout 0, old = new.detach() | exact on-policy gradient; one forward pass saved |
| Scoring | re-tokenized text, bf16 log-softmax, whole batch in memory | exact sampled token ids, fp32 log-softmax, per-chunk backward | correct log-probs; identical gradient (tested) at a fraction of the memory |
| Sampling | Qwen defaults silently added top_k=20, repetition_penalty=1.05 | top_k off, repetition_penalty 1.0, explicit | sampling distribution fully specified by the config |
| Answer parsing | first `\boxed{...}`, regex broke nested braces | last balanced `\boxed{}`, nested braces OK | MATH-500 answers like `\frac{1}{2}` were truncated |
| Equivalence | sympy `eval` on raw model text, no timeout; `50%` -> 0.5 | whitelist + 2 s timeout; `%` stripped | no code-injection or hang risk; GSM8K percent answers score correctly |
| Final eval | 50 problems x 10 samples | fixed 500-problem subset x 32 samples, k up to 32, correctness matrices saved | Req 41.3; paired base/run comparison |

Unchanged from Kaggle: the CB-GRPO macro gate, group-relative advantages, PPO-clip objective,
cosine schedule with 10% warmup, LoRA r=16/alpha=32 on all projection layers.

## Extensions (registered in PREREGISTRATION.md, not yet run)

| What | How | Cost |
|---|---|---|
| 5 seeds per cell at 120 steps | `bash scripts/run_queue.sh queues/extension_seeds_0.5B.txt --benchmarks math500 --train-args "--steps 120 --lr 1e-5"` | ~75 GPU-h |
| 240 steps, Pass@k at steps 120/180/240 | `bash scripts/run_queue.sh queues/extension_240_0.5B.txt --benchmarks math500 --train-args "--lr 1e-5"` | ~150 GPU-h |

After the runs: `python analyze.py --benchmark math500` (the final model), `python analyze.py --eval_subdir eval_step120`
and `--eval_subdir eval_step180` for the trajectory, then `python scripts/power.py` and `python scripts/mechanism.py`.
`analyze.py` also reports whether each contrast's 90% bootstrap CI lies inside +-`--sesoi` (default 1 pp per ln k).
