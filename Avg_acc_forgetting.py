"""
Compute Average Forgetting from a continual/federated learning log file.

Forgetting for task i  =  acc(task i, last round of training on task i)
                        -  acc(task i, last round of training on last task)

Average Forgetting = mean of forgetting over all tasks except the last one.
"""

import re
import sys

LOG_PATH = "res_cifar10.log"

# ── Parse log ────────────────────────────────────────────────────────────────
# Lines look like:
#   Mean accuracy up to task 3: 10.65 % Task accuracies: [10.1, 7.15, 14.7]
PATTERN = re.compile(
    r"Mean accuracy up to task (\d+):\s+[\d.]+\s+%\s+Task accuracies:\s+\[([^\]]+)\]"
)

# For each "training phase" (= each task being trained), collect all rounds.
# We track:   records[current_training_task]  ->  list of per-round accuracy lists
records = {}   # { training_task_idx (0-based): [ [acc_t0, acc_t1, ...], ... ] }

with open(LOG_PATH, "r", errors="replace") as f:
    for line in f:
        # Strip carriage-returns (log uses \r for progress overwriting)
        line = line.replace("\r", "\n")
        for part in line.split("\n"):
            m = PATTERN.search(part)
            if m:
                current_task = int(m.group(1)) - 1   # 0-based
                accs = [float(x.strip()) for x in m.group(2).split(",")]
                records.setdefault(current_task, []).append(accs)

if not records:
    sys.exit("No accuracy lines found – check the log path.")

num_tasks = max(records.keys()) + 1
print(f"Detected {num_tasks} tasks.\n")

# ── Extract the key accuracy snapshots ───────────────────────────────────────
# For each task t, we need:
#   a) acc_own[t]  : accuracy on task t at the LAST round while training on task t
#   b) acc_final[t]: accuracy on task t at the LAST round while training on the LAST task

last_task_idx = num_tasks - 1

acc_own   = {}   # acc_own[t]   = acc of task t at end of training phase t
acc_final = {}   # acc_final[t] = acc of task t at end of training phase (last_task)

for t, round_list in records.items():
    # last round of this training phase
    last_round_accs = round_list[-1]
    # accuracy on task t itself (index t in the accuracy list)
    if t < len(last_round_accs):
        acc_own[t] = last_round_accs[t]

# Final training phase
if last_task_idx in records:
    final_phase_last_round = records[last_task_idx][-1]
    for t in range(num_tasks):
        if t < len(final_phase_last_round):
            acc_final[t] = final_phase_last_round[t]

# ── Compute forgetting ───────────────────────────────────────────────────────
print(f"{'Task':>6}  {'acc (own phase)':>16}  {'acc (final phase)':>18}  {'Forgetting':>11}")
print("-" * 58)

forgetting_values = []
for t in range(num_tasks - 1):   # exclude the last task
    if t not in acc_own or t not in acc_final:
        print(f"  {t+1:>4}  {'N/A':>16}  {'N/A':>18}  {'N/A':>11}")
        continue
    f_t = acc_own[t] - acc_final[t]
    forgetting_values.append(f_t)
    print(f"  {t+1:>4}  {acc_own[t]:>15.2f}%  {acc_final[t]:>17.2f}%  {f_t:>10.2f}%")

if forgetting_values:
    avg_forgetting = sum(forgetting_values) / len(forgetting_values)
    print("-" * 58)
    print(f"\nAverage Forgetting = {avg_forgetting:.4f}%")
else:
    print("Could not compute forgetting (not enough tasks).")

# ── Final Accuracy ────────────────────────────────────────────────────────────
# Final accuracy for each task = acc of that task at the end of the last training phase
print(f"\n{'Task':>6}  {'Final Accuracy':>15}")
print("-" * 25)
final_accs = []
for t in range(num_tasks):
    if t in acc_final:
        final_accs.append(acc_final[t])
        print(f"  {t+1:>4}  {acc_final[t]:>14.2f}%")
    else:
        print(f"  {t+1:>4}  {'N/A':>14}")

if final_accs:
    avg_final = sum(final_accs) / len(final_accs)
    print("-" * 25)
    print(f"\nAverage Final Accuracy = {avg_final:.4f}%")
