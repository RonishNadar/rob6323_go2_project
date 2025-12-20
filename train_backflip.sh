#!/usr/bin/env bash
# BONUS TASK: Train Backflip with Recovery for Go2

ssh -o StrictHostKeyChecking=accept-new burst "cd ~/rob6323_go2_project && sbatch --job-name='backflip_${USER}' --mail-user='${USER}@nyu.edu' train_backflip.slurm '$@'"

