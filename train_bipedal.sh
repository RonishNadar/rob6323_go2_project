#!/usr/bin/env bash
# BONUS TASK: Train Bipedal Walking for Go2
# Reference: https://arxiv.org/pdf/2509.00215v2 (DMO Paper)

ssh -o StrictHostKeyChecking=accept-new burst "cd ~/rob6323_go2_project && sbatch --job-name='bipedal_${USER}' --mail-user='${USER}@nyu.edu' train_bipedal.slurm '$@'"


