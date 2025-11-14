#!/usr/bin/env python3
import csv

with open("results/grid_search/GAN_cifar10/wgan-gp-in/missing_lr.csv") as f:
    reader = csv.DictReader(f)
    groups = []
    for row in reader:
        lx = row["lr_x"]
        ly = row["lr_y"]
        if lx == "N/A" or ly == "N/A":
            continue
        groups.append(f"'optimizers.lr_x={lx} optimizers.lr_y={ly}'")

cmd = "python main.py -m " + " ".join(groups)
print(cmd)
