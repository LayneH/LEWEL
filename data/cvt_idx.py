# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

if __name__ == "__main__":
    train_file = "/mnt/lustre/share/data/images/meta/train.txt"
    idx_file = "data/10percent.txt"
    out_file = idx_file + ".ext"
    max_class = 1000

    with open(idx_file, "r") as fin, open(train_file, "r") as f_train:
        all_samples = {}
        idx_samples = []
        selected_samples = []
        for line in f_train.readlines():
            name, label = line.strip().split()
            label = int(label)
            if label < max_class:
                base_name = name.split("/")[1]
                all_samples[base_name] = (label, name)
        print(f"len of all samples: {len(all_samples)}")
        
        for line in fin.readlines():
            nm = line.strip()
            selected_samples.append(all_samples[nm])
        
        print(f"Len of selected samples {len(selected_samples)}")

    with open(out_file, "w") as fout:
        for (lb, nm) in selected_samples:
            fout.write(f"{lb} {nm}\n")
