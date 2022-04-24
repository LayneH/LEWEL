# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    print("Loading {} (epoch {})".format(input, obj['epoch']))
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if not (k.startswith("module.encoder_q.backbone") or k.startswith("module.online_net.backbone")) or 'fc' in k:
            continue
        old_k = k
        k = k.replace("backbone.", "")
        k = k.replace("module.encoder_q.", "")
        k = k.replace("module.online_net.", "")
        print(old_k, "->", k)
        newmodel[k] = v

    with open(sys.argv[2], "wb") as f:
        torch.save(newmodel, f, _use_new_zipfile_serialization=False)
