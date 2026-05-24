import torch
from pathlib import Path
for p in sorted(Path("weight").glob("*.pth")):
    if "ffdnet" in p.name.lower():
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        body_weights = [k for k in state.keys() if k.startswith("body.") and ".weight" in k]
        body_bias = [k for k in state.keys() if k.startswith("body.") and ".bias" in k]
        print(f"{p.name}:")
        print(f"  body weights: {len(body_weights)}, body bias: {len(body_bias)}")
        for k in sorted(state.keys()):
            if "body.41" in k or "body.42" in k or "body.43" in k or "body.44" in k or "body.45" in k or "body.46" in k or "body.47" in k:
                print(f"  {k}: {state[k].shape}")

