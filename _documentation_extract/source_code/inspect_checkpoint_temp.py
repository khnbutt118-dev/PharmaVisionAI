import torch
p = "models/combined_binary/best_model.pth"
c = torch.load(p, map_location="cpu", weights_only=False)
for k, v in c.items():
    if k == "model_state_dict":
        print(k, "contains", len(v), "parameters")
    else:
        print(k, "=", v)
