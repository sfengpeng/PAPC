from point_sam.build_model import build_point_sam
import numpy as np
import torch
from utils import load_ply

# load model
ckpt_path = "../../pretrained_model/model.safetensors"
model = build_point_sam(ckpt_path, 512, 64).cuda()  # (ckpt_path, num_centers, KNN size)
NUM_PROMPTS = 10

# load point cloud
points = load_ply("../assets/eyeglass_1.ply")
xyz = points[:, :3]  # (n_points, 3)
rgb = points[:, 3:6]  # (n_points, 3)
xyz = torch.tensor(xyz).unsqueeze(0).cuda().float()  # (batch_size, n_points, 3)
rgb = torch.tensor(rgb).unsqueeze(0).cuda().float()  # (batch_size, n_points, 3)

# add single prompt
prompt_coords = xyz[:, np.random.choice(xyz.shape[1], 1)]  # (batch_size, 1, 3)
prompt_labels = torch.tensor([[1]])  # (batch_size, 1)
model.set_pointcloud(xyz, rgb)
model.predict_masks(prompt_coords, prompt_labels)

# add multiple prompts
prompt_points = []
prompt_labels = []
for i in range(NUM_PROMPTS):
    prompt_point = xyz[:, np.random.choice(xyz.shape[1], 1)]  # (batch_size, 1, 3)
    prompt_label = torch.tensor([[1]])  # (batch_size, 1)
    prompt_points.append(prompt_point)
    prompt_labels.append(prompt_label)

prompt_points = torch.cat(prompt_points, dim=1)
prompt_labels = torch.cat(prompt_labels, dim=1)

model.set_pointcloud(xyz, rgb)
model.predict_masks(prompt_points, prompt_labels)
