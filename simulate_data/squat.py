import os
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mujoco


matplotlib.use("TkAgg")
plt.ion()  
os.environ["MUJOCO_GL"] = "egl"

env = gym.make("Humanoid-v5", render_mode="rgb_array")
model = env.unwrapped.model 
data = env.unwrapped.data

obs, info = env.reset()

print(f"현재 qpos 크기: {data.qpos.shape[0]}") 

squat_qpos = np.array([
    0, 0, 0.85,   # torso 위치 (z를 낮추지 않고 기본보다 약간만 유지)
    0.98, 0, 0.2, 0,  # torso 방향 (약간 앞으로 기울임)
    0, 0.3, 0,   # 척추 회전 (살짝 구부리기)
    
    -0.2, -0.3, -0.8, 1.0, 0.2, 0,  # 오른쪽 다리 (hip_x, hip_z, hip_y, knee, ankle_y, ankle_x)
    -0.2, -0.3, -0.8, 1.0, 0.2, 0,  # 왼쪽 다리 (hip_x, hip_z, hip_y, knee, ankle_y, ankle_x)
    
    0.3, -0.5  # 팔을 앞으로 뻗기 (squat 균형 잡기)
])


qpos_size = data.qpos.shape[0]

if len(squat_qpos) > qpos_size:
    squat_qpos = squat_qpos[:qpos_size]  
elif len(squat_qpos) < qpos_size:
    squat_qpos = np.pad(squat_qpos, (0, qpos_size - len(squat_qpos)), 'constant') 

data.qpos[:] = squat_qpos
data.qvel[:] = np.zeros_like(data.qvel)  

mujoco.mj_forward(model, data)
mujoco.mj_step1(model, data)

img = env.render()

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
plt.show(block=True) 

env.close()
