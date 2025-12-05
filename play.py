import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- (QNetwork 클래스는 위와 동일하게 정의해줘야 불러올 수 있습니다) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 1. 환경 및 모델 준비
env = gym.make("LunarLander-v3", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QNetwork(8, 4).to(device)
# 저장된 모델 불러오기
model.load_state_dict(torch.load('lunar_custom_model.pth', map_location=device, weights_only=True))
model.eval()

# 2. 실행
for i in range(5):
    state, _ = env.reset()
    score = 0
    while True:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = np.argmax(model(state_tensor).cpu().data.numpy())

        state, reward, done, truncated, _ = env.step(action)
        score += reward

        if done or truncated:
            print(f"Episode {i+1} Score: {score}")
            break

env.close()