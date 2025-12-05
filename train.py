import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# === 하이퍼파라미터 (학습 설정값) ===
LEARNING_RATE = 0.0005  # 학습률 (너무 크면 발산, 작으면 느림)
GAMMA = 0.99            # 할인율 (미래의 보상을 얼마나 중요하게 여길지)
BUFFER_SIZE = 50000     # 기억 공간 크기
BATCH_SIZE = 64         # 한 번 학습할 때 꺼내 볼 데이터 수
EPSILON_START = 1.0     # 탐험 확률 (처음엔 100% 무작위 행동)
EPSILON_END = 0.01      # 최소 탐험 확률 (마지막엔 1%만 무작위)
EPSILON_DECAY = 0.995   # 탐험 확률 감소 비율

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. 신경망 정의 (AI의 뇌) ===
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 입력(8개: 좌표, 속도 등) -> 은닉층(64) -> 은닉층(64) -> 출력(4개: 엔진 행동)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === 2. 에이전트 정의 (행동 및 학습 담당) ===
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # 두 개의 네트워크 사용 (학습 안정화를 위한 DQN의 핵심 기술)
        self.q_network_local = QNetwork(state_size, action_size).to(device)
        self.q_network_target = QNetwork(state_size, action_size).to(device)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict()) # 동기화
        
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=BUFFER_SIZE) # 경험 저장소
        self.epsilon = EPSILON_START

    # 행동 결정 (Epsilon-Greedy)
    def act(self, state):
        # 무작위 탐험
        if random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        # 아는 대로 행동 (모델 예측)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_local.eval() # 평가 모드
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() # 다시 학습 모드
        return np.argmax(action_values.cpu().data.numpy())

    # 기억하기
    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        # 데이터가 충분히 모이면 학습 시작
        if len(self.memory) >= BATCH_SIZE:
            self.learn()

    # 학습하기 (여기가 핵심!)
    def learn(self):
        # 1. 랜덤하게 과거 경험 샘플링
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # 텐서 변환
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 2. Q함수 업데이트 수식 (Bellman Equation)
        # 현재 상태에서의 예상 점수 (Q_expected)
        q_expected = self.q_network_local(states).gather(1, actions)
        
        # 다음 상태에서의 최대 점수 (Q_target)
        # Target 네트워크를 사용하여 학습 안정성을 높임
        q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))

        # 3. 오차 계산 (MSE Loss)
        loss = F.mse_loss(q_expected, q_targets)

        # 4. 역전파 (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. 타겟 네트워크 부드럽게 업데이트 (Soft Update)
        self.soft_update(self.q_network_local, self.q_network_target)

    def soft_update(self, local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# === 3. 메인 실행 루프 ===
env = gym.make("LunarLander-v3", render_mode=None) # 학습 속도를 위해 화면 끔
agent = Agent(state_size=8, action_size=4)

print(f"Device: {device}")
print("----- 훈련 시작! (목표: 평균 200점) -----")

scores = []
scores_window = deque(maxlen=100) # 최근 100게임 점수 평균용

for episode in range(1, 2001): # 최대 2000판 진행
    state, info = env.reset()
    score = 0
    
    for t in range(1000):
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        
        if done or truncated:
            break
            
    # 에피소드 종료 후 처리
    scores_window.append(score)
    scores.append(score)
    
    # 엡실론(탐험 확률) 줄이기
    agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

    # 로그 출력
    if episode % 100 == 0:
        print(f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.2f}")
    
    # 평균 점수가 200점을 넘으면 조기 종료 및 저장
    if np.mean(scores_window) >= 200.0:
        print(f"\n축하합니다! 문제를 해결했습니다! (평균 점수: {np.mean(scores_window):.2f})")
        torch.save(agent.q_network_local.state_dict(), 'lunar_custom_model.pth')
        break

env.close()