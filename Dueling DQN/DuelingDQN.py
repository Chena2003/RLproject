import torch
import torch.nn as nn
import collections
from collections import deque
import numpy as np
import gym
import random
from net import *
import time
import cv2
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enabled = False
BATCH_SIZE = 32
LR = 0.0001
START_EPSILON = 1
FINAL_EPSILON = 0.05
EPSILON = START_EPSILON
EXPLORE = 100000
GAMMA = 0.99
TOTAL_EPISODES = 1000000
MEMORY_SIZE = 10000
MEMORY_THRESHOLD = 10000
TEST_FREQUENCY = 50

test = False # test为True，展示模型；否则训练模型
render = True if test else False

if render:
    env = gym.make('PongDeterministic-v4', render_mode='human')
else:
    env = gym.make('PongDeterministic-v4')
env = env.unwrapped  #打开限制操作
ACTIONS_SIZE = env.action_space.n #0，1不动；2，4向上；3，5向下


# net = CnnDQN([4,84,84],ACTIONS_SIZE).to(device)
# net_tar = CnnDQN([4,84,84],ACTIONS_SIZE).to(device)
net = DuelingNet([4,84,84],ACTIONS_SIZE).to(device)
net_tar = DuelingNet([4,84,84],ACTIONS_SIZE).to(device)
#  模型加载
net.load_state_dict(torch.load('.dueling_dqn.pkl', map_location=torch.device(device)))
net_tar.load_state_dict(torch.load('.dueling_dqn_target.pkl', map_location=torch.device(device)))

class Agent(object):
    def __init__(self):
        # self.network = AtariNet(ACTIONS_SIZE).to(device)
        self.network = net
        self.tar_net = net_tar

        self.network.eval()
        self.tar_net.eval()

        self.memory = deque() #创建了一个双向列队
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.num = 0
        self.epsilon = EPSILON
        self.learn_step_counter = 0
        self._obs_buffer = collections.deque(maxlen=2)


    def action(self, state, israndom):
		if not test:
			 if self.epsilon > FINAL_EPSILON:
				 self.epsilon -= (START_EPSILON - FINAL_EPSILON) / EXPLORE
			
			 if israndom and random.random() < self.epsilon:
				 return np.random.randint(0, ACTIONS_SIZE)

        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0) #给state加一个batch_size的维度，此时batch_size为1 shape(1,1,84,84)
        actions_value = self.network.forward(state.to(device))
        return torch.max(actions_value, 1)[1].data.to('cpu').numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        if reward==0:
            self.memory.append((state, action, reward, next_state, 1))
        else:
            self.memory.append((state, action, reward, next_state, 0))

        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()   #扔掉左边的数据
        if len(self.memory) < MEMORY_THRESHOLD: #小于MEMORY_THRESHOLD的时候不更新网络
            return
        self.num += 1
        # print(self.num)
        if self.num < BATCH_SIZE:
            return
        if self.learn_step_counter % 1000 ==0:
            self.tar_net.load_state_dict(self.network.state_dict())
        self.learn_step_counter +=1
        batch = random.sample(self.memory, BATCH_SIZE)
        state = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
        action = torch.LongTensor(np.array([[x[1]] for x in batch])).to(device)
        reward = torch.FloatTensor(np.array([[x[2]] for x in batch])).to(device)
        next_state = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
        done = torch.FloatTensor(np.array([[x[4]] for x in batch])).to(device)

        eval_q = self.network.forward(state).gather(1, action)
        actions_value = self.network.forward(next_state)
        next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)
        next_q = self.tar_net.forward(next_state).gather(1, next_action)
        target_q = reward + GAMMA * next_q * done

        loss = self.loss_func(eval_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_models(self,episode):
        torch.save(self.network.state_dict(), './dueling_dqn.pkl')
        torch.save(self.tar_net.state_dict(), './dueling_dqn_target.pkl')
        print('=====================')
        print('%d episode model has been save...' %(episode))
        
# 图像预处理
def preprocess(observation):
    img = np.reshape(observation, [210, 160, 3]).astype(np.float32)
    # RGB转换成灰度图像的一个常用公式是：ray = R*0.299 + G*0.587 + B*0.114
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114  # shape (210,160)
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)  # shape(110,84)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    x_t.astype((np.uint8))
    x_t = np.moveaxis(x_t, 2, 0)  # shape（1，84，84）
    return np.array(x_t).astype(np.float32) / 255.0

agent = Agent()
best_reward = 0
mean_test = collections.deque(maxlen=100)
f = open('./model/double_dqn_reward.txt','a+')
for i_episode in range(1000):

    state, _ = env.reset()
    state = preprocess(state)
    state = np.reshape(state, (84, 84))
    state_shadow = np.stack((state,state,state,state),axis=0)
    total_reward = 0
    while True:
        if render:
            env.render()
        action = agent.action(state_shadow, True)
        next_state, reward, terminated, truncated, info = env.step(action)
        reward_real = reward
        next_state = preprocess(next_state)
        next_state_shadow = np.append( next_state, state_shadow[:3,:,:],axis=0)

        done = np.logical_or(terminated, truncated)
        agent.learn(state_shadow, action, reward, next_state_shadow, done)
        state_shadow = next_state_shadow
        total_reward += reward_real
        if done:
            break

    print('episode: {} , total_reward: {}'.format(i_episode, round(total_reward, 3)))

     f.write(str(total_reward))
     f.write('\n')
    
     if i_episode % TEST_FREQUENCY == 0:
         agent.save_models(i_episode)
 f.close()
