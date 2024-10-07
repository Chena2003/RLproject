import pickle

import gymnasium as gym
import numpy as np

# hyperparameters
H = 200  
batch_size = 10 
learning_rate = 1e-4
gamma = 0.99 
decay_rate = 0.99 
resume = True  
test = True  # 测试模式
save_file = 'reinforce_model.p'

if test == True:
    render = True
else:
    render = False

D = 80 * 80  
    model = pickle.load(open(save_file, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  


def prepro(I):
    I = I[35:195] 
    I = I[::2, ::2, 0]  
    I[I == 144] = 0  
    I[I == 109] = 0 
    I[I != 0] = 1  
    return I.astype(np.float32).ravel()


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0 
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  


def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


if render:
    env = gym.make("Pong-v0", render_mode='human', mode=0, difficulty=0)
else:
    env = gym.make("Pong-v0")
observation, _ = env.reset()
prev_x = None  
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = policy_forward(x)
    if test == True:
        action = 2 if aprob > 0.5 else 3
    else:
        action = 2 if np.random.uniform() < aprob else 3 

    xs.append(x)  
    hs.append(h)  
    y = 1 if action == 2 else 0  
    dlogps.append(y - aprob)  

    observation, reward, terminated, truncated, info = env.step(action)
    done = np.logical_or(terminated, truncated)
    reward_sum += reward

    drs.append(reward)  

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], [] 

        discounted_epr = discount_rewards(epr)

        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]  

        if episode_number % batch_size == 0 and test == False:
            for k, v in model.items():
                g = grad_buffer[k]  
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open(save_file, 'wb'))
        reward_sum = 0
        observation, _ = env.reset()  
        prev_x = None

    if reward != 0:  
        print('ep %d: game finished, reward: %f' % (episode_number, reward)), ('' if reward == -1 else ' !!!!!!!!')
