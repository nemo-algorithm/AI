import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import bisect

env = gym.make("MountainCar-v0")
env.reset() # state를 반환

env._max_episode_steps = 100000

pos_space = np.arange(env.observation_space.low[0], env.observation_space.high[0], 0.1)
vel_space = np.arange(env.observation_space.low[1], env.observation_space.high[1], 0.01)

def get_state(name, value):

    if name == "pos":

        ret = bisect.bisect_left(pos_space, value - 1e-5)
        if ret > 0 and pos_space[ret] - value > value - pos_space[ret-1]:
            ret -= 1
        return ret

    if name == "vel":

        ret = bisect.bisect_left(vel_space, value - 1e-5)
        if ret > 0 and vel_space[ret] - value > value - vel_space[ret-1]:
            ret -= 1
        return ret

    return -1

def train(env, episodes, epsilon=0.8, min_epsilon=0.01, learning_rate=0.2, gamma=0.9, show=False):

    Q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    reward_history = []

    for episode in tqdm(range(episodes)):

        state = env.reset()

        pos = get_state("pos", state[0])
        vel = get_state("vel", state[1])
        reward = 0

        while True:

            # next action 고르기
            if np.random.random() < 1 - epsilon:
                action = Q[pos, vel].argmax()  # action_space
            else:
                action = np.random.randint(0, env.action_space.n)

            # 환경에 가하기
            state_next, reward_step, tr, _ = env.step(action)

            pos_next = get_state("pos", state_next[0])  # index
            vel_next = get_state("vel", state_next[1])  # index
            reward += reward_step

            if tr == True and state_next[0] >= 0.5:
                Q[pos, vel, action] = reward_step
                break

            Q[pos, vel, action] = (1 - learning_rate) * Q[pos, vel, action] + learning_rate * (reward_step + gamma * np.max(Q[pos_next, vel_next]))

            if tr == True:
                break

            pos, vel = pos_next, vel_next

        epsilon = max(min_epsilon, epsilon - 2 * (epsilon / episodes))
        reward_history.append(reward)

        if episode % 100 == 99 and show == True:
            print('episode : ', episode, ', reward : ', reward, ', epsilon : ', epsilon)

    return Q, reward_history

Q, reward_history = train(env, 100)
print(reward_history)
plt.plot(reward_history)
plt.show()

def rendering(env, Q, epsilon=0):

    state = env.reset()
    pos = get_state("pos", state[0])
    vel = get_state("vel", state[1])

    while True:

        if np.random.random() < 1 - epsilon:
            action = Q[pos, vel].argmax()  # action_space
        else:
            action = np.random.randint(0, env.action_space.n)

        # 환경에 가하기
        state_next, reward_step, tr, _ = env.step(action)
        env.render()

        pos_next = get_state("pos", state_next[0])  # index
        vel_next = get_state("vel", state_next[1])  # index

        if tr == True:
            break

        pos, vel = pos_next, vel_next

rendering(env, Q)