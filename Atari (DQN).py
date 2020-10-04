import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy as c
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import IPython.display as display
import os

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = 'BreakoutDeterministic-v4'
run_name = 'dqn_basic_{}'.format(env_name)

Height = 84
Width = 84

episodes = 50000
start_epsilon = 1.0
min_epsilon = 0.05
epsilon_step=1e-6
replay_capacity = 30000
learning_rate = 0.0001
gamma = 0.99

wandb.init(project='dqn', name=run_name, config={
    'env_name': env_name,
    'Height': Height,
    'Width': Width,
    'episodes': episodes,
    'start_epsilon': start_epsilon,
    'min_epsilon': min_epsilon,
    'replay_capacity': replay_capacity,
    'learning_rate': learning_rate,
    'gamma': 0.99
})

env = gym.make(env_name)
state = env.reset()

def preprocessing(image):

    image = Image.fromarray(image)
    image = image.resize((Height, Width))
    image = np.array(image.convert('L'), dtype=np.float32)
    image = image * 2 / 255 - 1

    return image


def calculate_Conv2d_dimension(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1


def calculate_MaxPool2d_dimension(input_size, max_pool):
    return input_size // max_pool


class CNN(nn.Module):

    def __init__(self, input_size, output_size):  # input size는 3차원

        super(CNN, self).__init__()

        calculate_H, calculate_W, channel = input_size

        calculate_H, calculate_W = [calculate_Conv2d_dimension(x, 8, 4, 0) for x in (calculate_H, calculate_W)]
        calculate_H, calculate_W = [calculate_Conv2d_dimension(x, 4, 2, 0) for x in (calculate_H, calculate_W)]
        calculate_H, calculate_W = [calculate_Conv2d_dimension(x, 3, 1, 0) for x in (calculate_H, calculate_W)]

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=calculate_H * calculate_W * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_size)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        return self.layers(x)


class HISTORY:

    def __init__(self, H, W):
        self.history = np.zeros((5, H, W))

    def start(self, x):
        for i in range(5):
            self.history[i] = c(x)

    def update(self, x):
        self.history[0:4, :, :] = c(self.history[1:5, :, :])  # TODO maybe bottleneck
        self.history[4] = c(x)


class DATA():

    def __init__(self, state, action, reward, done):
        self.state = np.float16(state)  # np array: 5 by 84 by 84
        self.action = c(action)
        self.reward = np.float16(reward)
        self.done = c(done)


class REPLAY_MEMORY():

    def __init__(self, capacity):

        self.replay = []
        self.capacity = c(capacity)
        self.time = 0

    def update(self, x):  # Confirmed

        if len(self.replay) < self.capacity:
            self.replay.append(c(x))

        else:
            pass  # TODO?

        self.replay[self.time] = c(x)
        self.time = (self.time + 1) % self.capacity

    def sample(self, sample_size):

        assert sample_size <= len(self.replay), "Error !! sample_size > length or capacity"

        sample_data = np.random.choice(self.replay, size=sample_size, replace=False)

        states = np.zeros((sample_size, 4, 84, 84))
        actions = np.zeros((sample_size), dtype=np.int64)
        rewards_step = np.zeros((sample_size))
        states_next = np.zeros((sample_size, 4, 84, 84))
        dones = np.zeros((sample_size))

        for i in range(sample_size):
            states[i] = sample_data[i].state[:4]
            actions[i] = sample_data[i].action
            rewards_step[i] = sample_data[i].reward
            states_next[i] = sample_data[i].state[1:]
            dones[i] = sample_data[i].done

        return states, actions, rewards_step, states_next, dones


def tensor(np_array):
    return torch.from_numpy(np_array).float().to(device)


def train(env, episodes, learning_rate=0.0001, epsilon=1.0, gamma=0.99, min_epsilon=0.05, epsilon_step=1e-6,
          reset=False, replay_capacity = 100000):
    main_cnn = CNN((Height, Width, 4), 4).to(device)  # Q initialization
    target_cnn = CNN((Height, Width, 4), 4).to(device)
    target_cnn.load_state_dict(main_cnn.state_dict())
    target_cnn.eval()

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.RMSprop(main_cnn.parameters(), lr=learning_rate)

    if reset == False:
        try:
            main_cnn.load_state_dict(torch.load('{}_cnn.pkl'.format(run_name)))
            target_cnn.load_state_dict(main_cnn.state_dict())
            optimizer.load_state_dict(torch.load('{}_optimizer'.format(run_name)))
        except:
            pass

    wandb.watch(main_cnn)

    step = 0
    history = HISTORY(Height, Width)
    replay_memory = REPLAY_MEMORY(replay_capacity)

    reward_history = []
    count_action_history = []

    for episode in tqdm(range(episodes)):

        state = env.reset()
        state = preprocessing(state)
        history.start(state)
        reward = 0

        count_action = [0, 0, 0, 0]

        while True:

            state = c(history.history[1:])

            # Choose Action
            if np.random.random() < 1 - epsilon:
                action = target_cnn(tensor(state)).to("cpu")
                action = torch.argmax(action).item()
            else:
                action = np.random.randint(0, 4)

            count_action[action] += 1
            epsilon = max(min_epsilon, epsilon - epsilon_step)

            # Step
            step += 1
            state_next, reward_step, done, info = env.step(action)
            state_next = preprocessing(state_next)
            history.update(state_next)

            reward += reward_step
            replay_memory.update(DATA(history.history, action, reward_step, int(done)))

            if step >= replay_capacity//2 and step % 4 == 0:

                main_cnn.train()
                states, actions, rewards_step, states_next, dones = replay_memory.sample(128)

                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards_step = torch.from_numpy(rewards_step).float().to(device)
                states_next = torch.from_numpy(states_next).float().to(device)
                dones = torch.from_numpy(dones).float().to(device)

                Q_main = torch.sum(main_cnn(states) * F.one_hot(actions, 4), dim=-1) # main: for training

                with torch.no_grad():
                    Q_target = rewards_step + (1-dones) * gamma * torch.max(target_cnn(states_next), dim=-1)[0].detach()

                optimizer.zero_grad()

                loss = criterion(Q_main, Q_target)
                loss.backward()
                optimizer.step()

            if step % 10000 == 0:
                target_cnn.load_state_dict(main_cnn.state_dict())

            if done:
                break

        if episode % 100 == 0:

            # display.clear_output()
            print(step, episode)
            plt.title("reward_history, episode: {} epsilon: {}".format(episode, epsilon))
            plt.plot(reward_history)
            plt.show()

        reward_history.append(reward)
        count_action_history.append(count_action)
        wandb.log({"Reward": reward, "episode": episode, "epsilon": epsilon, "step": step})

        if episode % 1000 == 0:

            torch.save(target_cnn.state_dict(), '{}_cnn.pkl'.format(run_name))
            torch.save(optimizer.state_dict(), '{}_cnn.pkl'.format(run_name))
            torch.save(target_cnn.state_dict(), os.path.join(wandb.run.dir, '{}_model.pt'.format(run_name)))
            torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, '{}_optimizer.pt'.format(run_name)))

    return cnn, reward_history

cnn, reward_history = train(env,
                            episodes,
                            learning_rate=learning_rate,
                            epsilon=start_epsilon,
                            gamma=gamma,
                            min_epsilon=min_epsilon,
                            epsilon_step=epsilon_step,
                            reset=True,
                            replay_capacity=replay_capacity
                            )