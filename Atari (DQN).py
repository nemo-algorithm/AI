import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy as c
import wandb
wandb.init(project="dqn-atari-breakout", name="0901-min_eps=0.05_epsilon_step=1e-8_learing_rate=1e-3_replay=1e5_batch=128_106", monitor_gym=True)
from tqdm import tqdm
import IPython.display as display
import os

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("BreakoutDeterministic-v4")
state = env.reset()
Height = 84
Width = 84


def preprocessing(image):

    image = image[30:-17, 7:-7, :]
    image = Image.fromarray(image)
    image = image.resize((84, 84))
    gray_filter = np.array([0.299, 0.587, 0.114])
    image = np.einsum("...i,i->...", image, gray_filter)
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
        self.history[0:4, :, :] = c(self.history[1:5, :, :])
        self.history[4] = c(x)


class DATA():

    def __init__(self, state, action, reward, done):
        self.state = c(state)  # np array: 5 by 84 by 84
        self.action = c(action)
        self.reward = c(reward)
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
            pass

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


def train(env, episodes, learning_rate=0.001, epsilon=1.0, gamma=0.99, min_epsilon=0.05, epsilon_step=1e-8,
          reset=False, replay_capacity=100000):
    main_cnn = CNN((Height, Width, 4), 4).to(device)  # Q initialization
    target_cnn = CNN((Height, Width, 4), 4).to(device)
    target_cnn.load_state_dict(main_cnn.state_dict())
    target_cnn.eval()

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.RMSprop(main_cnn.parameters(), lr=learning_rate)

    if reset == False:
        try:
            main_cnn.load_state_dict(torch.load("cnn.pkl"))
            target_cnn.load_state_dict(main_cnn.state_dict())
            optimizer.load_state_dict(torch.load("optimizer.pkl"))
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

        while True:

            state = c(history.history[1:])

            # Choose Action
            if np.random.random() < 1 - epsilon:
                action = target_cnn(tensor(state)).to("cpu")
                action = torch.argmax(action).item()
            else:
                action = np.random.randint(0, 4)

            epsilon = max(min_epsilon, epsilon - epsilon_step)

            # Step
            step += 1
            state_next, reward_step, done, info = env.step(action)
            state_next = preprocessing(state_next)
            history.update(state_next)

            reward += reward_step
            replay_memory.update(DATA(history.history, action, reward_step, done))

            if step >= replay_capacity and step % 10 == 0:
                main_cnn.train()
                states, actions, rewards_step, states_next, dones = replay_memory.sample(128)

                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards_step = torch.from_numpy(rewards_step).float().to(device)
                states_next = torch.from_numpy(states_next).float().to(device)
                dones = torch.from_numpy(dones).to(device)

                Q_main = torch.sum(main_cnn(states) * F.one_hot(actions, 4), dim=-1) # main: for training

                with torch.no_grad():
                    Q_target = rewards_step + gamma * torch.max(target_cnn(states_next), dim=-1)[0].detach()

                optimizer.zero_grad()

                loss = criterion(Q_main, Q_target)
                loss.backward()
                optimizer.step()
                wandb.log({"Loss": loss.to("cpu").item()})

            if step % 10000 == 0:
                target_cnn.load_state_dict(main_cnn.state_dict())

            if done:
                break

        if episode % 300 == 0:
            display.clear_output()
            print(step, episode)
            plt.title("reward_history, episode : {} epsilon : {}".format(episode, epsilon))
            plt.plot(reward_history)
            plt.show()

        reward_history.append(reward)
        count_action_history.append(count_action)
        wandb.log({"Reward": reward, "Step": episode, "epsilon": epsilon, "step": step})

        if episode % 1000 == 0:
            torch.save(target_cnn.state_dict(), "cnn.pkl")
            torch.save(optimizer.state_dict(), "optimizer.pkl")
            torch.save(target_cnn.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, 'optimizer.pt'))

    return cnn, reward_history


cnn, reward_history = train(env, 1000000, epsilon=1.0, reset=True)