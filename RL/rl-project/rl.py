import random

import gym
import gym_maze
import numpy as np


def q_learning(env):

    action_size = env.action_space.n
    state_size = 10

    q_table = np.zeros((state_size * state_size, action_size))

    total_episode = 10000
    # learning rate
    alfa = 0.8
    max_steps = 99
    # discount factor
    gamma = 0.95

    # Exploration parameters
    # Exploration rate
    epsilon = 1.0
    max_epsilon = 1
    min_epsilon = 0.01
    decay_rate = 0.01

    rewards = []

    for episode in range(total_episode):
        xy_state = env.reset()
        state = int(xy_state[0] * state_size + xy_state[1])
        total_rewards = 0

        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)

            if exp_exp_tradeoff > epsilon:
                action = np.argmax(q_table[state])

            else:
                action = env.action_space.sample()

            xy_new_state, reward, done, info = env.step(action)

            new_state = int(xy_new_state[0] * state_size + xy_new_state[1])

            q_table[state][action] = (1 - alfa) * q_table[state][action] + alfa * (reward + gamma * max(q_table[new_state]))

            total_rewards += reward

            state = new_state

            if done:
                break

        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)

    policy = np.zeros((state_size * state_size), dtype=int)
    for i in range(action_size * action_size):
        policy[i] = np.argmax(q_table[i])

    return policy


if __name__ == '__main__':
    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    space_size = 10
    xy_observation = env.reset()
    observation = int(xy_observation[0] * space_size + xy_observation[1])

    # Define the maximum number of iterations
    NUM_EPISODES = 1000

    policy = q_learning(env)

    for episode in range(NUM_EPISODES):

        action = policy[observation]

        # Perform the action and receive feedback from the environment
        xy_next_state, reward, done, truncated = env.step(action)

        observation = int(xy_next_state[0] * space_size + xy_next_state[1])

        if done or truncated:
            observation = env.reset()

        env.render()

    # Close the environment
    env.close()