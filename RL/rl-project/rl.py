import random

import gym
import gym_maze
import numpy as np


def q_learning(env):

    action_size = env.action_space.n
    state_size = 10 * 10

    q_table = np.zeros((state_size, action_size))

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
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)

            if exp_exp_tradeoff > epsilon:
                action = np.argmax(q_table[state])

            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            q_table[state, action] = (1 - alfa) * q_table[state, action] + alfa * (reward + gamma * max(q_table[new_state, :]))

            total_rewards += reward

            state = new_state

            if done:
                break
        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)

    return q_table


if __name__ == '__main__':
    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    print(q_learning(env))

    # Define the maximum number of iterations
    NUM_EPISODES = 1000

    # for episode in range(NUM_EPISODES):
    #
    #     env.render()
    #
    #     # TODO: Implement the agent policy here
    #     # Note: .sample() is used to sample random action from the environment's action space
    #
    #     # Choose an action (Replace this random action with your agent's policy)
    #     action = env.action_space.sample()
    #
    #     # Perform the action and receive feedback from the environment
    #     next_state, reward, done, truncated = env.step(action)
    #
    #     if done or truncated:
    #         observation = env.reset()
    #
    # # Close the environment
    # env.close()