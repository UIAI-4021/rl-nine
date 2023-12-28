import gym
import gym_maze
import numpy as np


if __name__ == '__main__':
    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    space_size = 10
    action_size = env.action_space.n
    xy_observation = env.reset()
    observation = int(xy_observation[0] * space_size + xy_observation[1])

    q_table = np.zeros((space_size * space_size, action_size))

    # Define the maximum number of iterations
    NUM_EPISODES = 100

    # learning rate
    alfa = 0.8
    max_steps = 99
    # discount factor
    gamma = 0.95

    for episode in range(NUM_EPISODES):
        env.render()
        xy_state = env.reset()
        state = int(xy_state[0] * space_size + xy_state[1])

        for step in range(max_steps):

            action = np.argmax(q_table[state])

            xy_new_state, reward, done, info = env.step(action)

            new_state = int(xy_new_state[0] * space_size + xy_new_state[1])

            q_table[state][action] = (1 - alfa) * q_table[state][action] + alfa * (reward + gamma * max(q_table[new_state]))

            state = new_state

            if done:
                break
    # Close the environment
    env.close()