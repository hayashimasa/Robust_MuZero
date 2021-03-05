# Bakr Ouairem Main PacMan File


import gym
env = gym.make("PacMan:phypacman-v0")
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(env.action_dict[action])
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break
env.close()
