#python 3.10.11 
#pip install gym

import gym
import numpy as np
import matplotlib.pyplot as plt
import os


# A first attemp to collect data from the CoinRun environment as indicated in the GENIE paper page 26.

# To do : It's gonna be the input for the tokenizer so we need to make sure those are matching.

class DataCollector:
    
    """ A class to collect data from the CoinRun environment."""

    """Array Observation: 
    Each element of the array is an observation, which is an image of the game state.
    In the case of the CoinRun environment, each observation is a 64x64 pixel image with 3 color channels (RGB).
    Format: (num_levels*num_steps, 64, 64, 3)."""

    def __init__(self, env_id, num_levels, num_steps):
        try:
            self.env = gym.make(env_id, num_levels=num_levels)
        except gym.error.Error as e:
            print(f"Error creating environment: {e}")
            raise

        self.num_levels = num_levels
        self.num_steps = num_steps
        self.observations = []
        self.rewards = []
        self.actions = []
        self.dones = []


    def collect_data(self):
        for level in range(self.num_levels):
            obs = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < self.num_steps:
                action = self.env.action_space.sample() 
                obs, reward, done, info = self.env.step(action)
                self.observations.append(obs)
                self.rewards.append(reward)
                self.actions.append(action)
                self.dones.append(done)
                steps += 1
                    
        return self.observations, self.rewards, self.actions, self.dones

    def save_data(self, filename):
        try:
            np.savez(filename, observations=self.observations, rewards=self.rewards, actions=self.actions, dones=self.dones)
        except Exception as e:
            print(f"Error saving data: {e}")
            raise


# To do : the class helped to visualize the data. we can improve it by making it by levels.
class ImageVisualizer:

    """A class  to save images by level in order to visualize the data."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_images(self, images, level):
        level_dir = os.path.join(self.output_dir, f'level_{level}')
        os.makedirs(level_dir, exist_ok=True)

        for i, img in enumerate(images):
            z= 0 
            while z <200 : 
                plt.imshow(img)
                plt.savefig(os.path.join(level_dir, f'image_{i}.png'))
                plt.close()
                z= z+1

