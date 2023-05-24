import gym
import numpy as np
import cv2
import torch
from gym.wrappers import PixelObservationWrapper

# env = gym.make('CartPole-v1')
env = gym.make('Pendulum-v1')
# env = gym.make('BipedalWalker-v3')

video_dir ='./video/'

env.reset()
_ = env.render(mode='rgb_array')
env.seed(1)

env = PixelObservationWrapper(env)
SAVING_IDX = 10
MAX_STEPS = 300
MAX_EPISODE = 1000000


for i_episode in range(MAX_EPISODE):
    observation = env.reset()
    pixel_data = observation['pixels'][100:400,100:400,:]
    pixel_data = cv2.resize(pixel_data,(100,100))
    video_ = np.expand_dims(pixel_data,axis=0)
    shape_video = video_.shape

    print(f'Episode :{i_episode}')
    step=0
    for _ in range(MAX_STEPS):
        step+=1
        print(f'STEP : {step}',end='\r')
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        pixel_data = observation['pixels'][100:400,100:400,:]
        pixel_data = cv2.resize(pixel_data,(100,100))
        
        video_ = np.concatenate([video_,np.expand_dims(pixel_data,axis=0)],axis=0)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break

    if  i_episode % SAVING_IDX == 0:
        print(video_.shape)
        video = video_
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_dir+'output_' + str(i_episode) + '.avi', fourcc, 20.0, ((100,100)))

        for j in range(video.shape[0]):
            bgr_frame = cv2.cvtColor(video[j].astype('uint8'), cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()

env.close()

