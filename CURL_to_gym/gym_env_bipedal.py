import gym
import numpy as np
import cv2
import torch
from gym.wrappers import PixelObservationWrapper
import argparse
import time
import utils
from video import VideoRecorder
import os
import json
from curl_sac import CurlSacAgent
from logger import Logger

from collections import deque





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=2, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=1000, type=int)
    args = parser.parse_args()
    return args


def video_save(video_,video_dir,shape_video,i):

    # print(video_.shape)
    video = video_
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_dir+'bipedal'+str(i) + '.avi', fourcc, 20.0, ((shape_video[2], shape_video[1])))

    for j in range(video.shape[0]):
        bgr_frame = cv2.cvtColor(video[j].astype('uint8'), cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()



def cropper_100x100(pixel):
    pixel_data = pixel[100:400,100:400,:]
    pixel_data = cv2.resize(pixel_data,(100,100))
    return pixel_data




def make_9x100x100(stack_frame,pixel_100):
    stack_frame.append(pixel_100)
    while len(stack_frame) < 3:
        stack_frame.append(pixel_100)

    return np.concatenate(stack_frame, axis=2) 



def evaluate(env, agent, video, num_episodes, L, step, args,stack_frame,max_eval_episode_steps):
    all_ep_rewards = []

    print('-------------evaluating Process---------------')
    def run_eval_loop(sample_stochastically=False):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            print(f'evaluating episode : {i}',end='\r')
            observation = env.reset()
            pixel = np.array(observation['pixels'])
            obs_frame = cropper_100x100(pixel)
            obs = make_9x100x100(stack_frame,obs_frame)
            obs = np.transpose(obs, (2, 0, 1))

            # video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            video_ = np.expand_dims(obs_frame,axis=0)
            shape_video = video_.shape
            for _ in range(max_eval_episode_steps):
                # # center crop image
                # if args.encoder_type == 'pixel':
                #     obs = utils.center_crop_image(obs,args.image_size)
                action= agent.sample_action(obs)
                # with utils.eval_mode(agent):
                #     if sample_stochastically:
                #         action = agent.sample_action(obs)
                #     else:
                #         action = agent.select_action(obs)
                observation, reward, done, _ = env.step(action)
                # save video first episode of each eval
                if i==0:
                    saving_pixel = observation['pixels']
                    saving_frame = cropper_100x100(saving_pixel)
                    video_ = np.concatenate([video_,np.expand_dims(saving_frame,axis=0)],axis=0)
                    video_save(video_,'./video/',shape_video,step)

                if done:
                    break


                episode_reward += reward

            # video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    print('-----------done evaluating Proces')



def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    else:
        assert 'agent is not supported: %s' % args.agent







def main():
    max_episode_steps=5000
    args = parse_args()
    # env = gym.make('CartPole-v1')
    # env = gym.make('Pendulum-v1')
    # env = gym.make('BipedalWalker-v3')

    env = gym.make(args.domain_name)
    frame_stack = deque(maxlen=args.frame_stack)




    env.reset()
    _ = env.render(mode='rgb_array')


    env = PixelObservationWrapper(env)

    env.seed(1)


    # make dictionary
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(device)
    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape


    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)



    episode, episode_reward, done = 0, 0, True
    start_time = time.time()




    
    episode_step = 0

    # Start Episode
    for step in range(args.num_train_steps):
        # evaluate agent periodically
        print(f'step : {step}',end='\r')
        if step % args.eval_freq ==0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args,frame_stack,300)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)





        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            observation = env.reset()
            pixel = np.array(observation['pixels'])
            obs_frame = cropper_100x100(pixel)
            obs = make_9x100x100(frame_stack,obs_frame)
            obs = np.transpose(obs, (2, 0, 1))
            # print(obs.shape)
            


            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)


        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()

        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_observation, reward, done, _ = env.step(action)
        next_pixel = next_observation['pixels']
        next_obs_frame = cropper_100x100(next_pixel)
        next_obs = make_9x100x100(frame_stack,next_obs_frame)
        next_obs = np.transpose(next_obs, (2, 0, 1))


        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1



        

    env.close()




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()


