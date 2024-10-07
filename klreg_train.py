from klreg import BatchRunLLM, ChatEnv, init_prompt, get_model_and_tokenizer, get_sentiment_pipeline, PPO, run_episodes, RolloutBuffer
from klreg_config import config_dicts

import asyncio
import torch
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='Which config id to use from the list defined in klreg_config; default 0')
    args = parser.parse_args()
    file_str = 'chat'

    id = 0
    if args.id is not None:
        id = args.id
    locals().update(config_dicts[id]) # gives values to the variables commented out below
    # max_training_timesteps, max_chat_length, n_envs, env_temp, base_pol_temp, save_model_freq, update_timestep, K_epochs, update_batch_size, eps_clip, ent_coef, per_episode_kl_budget, gamma, max_grad_norm, lr_actor, lr_critic, random_seed, run_num_pretrained

    # redirect print to file
    if not os.path.exists('stdout/'):
        os.makedirs('stdout/')
    file = open("stdout/{}_{}.txt".format(id, file_str), 'w')
    sys.stdout = file
    sys.stderr = file

    print(config_dicts[id])
    sys.stdout.flush()

    pretrain = False
    continue_prev_run = False
    env_name = "Chat"

    model, tokenizer = get_model_and_tokenizer()
    sentiment_pipeline = get_sentiment_pipeline()
    batchrunllm = BatchRunLLM(model, n_envs)
    envs = [ChatEnv(init_prompt, "Teacher", ["Student"], model, tokenizer, sentiment_pipeline, batchrunllm, base_pol_temp=base_pol_temp, env_temp=env_temp, max_chat_length=256) for i in range(n_envs)]
    env = envs[0]
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    ###################### interruption ######################
    stop_fname = "stop.txt"
    def interrupt_signal_recieved():
        return os.path.isfile(stop_fname)

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs/{}/".format(id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 1
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name_chat = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".txt"

    directory = "PPO_checkpoints/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}_{}.pth".format(env_name, id, random_seed, file_str)
    print("save checkpoint path : " + checkpoint_path)

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max chat length : ", max_chat_length)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    if update_batch_size is not None:
        print("PPO batch size :", update_batch_size)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, update_batch_size, eps_clip, ent_coef, max_grad_norm, gae_lam=None, dtype=torch.float32)
    if continue_prev_run:
        ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")
    sys.stdout.flush()

    if pretrain:
        for _ in range(steps_pretrain): # was 1000
            ppo_agent.load_random_buffer('saved_buffers')
            ppo_agent.update(override_Kepochs=2, override_kl_budget=0.1)

        print("Done pre-training at (GMT) : ", datetime.now().replace(microsecond=0))
        print("============================================================================================")
        sys.stdout.flush()

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    n_updates = 0
    # training loop
    all_kls = []
    episode_kl_history = []
    episode_reward_quantiles = [[], [], [], [], []]
    episode_rewards = []
    while time_step <= max_training_timesteps:

        training_kl_budget = per_episode_kl_budget * ((time_step+1)/train_steps_til_full_kl_budget)
        if training_kl_budget > per_episode_kl_budget:
            training_kl_budget = per_episode_kl_budget
        outputs = asyncio.run(run_episodes(ppo_agent, envs, training_kl_budget, verbose=True))
        log_f_chat = open(log_f_name_chat,"a+")
        
        for output in outputs:
            t, chat, rewards_by_token, kls_by_token, kls, log_ratios_by_token, log_ratios, speaking_record, base_pol_ents = output
            episode_kl_history.append(sum(log_ratios))
            time_step += t
            i_episode += 1

            # logging
            log_f_chat.write(f"{n_updates}\n"+",".join(map(str, chat))+"\n"+ \
                        ",".join(map(str, kls_by_token))+"\n"+ \
                        ",".join(map(str, log_ratios_by_token))+"\n"+ \
                        ",".join(map(str, rewards_by_token))+"\n\n")
            log_f_chat.flush()
            all_kls += kls
            episode_rewards.append(sum(rewards_by_token))
        log_f_chat.close()

        if time_step // update_timestep > n_updates:
            ppo_agent.update()
            n_updates += 1
            print("--------------------------------------------------------------------------------------------")
            print(f"after {n_updates} updates (timestep {time_step}), saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            sns.histplot(np.array(all_kls)+1e-15, log_scale=True).set(title='Distribution of per-token-KL-divergence')
            plt.savefig('figs/latest_kls_{}_{}.png'.format(id, file_str))
            plt.close()

            sns.scatterplot(x = list(range(len(episode_kl_history))), y = episode_kl_history, s=2).set(title='Per-episode KL')
            plt.savefig('figs/per_episode_kl_{}_{}.png'.format(id, file_str))
            plt.close()

            episode_rewards.sort()
            l = len(episode_rewards)
            indcs = [0, l//4, l//2, l-l//4, l-1]
            for i, ind in enumerate(indcs):
                episode_reward_quantiles[i].append(episode_rewards[ind])
            sns.lineplot(data=episode_reward_quantiles).set(title='Reward quantiles')
            plt.savefig('figs/latest_rewards_{}_{}.png'.format(id, file_str))
            plt.close()

            all_kls = []
            episode_rewards = []
            sys.stdout.flush()

        if interrupt_signal_recieved(): # the interrupt signal is a certain file with a certian name existing
            break

    for env in envs:
        env.close()




    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")