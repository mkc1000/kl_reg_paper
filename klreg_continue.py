from klreg import BatchRunLLM, ChatEnv, init_prompt, get_model_and_tokenizer, get_sentiment_pipeline, PPO, run_episodes, RolloutBuffer

import asyncio
import torch
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import ast
import re

def read_lines_reverse(file):
    file.seek(0, 2)  # Move the file pointer to the end of the file
    position = file.tell()  # Get the current position (end of file)
    
    line = ''
    while position >= 0:
        file.seek(position)
        next_char = file.read(1)
        if next_char == '\n':
            yield line[::-1]
            line = ''
        else:
            line += next_char
        position -= 1
    
    if line:
        yield line[::-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, help='What is the name of the logfile for the run we are continuing?')
    parser.add_argument('--steps', type=int, help='How many timesteps to train for?')
    parser.add_argument('--chat_len', type=int, default=256, help='New max chat length (default 256)')
    args = parser.parse_args()

    var_dict = {} # populate from logfile
    logfile = "stdout/" + args.logfile
    with open(logfile, 'r') as file:
        lines = [next(file) for _ in range(100)]
        for line in lines:
            if line.startswith("save checkpoint path : "):
                checkpoint_path = line.split(" : ")[1].strip()
                break
        else:
            assert(False, "Checkpoint path not found")
        first_line = lines[0].strip()
        var_dict = ast.literal_eval(first_line)
        n_updates = None
        time_step = None
        pattern = re.compile(r"after (\d+) updates \(timestep (\d+)\), saving model at : ")
        for line in read_lines_reverse(file):
            match = pattern.search(line)
            if match:
                n_updates = int(match.group(1))
                time_step = int(match.group(2))
                break
        if n_updates is None:
            assert(False, "No match in logfile defining n_updates and time_step")
    
    var_dict['max_training_timesteps'] = time_step + args.steps
    var_dict['max_chat_length'] = args.chat_len
    locals().update(var_dict) # gives values to the variables commented out below
    # max_training_timesteps, max_chat_length, n_envs, env_temp, base_pol_temp, save_model_freq, update_timestep, K_epochs, update_batch_size, eps_clip, ent_coef, per_episode_kl_budget, gamma, max_grad_norm, lr_actor, lr_critic, random_seed, run_num_pretrained

    # redirect print to file
    file = open("stdout/cont_{}".format(args.logfile), 'w')
    sys.stdout = file
    sys.stderr = file

    print(var_dict)
    sys.stdout.flush()

    continue_prev_run = False
    env_name = "Chat"

    model, tokenizer = get_model_and_tokenizer()
    sentiment_pipeline = get_sentiment_pipeline()
    batchrunllm = BatchRunLLM(model, n_envs)
    envs = [ChatEnv(init_prompt, "Teacher", ["Student"], model, tokenizer, sentiment_pipeline, batchrunllm, base_pol_temp=base_pol_temp, env_temp=env_temp, max_chat_length=max_chat_length) for i in range(n_envs)]
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

    file_str = args.logfile[:-4] 

    log_dir = "PPO_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 1
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    # if continue_prev_run: # for some reason it overwrites, rather than adding to the end
    #     run_num -= 1
    log_f_name_chat = log_dir + file_str + "_cont.txt"

    directory = "PPO_checkpoints/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    new_checkpoint_path = checkpoint_path[:-4] + "_cont.pth"
    print("save checkpoint path : " + new_checkpoint_path)

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
    ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")
    sys.stdout.flush()

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    # training loop
    all_kls = []
    episode_kl_history = []
    episode_reward_quantiles = [[], [], [], [], []]
    episode_rewards = []
    while time_step <= max_training_timesteps:

        # training_kl_budget = per_episode_kl_budget * ((time_step+1)/train_steps_til_full_kl_budget)
        # if training_kl_budget > per_episode_kl_budget:
        #     training_kl_budget = per_episode_kl_budget
        training_kl_budget = per_episode_kl_budget
        outputs = asyncio.run(run_episodes(ppo_agent, envs, training_kl_budget, verbose=True))
        log_f_chat = open(log_f_name_chat,"a+")
        # alt kls
        # kl_lefts = ppo_agent.kl_budget_left.values()
        # for kl_left in kl_lefts:
        #     episode_kl_history.append(training_kl_budget - kl_left.cpu().item()) # what's gone wrong? Just use log ratios?
        for output in outputs:
            t, chat, rewards_by_token, kls_by_token, kls, log_ratios_by_token, log_ratios, speaking_record, base_pol_ents = output
            episode_kl_history.append(sum(log_ratios))
            time_step += t

            # logging
            log_f_chat.write(f"{n_updates}\n"+",".join(map(str, chat))+"\n"+ \
                        ",".join(map(str, kls_by_token))+"\n"+ \
                        ",".join(map(str, log_ratios_by_token))+"\n"+ \
                        ",".join(map(str, rewards_by_token))+"\n\n")
            log_f_chat.flush()
            all_kls += kls
            # episode_kl_history.append(sum(kls))
            episode_rewards.append(sum(rewards_by_token))
        log_f_chat.close()

        if time_step // update_timestep > n_updates:
            ppo_agent.update()
            n_updates += 1
            print("--------------------------------------------------------------------------------------------")
            print(f"after {n_updates} updates (timestep {time_step}), saving model at : " + new_checkpoint_path)
            ppo_agent.save(new_checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            sns.histplot(np.array(all_kls)+1e-15, log_scale=True).set(title='Distribution of per-token-KL-divergence')
            plt.savefig('figs/latest_kls_{}_cont.png'.format(file_str))
            plt.close()

            sns.scatterplot(x = list(range(len(episode_kl_history))), y = episode_kl_history, s=2).set(title='Per-episode KL')
            plt.savefig('figs/per_episode_kl_{}_cont.png'.format(file_str))
            plt.close()

            episode_rewards.sort()
            l = len(episode_rewards)
            indcs = [0, l//4, l//2, l-l//4, l-1]
            for i, ind in enumerate(indcs):
                episode_reward_quantiles[i].append(episode_rewards[ind])
            sns.lineplot(data=episode_reward_quantiles).set(title='Reward quantiles')
            plt.savefig('figs/latest_rewards_{}_cont.png'.format(file_str))
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