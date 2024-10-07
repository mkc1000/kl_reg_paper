config_option_dict = { 
    "max_training_timesteps": int(3e6), 
    "train_steps_til_full_kl_budget": int(3e6),
    "max_chat_length": 256,
    "n_envs": 64,
    "env_temp": 0.05,
    "base_pol_temp": 1.0,

    "save_model_freq": int(1e3),      # save model frequency (in num timesteps)

    "update_timestep": 8000,      # update policy every n timesteps
    "K_epochs": 8,               # update policy for K epochs
    "update_batch_size": int(2**13),     # batch size when updating policy
    "eps_clip": 0.1,              # clip parameter for PPO
    "ent_coef": 0.0001,           # entropy coefficient
    "per_episode_kl_budget": [10., 10., 10., 20., 20., 20.],
    "gamma": 1.0,                # discount factor
    "max_grad_norm": 0.1,

    "steps_pretrain": 0,

    "lr_actor": 2e-5,      # learning rate for actor network # last tried at 4e-4 with deterministic kl penalty
    "lr_critic": 1e-4,      # learning rate for critic network

    "random_seed": 0,         # set random seed if required (0 = no random seed)

    "run_num_pretrained": 0      #### change this to prevent overwriting weights in same env_name folder
}

import copy
def expand(list_of_dicts):
    out_dicts = []
    for dict in list_of_dicts:
        any_lists = False
        for key, value in dict.items():
            if isinstance(value, list):
                any_lists = True
                subdicts = []
                for item in value:
                    subdicts.append(copy.deepcopy(dict))
                    subdicts[-1][key] = item
                out_dicts += expand(subdicts)
                break
        if not any_lists:
            out_dicts += [dict]
    return out_dicts

config_dicts = expand([config_option_dict])