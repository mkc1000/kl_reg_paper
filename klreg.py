import os
import torch
device = "cuda:0"
env_device = "cuda:1"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


init_prompt = """The following contains private conversations, collected for training purposes. Do not distribute.

Student: For our science reports...

Teacher: Yeah?

Student: I don't understand what is supposed to go in the discussion section.

Teacher: Do you"""

mixtral_cache_dir = './mixtral_cache/'
def get_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=mixtral_cache_dir)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map=env_device, quantization_config=bnb_config, cache_dir=mixtral_cache_dir)
    return model, tokenizer

def detach_list_of_lists(list_of_lists):
    return [[tensor.detach() for tensor in lst] for lst in list_of_lists]


# HUGGINGFACE-SPECIFIC FUNCTIONS

def last_hid_layer_activations_and_logits(llm, input_tokens, temp, cache=None, context_window_limit=256):
    input_tokens = input_tokens.view(1, -1)
    if input_tokens.shape[1] > context_window_limit:
        input_tokens = input_tokens[:, -context_window_limit:]
    if cache is not None:
        out = llm(input_ids=input_tokens[:, -1:], output_hidden_states=True, use_cache=True, past_key_values=cache)
    else:
        out = llm(input_ids=input_tokens, output_hidden_states=True, use_cache=True)
    logits = out.logits[0, -1].detach()
    hid_states = torch.cat([out.hidden_states[-i][0, -1] for i in range(1, 4)]).detach()
    cache_for_next_time = detach_list_of_lists(out.past_key_values)
    del out
    return hid_states, logits / temp, cache_for_next_time

def sample_next_token(llm, input_tokens, cache=None, context_window_limit=256):
    input_tokens = input_tokens.view(1, -1)
    if input_tokens.shape[1] > context_window_limit:
        input_tokens = input_tokens[:, -context_window_limit:]
    if cache is not None:
        out = llm(input_ids=input_tokens[:, -1:], use_cache=True, past_key_values=cache)
    else:
        out = llm(input_ids=input_tokens, use_cache=True)
    probs = torch.nn.functional.softmax(out.logits[0, -1], dim=-1).detach()
    dist = torch.distributions.Categorical(probs)
    cache_for_next_time = detach_list_of_lists(out.past_key_values)
    del out
    return dist.sample(), cache_for_next_time

def tokenize(tokenizer, string, device):
    """Returns a 1d tensor of integer tokens on device"""
    return tokenizer([string], return_tensors="pt").to(device)['input_ids'].view(-1)

def detokenize(tokenizer, tokens):
    """Takes a 1d tensor of integer tokens; returns a string"""
    return tokenizer.batch_decode(tokens.view(1, -1))[0]

# END HUGGINGFACE-SPECIFIC FUNCTIONS

import nest_asyncio
nest_asyncio.apply()
import asyncio

from collections import defaultdict
class BatchRunLLM():
    def __init__(self, llm, batch_size):
        self.llm = llm
        self.device = llm.device
        self.batch_size = batch_size
        self.events = {}
        self.input_tokens = defaultdict(lambda: [])
        self.caches = defaultdict(lambda: [])
        self.output_hid_states = {}
        self.output_logits = {}
        self.new_caches = {}
        self.read_checks = defaultdict(lambda: [False for i in range(self.batch_size)])

    async def run_llm(self, input_tokens, cached_activations, temp, sample=False):
        seq_l = len(input_tokens)
        self.input_tokens[seq_l].append(input_tokens.view(1, -1)[:, -1:])
        self.caches[seq_l].append(cached_activations)
        i = len(self.input_tokens[seq_l])
        if i == 1:
            self.events[seq_l] = asyncio.Event()
        if i < self.batch_size:
            await self.events[seq_l].wait()
        else:
            batch_input_tokens = torch.vstack(self.input_tokens[seq_l])
            cache_dim0 = len(self.caches[seq_l][0])
            cache_dim1 = len(self.caches[seq_l][0][0])
            batch_cache = [[torch.cat([cache[i][j] for cache in self.caches[seq_l]], dim=0) for j in range(cache_dim1)] for i in range(cache_dim0)]
            out = self.llm(input_ids=batch_input_tokens, output_hidden_states=True, use_cache=True, past_key_values=batch_cache)
            self.output_logits[seq_l] = out.logits[:, -1, :].detach()
            self.output_hid_states[seq_l] = torch.hstack([out.hidden_states[-i][:, -1, :] for i in range(1, 4)]).detach()
            self.new_caches[seq_l] = detach_list_of_lists(out.past_key_values)
            self.events[seq_l].set()
            del self.input_tokens[seq_l]
            del self.caches[seq_l]
        hid_states, logits = self.output_hid_states[seq_l][i-1], self.output_logits[seq_l][i-1] / temp
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        cache_for_next_time = [[tensor[i-1:i] for tensor in row] for row in self.new_caches[seq_l]]
        self.read_checks[seq_l][i-1] = True
        if all(self.read_checks[seq_l]):
            del self.output_hid_states[seq_l]
            del self.output_logits[seq_l]
            del self.new_caches[seq_l]
            del self.read_checks[seq_l]
        if sample:
            probs = torch.nn.functional.softmax(logits, dim=-1).detach()
            dist = torch.distributions.Categorical(probs)
            return dist.sample(), cache_for_next_time
        else:
            return hid_states, logits, cache_for_next_time

from transformers import pipeline, AutoModelForSequenceClassification
sentiment_cache_dir = './sentiment_cache/'

def get_sentiment_pipeline():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=sentiment_cache_dir)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map=env_device, cache_dir=sentiment_cache_dir)
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    return sentiment_pipeline

from functools import reduce
def find_substring(string_tensor, substring_tensor, first_char_after=False):
    """Returns indexes of first character of substring, where substring_tensor appears in string_tensor"""
    T = string_tensor.view(-1) # reshaping should be unnecessary
    t = substring_tensor.view(-1) # reshaping should be unnecessary
    hits = []
    for character in t:
        hits.append(torch.isin(T, character))
    l = len(t)
    hits_slided = [hit[i:hit.shape[0]-l+i+1] for i, hit in enumerate(hits)]
    matches = reduce(torch.logical_and, hits_slided)
    nums = torch.arange(matches.shape[0], device=matches.device)
    return nums[matches] + (l if first_char_after else 0)

class DiscreteActionSpace():
    def __init__(self, n):
        self.n = n

class ContinuousObsSpace():
    def __init__(self, shape):
        self.shape = shape

class ChatEnv():
    def __init__(self, chat_start, my_name, other_names, llm, tokenizer, sentiment_pipeline,
                 batchrunllm, base_pol_temp=1.0, env_temp=1.0, device='auto', max_chat_length=256, verbose=False):
        # no name (my_name or elements of other_names) may be a suffix of another
        self.start_str = chat_start
        self.me_str = my_name
        self.others_str = other_names
        self.llm = llm
        self.base_pol_temp = base_pol_temp
        self.env_temp = env_temp
        self.tok = tokenizer
        self.sentiment_pipeline = sentiment_pipeline
        self.device = device
        if self.device == 'auto':
            if hasattr(self.llm, "device"):
                self.device = self.llm.device
            else:
                raise Exception('If llm does not have an attribute "device", device cannot be "auto"')
        self.max_chat_length = max_chat_length

        self.start_chat = tokenize(self.tok, self.start_str, self.device)
        self.chat = self.start_chat.clone()
        self.speaking_record = torch.zeros(self.chat.shape[0], dtype=torch.bool, device=self.device)
        self.me, self.others = self._get_name_tokens()
        self.me_len = self.me.shape[0]
        self.cutoff_tokens = tokenize(self.tok, "--\n\n", self.device)
        self.cutoff_tokens = torch.cat([self.cutoff_tokens, self.me], dim=0)
        self.others_lens = [other.shape[0] for other in self.others]
        self.start_obs, self.start_base_pol_logits, self.start_llm_cache = last_hid_layer_activations_and_logits(self.llm, self.chat, self.base_pol_temp, context_window_limit=max_chat_length)
        self.start_obs = torch.cat([self.start_obs, torch.tensor([self.chat.shape[0]/self.max_chat_length], device=self.start_obs.device, dtype=self.start_obs.dtype)])
        self.llm_cache = [[tensor.clone() for tensor in row] for row in self.start_llm_cache]
        self.action_space = DiscreteActionSpace(self.start_base_pol_logits.shape[0])
        self.observation_space = ContinuousObsSpace(self.start_obs.shape)
        self.batchrunllm = batchrunllm
        self.verbose=verbose

    def _get_name_tokens(self):
        # get chat up to just before name, and tokenize it
        # get chat up to just after name, and tokenize it
        # take difference
        name_tokens = []
        name_strs = [self.me_str] + self.others_str
        name_strs = [name + ":" for name in name_strs]
        for name in name_strs:
            name_index = self.start_str.find(name)
            if name_index == -1:
                raise Exception("The name '" + name + "' must appear in chat_start")
            c1 = self.start_str[:name_index]
            c2 = c1 + name
            tokens1 = tokenize(self.tok, c1, self.device)
            tokens2 = tokenize(self.tok, c2, self.device)
            len_c1 = tokens1.shape[0]
            this_name_tokens = tokens2[len_c1:]
            if detokenize(self.tok, this_name_tokens[-1]) == "":
                this_name_tokens = this_name_tokens[:-1]
            name_tokens.append(this_name_tokens)
        # check
        for name_tok, name in zip(name_tokens, name_strs):
            assert(detokenize(self.tok, name_tok) == name)
        return name_tokens[0], name_tokens[1:]

    def _is_speaking(self):
        # identify if self.me is speaking
        last_debut = torch.max(find_substring(self.chat, self.me))
        last_other_debut = reduce(torch.maximum, [torch.max(find_substring(self.chat, other)) for other in self.others])
        return last_debut > last_other_debut

    def reset(self, out_device=None, verbose=False):
        if verbose:
            print("\n\n\n\n\n\nReset Environment")
        self.chat = self.start_chat.clone()
        self.speaking_record = torch.zeros(self.chat.shape[0], dtype=torch.bool, device=self.device)
        self.llm_cache = [[tensor.clone() for tensor in row] for row in self.start_llm_cache]
        base_pol_logprobs = torch.nn.functional.log_softmax(self.start_base_pol_logits, dim=-1)
        if out_device is None:
            return self.start_obs, base_pol_logprobs
        else:
            return self.start_obs.to(device=out_device), base_pol_logprobs.to(device=out_device)

    def _recent_sentiment_of_other(self):
        last_debut = torch.max(find_substring(self.chat, self.me))
        last_other_debut = reduce(torch.maximum, [torch.max(find_substring(self.chat, other, first_char_after=True)) for other in self.others])
        other_tokens = self.chat[last_other_debut:last_debut]
        text = detokenize(self.tok, other_tokens)
        sentiment_dict = self.sentiment_pipeline([text])[0]
        r = sentiment_dict['score'] if sentiment_dict['label'] == 'POSITIVE' else -sentiment_dict['score']
        return (r + 1) / 2

    async def step(self, action, reply_cutoff=512):
        input_device = action.device
        if input_device != self.device:
            action = action.to(device=self.device)
        if self.verbose:
            print("-", detokenize(self.tok, action))
        self.chat = torch.cat([self.chat, action.view(1)], dim=0)
        self.speaking_record = torch.cat([self.speaking_record, torch.ones(1, dtype=torch.bool, device=self.device)], dim=0)

        i = 0
        while not self._is_speaking():
            if len(self.chat) == self.max_chat_length:
                break
            token, self.llm_cache = await self.batchrunllm.run_llm(self.chat, self.llm_cache, self.env_temp, sample=True)
            if self.verbose:
                print("+", detokenize(self.tok, token))
            if i >= reply_cutoff:
                token = self.cutoff_tokens[i - reply_cutoff]
            self.chat = torch.cat([self.chat, token.view(1)], dim=0)
            self.speaking_record = torch.cat([self.speaking_record, torch.zeros(1, dtype=torch.bool, device=self.device)], dim=0)
            assert(i < 1e5)
            i += 1
        reward = self._recent_sentiment_of_other() if i > 0 else 0
        obs, base_pol_logits, self.llm_cache = await self.batchrunllm.run_llm(self.chat, self.llm_cache, self.base_pol_temp, sample=False)
        obs = torch.cat([obs, torch.tensor([1 - self.chat.shape[0]/self.max_chat_length], device=obs.device, dtype=obs.dtype)])
        base_pol_logprobs = torch.nn.functional.log_softmax(base_pol_logits, dim=-1)
        done = False
        if len(self.chat) == self.max_chat_length:
            done = True
        truncated = False
        if input_device != self.device:
            obs = obs.to(device=input_device)
            base_pol_logprobs = base_pol_logprobs.to(device=input_device)
        info = {'base_pol_logprobs' : base_pol_logprobs}
        return obs, reward, done, truncated, info

    def seed(self, random_seed):
        torch.manual_seed(random_seed)

    def render(self):
        print(detokenize(self.tok, self.chat))

    def close(self):
        pass

# Fixed KL budget
import math
    
def logx_and_kl(loga, logb, alpha):
    logx = torch.logaddexp(loga + torch.log(alpha), logb + torch.log(1-alpha))
    kl_i = torch.exp(logx) * (logx - logb)
    return logx, torch.sum(kl_i, dim=-1, keepdims=True)

def dkl_dalpha(loga, logb, logx, alpha):
    a_minus_b = torch.exp(loga) - torch.exp(logb)
    dkl_dalpha = torch.sum((a_minus_b * (logx - logb + 1)), dim=-1, keepdims=True)
    return dkl_dalpha

def findalphafork(loga, logb, k, n_steps=20, verbose=False):
    """Identify a value of lambd such that, with x= alpha*a + (1-alpha)*b,
    KL(x || b) = k, or get as close as possible if k is too big
    [Batched]
    a : (n, d)
    logb : (n, d)
    k : (n, 1)
    """
    alpha_top = torch.ones_like(k)
    alpha_bot = torch.zeros_like(k)
    # x = alpha * torch.exp(a) + (1-alpha) * torch.exp(logb)
    alpha = torch.ones_like(k)
    for i in range(n_steps):
        logx, kl = logx_and_kl(loga, logb, alpha)
        alpha_bot[(k >= kl).view(-1)] = alpha[(k >= kl).view(-1)]
        alpha_top[(k <= kl).view(-1)] = alpha[(k <= kl).view(-1)]
        dkl_dalp = dkl_dalpha(loga, logb, logx, alpha)
        aim = k - kl
        delta = aim / (dkl_dalp + 1e-12)
        alpha = torch.clamp(alpha+delta, min=alpha_bot*15/16+alpha_top*1/16, max=alpha_bot*1/2+alpha_top*1/2)
        if verbose:
            print("kl", kl)
            print("alpha", alpha)
    return alpha

def a_minus_b_over_x(loga, logb, alpha):
    mean = (loga + logb) / 2
    loga_, logb_ = loga - mean, logb - mean
    a, b = torch.exp(loga_), torch.exp(logb_)
    x = alpha * a + (1-alpha) * b
    # out = (a - b) / x
    # out[torch.isnan(out)] = 1.0
    return (a - b) / x

def amiss(tensor):
    return torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor))

class KLFixedMixture(torch.autograd.Function):
    """
    Return x = alpha * a + (1-alpha) * b minimizing [KL(x || b) - k]**2
    Autograd is implemented for a and k, but not b
    Squared error between k and k_achieved should be added to any loss function
    so that k can be moved to a region where changing it affects the value of x
    """
    @staticmethod
    def forward(ctx, k, loga, logb):
        dim = loga.dim()
        if dim == 1:
            loga = loga.unsqueeze(0)
            logb = logb.unsqueeze(0)
            k = k.unsqueeze(0)
        alpha = findalphafork(loga, logb, k)
        logx, kl_achieved = logx_and_kl(loga, logb, alpha)
        ctx.save_for_backward(loga, logb, logx, k, alpha)
        if dim == 1:
            logx = logx.squeeze(0)
            kl_achieved = kl_achieved.squeeze(0)
        return logx, kl_achieved

    @staticmethod
    def backward(ctx, grad_wrt_logx, grad_wrt_kl_achieved, debug=False):
        dim = grad_wrt_logx.dim()
        if dim == 1:
            grad_wrt_logx = grad_wrt_logx.unsqueeze(0)
        loga, logb, logx, k, alpha = ctx.saved_tensors
        a, b, x = torch.exp(loga), torch.exp(logb), torch.exp(logx)
        dkl_dalp = dkl_dalpha(loga, logb, logx, alpha)
        dalp_dk = 1 / (dkl_dalp + 1e-12)
        # dx_dk = (a - b) * dalp_dk
        # dlogx_dx = 1/x
        a_minus_b_over_x_ = a_minus_b_over_x(loga, logb, alpha)
        dlogx_dk = dalp_dk * a_minus_b_over_x_
        grad_wrt_k = torch.sum(grad_wrt_logx * dlogx_dk, dim=-1, keepdim=True)
        dkl_da = alpha * (1 + logx - logb)
        dalp_da = - dalp_dk * dkl_da
        grad_wrt_a_part = torch.sum(grad_wrt_logx * a_minus_b_over_x_, dim=-1, keepdim=True) * dalp_da
        # grad_wrt_a = grad_wrt_a_part + (alpha / x * grad_wrt_logx)
        # grad_wrt_loga = grad_wrt_a * a
        grad_wrt_loga = a * grad_wrt_a_part + (alpha * grad_wrt_logx) * torch.exp(loga - logx)
        # different form if alpha >= 1
        grad_wrt_k[alpha.squeeze(-1) == 1] = 0
        grad_wrt_loga[alpha.squeeze(-1) == 1] = grad_wrt_logx[alpha.squeeze(-1) == 1]
        if dim == 1:
            grad_wrt_k = grad_wrt_k.squeeze(0)
            grad_wrt_loga = grad_wrt_loga.squeeze(0)
        if debug:
            if amiss(grad_wrt_loga) or amiss(grad_wrt_k):
                for var_name, var_value in locals().items():
                    print(f"{var_name}: {var_value}")
                total_mask = torch.logical_or(torch.isnan(grad_wrt_loga), torch.isinf(grad_wrt_loga))
                row_mask = torch.any(total_mask, dim=-1)
                col_mask = torch.any(total_mask, dim=-2)
                for var_name, var_value in locals().items():
                    if torch.is_tensor(var_value):
                        try:
                            print(f"{var_name} selected: {var_value[row_mask][:, col_mask]}")
                        except:
                            pass
                        try:
                            print(f"{var_name} selected: {var_value[0][:, row_mask]}")
                        except:
                            pass
                        try:
                            print(f"{var_name} selected whole row: {var_value[row_mask]}")
                        except:
                            pass
                    else:
                        print(var_name, str(type(var_value)))
                row_mask = torch.logical_or(torch.isnan(grad_wrt_k), torch.isinf(grad_wrt_k))
                for var_name, var_value in locals().items():
                    if torch.is_tensor(var_value):
                        try:
                            print(f"{var_name} selected (k error): {var_value[row_mask]}")
                        except:
                            pass
                    else:
                        print(var_name, str(type(var_value)))
        return grad_wrt_k, grad_wrt_loga, None

class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim, n_versions_last_layer=1):
        super().__init__()
        self.depth = depth
        self.n_versions_last_layer = n_versions_last_layer
        input_sizes = [hidden_dim for _ in range(depth)]
        output_sizes = [hidden_dim for _ in range(depth)]
        input_sizes[0] = input_dim + 1
        output_sizes[-1] = (output_dim + 1) * self.n_versions_last_layer
        self.linear_layers = [torch.nn.Linear(in_size, out_size) for in_size, out_size in zip(input_sizes, output_sizes)]
        for i, lin_lay in enumerate(self.linear_layers):
            gain = 1.0
            if i == 0:
                gain = 0.05
            if i == depth - 1:
                gain = 0.0
            torch.nn.init.orthogonal_(lin_lay.weight.data, gain=gain)
            self.add_module("layer"+str(i), lin_lay)

    def forward(self, x, kl_budget, log_base_dist, actions_left=16): # Have kl_budget be input to linear layers too?
        if x.dim() < 2:
            x = x.unsqueeze(0)
        which_last_layer = (torch.clamp(x[:, -1], min=0, max=1-1e-4) * self.n_versions_last_layer).to(dtype=torch.int64)
        if kl_budget.dim() < 2:
            kl_budget = kl_budget.unsqueeze(0)
        x = torch.hstack((x, kl_budget))
        if log_base_dist.dim() < 2:
            log_base_dist = log_base_dist.unsqueeze(0)
        for i, linear_layer in enumerate(self.linear_layers):
            if i != 0:
                x = torch.nn.functional.tanh(x)
            x = linear_layer(x)
        if self.n_versions_last_layer > 1:
            selected_x = x.view(x.shape[0], self.n_versions_last_layer, x.shape[1] // self.n_versions_last_layer)[:, which_last_layer, :].view(x.shape[0], -1)
        else:
            selected_x = x
        kl_target = torch.sigmoid(selected_x[:, :1] - math.log(actions_left)) * kl_budget + 1e-6
        loga = selected_x[:, 1:] + log_base_dist
        loga = loga - torch.max(loga, dim=-1, keepdim=True).values
        loga = loga - torch.logsumexp(loga, dim=-1, keepdim=True)
        x_out, kl_achieved = KLFixedMixture.apply(kl_target, loga, log_base_dist)
        return torch.softmax(x_out, dim=-1), torch.square(kl_achieved - kl_target) # softmax should be same as exp

############################### Import libraries ###############################


import torch.nn as nn
from torch.distributions import Categorical


################################## PPO Policy ##################################

import pickle
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.base_pol_logprobs = []
        self.kl_costs = []
        self.kl_budget_left = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.base_pol_logprobs[:]
        del self.kl_costs[:]
        del self.kl_budget_left[:]

    def __add__(self, other_buffer):
        new = RolloutBuffer()
        new.actions = self.actions + other_buffer.actions
        new.states = self.states + other_buffer.states
        new.logprobs = self.logprobs + other_buffer.logprobs
        new.rewards = self.rewards + other_buffer.rewards
        new.state_values = self.state_values + other_buffer.state_values
        new.is_terminals = self.is_terminals + other_buffer.is_terminals
        new.base_pol_logprobs = self.base_pol_logprobs + other_buffer.base_pol_logprobs
        new.kl_costs = self.kl_costs + other_buffer.kl_costs
        new.kl_budget_left = self.kl_budget_left + other_buffer.kl_budget_left
        return new

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, dtype=torch.float32):
        super(ActorCritic, self).__init__()

        # self.width = 4096
        self.width = 12289

        orig_type = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        self.actor = Actor(state_dim, self.width, 1, action_dim)
        # self.actor = Actor(state_dim, self.width, 2, action_dim)


        # critic
        attn_dim = 128
        self.critic = nn.Sequential(
                        nn.Linear(state_dim+1, attn_dim),
                        nn.Tanh(),
                        nn.Linear(attn_dim, attn_dim),
                        nn.Tanh(),
                        nn.Linear(attn_dim, 1)
                    )

        self.critic[-1].weight.data.fill_(0.0)
        self.critic[-1].bias.data.fill_(0.0)
        self.critic_with_budg = lambda state, kl_budget_left: self.critic(torch.cat([state, kl_budget_left], dim=-1))
        torch.set_default_dtype(orig_type)
        self.log_last_kl = None

    def forward(self):
        raise NotImplementedError


    def act(self, state, base_pol_logprobs, kl_budget_left, actions_left=16):

        action_probs, extra_loss = self.actor(state, kl_budget_left, base_pol_logprobs, actions_left)
        action_logprobs = torch.log(torch.clamp(action_probs, min=1e-12))
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic_with_budg(state, kl_budget_left)

        self.log_last_kl = torch.sum(action_probs * (action_logprobs - base_pol_logprobs)).detach()

        return action.detach(), action_logprob.detach(), state_val.detach(), self.log_last_kl


    def evaluate(self, state, action, base_pol_logprobs, kl_budget_left, actions_left=16):
        # With base_pol, no longer supports continuous action space

        action_probs, extra_loss = self.actor(state, kl_budget_left, base_pol_logprobs, actions_left)
        action_logprobs = torch.log(torch.clamp(action_probs, min=1e-12))
        dist = Categorical(action_probs)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        kls = torch.sum(action_probs * (action_logprobs - base_pol_logprobs), dim=-1)
        state_values = self.critic_with_budg(state, kl_budget_left - kls.detach().view(-1, 1))

        return action_logprob, state_values, dist_entropy, kls, extra_loss

import random
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, update_batch_size, eps_clip, ent_coef, max_grad_norm, gae_lam=None, dtype=torch.float32, dev=None):

        self.gamma = gamma
        self.gae_lam = gae_lam
        if self.gae_lam is None:
            print("gae_lam is None, so doing average reward, not gamma-discounted")
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = update_batch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.last_used_episode_kl_budget = None

        self.buffers = defaultdict(lambda: RolloutBuffer())
        self.buffer = RolloutBuffer()

        if dev is None:
            dev = device
        self.policy = ActorCritic(state_dim, action_dim, dtype=dtype)
        self.policy = self.policy.to(dev)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'eps': 1e-5},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic, 'eps': 1e-5}
                    ])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0) # i.e. no lr decay

        self.policy_old = ActorCritic(state_dim, action_dim, dtype=dtype).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.dtype = dtype

        self.log_latest_mse_loss = 0
        self.log_latest_kl_loss = 0
        self.log_surr_loss = 0

        self.kl_budget_left = {}

    def set_kl_budget_left(self, buffer_key, kl_budget_left):
        if not torch.is_tensor(kl_budget_left):
            kl_budget_left = torch.tensor([kl_budget_left], dtype=self.dtype, device=device)
        self.kl_budget_left[buffer_key] = kl_budget_left
        self.last_used_episode_kl_budget = kl_budget_left

    def del_kl_budget_left(self, buffer_key):
        del self.kl_budget_left[buffer_key]

    def select_action(self, state, base_pol_logprobs, buffer_key, actions_left=16, pytorch_env=True, follow_base_pol=False):
        kl_budget_left = self.kl_budget_left[buffer_key]
        state = state.to(dtype=self.dtype)
        base_pol_logprobs = base_pol_logprobs.to(dtype=self.dtype)

        if follow_base_pol:
            action_logprob = base_pol_logprobs
            dist = Categorical(action_logprob)
            action = dist.sample()
            kl_cost = 0
            state_val = self.policy_old.critic_with_budg(state, kl_budget_left)
        else:
            with torch.no_grad():
                if not pytorch_env:
                    state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, kl_cost = self.policy_old.act(state, base_pol_logprobs, kl_budget_left, actions_left)

        self.buffers[buffer_key].states.append(state.to('cpu'))
        self.buffers[buffer_key].actions.append(action.to('cpu'))
        self.buffers[buffer_key].logprobs.append(action_logprob.to('cpu'))
        self.buffers[buffer_key].state_values.append(state_val.to('cpu'))
        self.buffers[buffer_key].base_pol_logprobs.append(base_pol_logprobs.to('cpu'))
        self.buffers[buffer_key].kl_costs.append(kl_cost.to('cpu').item())
        self.buffers[buffer_key].kl_budget_left.append(kl_budget_left.to('cpu'))

        self.kl_budget_left[buffer_key] = self.kl_budget_left[buffer_key] - (action_logprob - base_pol_logprobs[action])
        self.kl_budget_left[buffer_key] = torch.clamp(self.kl_budget_left[buffer_key], min=0)

        info = {"action_logprob" : self.buffers[buffer_key].logprobs[-1].item(),
                "base_pol_logprob" : self.buffers[buffer_key].base_pol_logprobs[-1][self.buffers[buffer_key].actions[-1]].item(),
                "kl" : self.buffers[buffer_key].kl_costs[-1]}

        if pytorch_env:
            return action, info
        return action.item(), info

    def update(self, critic_only=False, override_Kepochs=None, override_kl_budget=None):
        for _, buffer in self.buffers.items():
            self.buffer = self.buffer + buffer # buffers should all end with done = True, but it would be good to ensure this
            buffer.clear()

        returns = []
        sum_reward = 0
        i = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                sum_reward = 0
                i = 0
            i += 1
            sum_reward = reward + sum_reward
            returns.insert(0, sum_reward)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device, dtype=self.dtype)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device, dtype=torch.int64)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device, dtype=self.dtype)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device, dtype=self.dtype)
        old_base_pol_logprobs = torch.squeeze(torch.stack(self.buffer.base_pol_logprobs, dim=0)).detach().to(device, dtype=self.dtype)
        kl_budget_left = torch.stack(self.buffer.kl_budget_left, dim=0).detach().to(device, dtype=self.dtype)
        if override_kl_budget is not None:
            kl_budget_left.fill_(override_kl_budget)

        rewards = torch.tensor(returns, dtype=self.dtype).to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        if self.gae_lam is not None:
            gaes = torch.empty_like(old_state_values)
            self.buffer.is_terminals[-1] = True
            latest_gae = None
            for i in reversed(range(len(self.buffer.rewards))):
                reward = self.buffer.rewards[i]
                is_terminal = self.buffer.is_terminals[i]
                if is_terminal:
                    next_est_state_value = 0
                else:
                    next_est_state_value = old_state_values[i+1]
                delta = reward + self.gamma * next_est_state_value - old_state_values[i]
                latest_gae = delta
                if is_terminal:
                    latest_gae = delta
                else:
                    latest_gae = delta + latest_gae * self.gamma * self.gae_lam
                gaes[i] = latest_gae
            advantages = gaes

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Batches
        N = rewards.shape[0]
        if self.batch_size is None:
            self.batch_size = N
        if self.batch_size > N:
            self.batch_size = N
        n_batches = N // self.batch_size

        # Optimize policy for K epochs
        epochs = self.K_epochs
        if override_Kepochs is not None:
            epochs = override_Kepochs
        for _ in range(epochs):
            # # Over n_batches batches
            batch_indices = torch.randperm(N, device=device)[:n_batches*self.batch_size].reshape(n_batches, self.batch_size)
            for batch in batch_indices:
                rewards_b = rewards[batch]
                old_states_b = old_states[batch]
                old_actions_b = old_actions[batch]
                old_logprobs_b = old_logprobs[batch]
                old_base_pol_logprobs_b = old_base_pol_logprobs[batch]
                advantages_b = advantages[batch]
                kl_budget_left_b = kl_budget_left[batch]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy, kls, extra_loss = self.policy.evaluate(old_states_b, old_actions_b, old_base_pol_logprobs_b, kl_budget_left_b)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs_b.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages_b
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_b

                # final loss of clipped objective PPO
                if critic_only:
                    loss = 0.5 * self.MseLoss(state_values, rewards_b)
                else:
                    loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_b) - self.ent_coef * dist_entropy + extra_loss.view(-1)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
            if critic_only:
                print("critic loss:", loss)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save_old(self, checkpoint_path):
        checkpoint = {
            'policy_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def load_old(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        checkpoint_copy = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint_copy['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def save(self, checkpoint_path):
        checkpoint = {
            'policy_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'gamma': self.gamma,
            'gae_lab': self.gae_lam, # too late to fix this now
            'eps_clip': self.eps_clip,
            'K_epochs': self.K_epochs,
            'batch_size': self.batch_size,
            'ent_coef': self.ent_coef,
            'max_grad_norm': self.max_grad_norm,
            'last_used_episode_kl_budget': self.last_used_episode_kl_budget
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        checkpoint_copy = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.load_state_dict(checkpoint_copy['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.gamma = checkpoint['gamma']
        self.gae_lam = checkpoint['gae_lab'] # too late to fix this now
        self.eps_clip = checkpoint['eps_clip']
        self.K_epochs = checkpoint['K_epochs']
        self.batch_size = checkpoint['batch_size']
        self.ent_coef = checkpoint['ent_coef']
        self.max_grad_norm = checkpoint['max_grad_norm']
        self.last_used_episode_kl_budget = torch.tensor([checkpoint['last_used_episode_kl_budget']], dtype=self.dtype, device=device)

    def save_and_clear_buffers(self, buffer_folder):
        for _, buffer in self.buffers.items():
            self.buffer = self.buffer + buffer # buffers should all end with done = True, but it would be good to ensure this
            buffer.clear()
        buffer_path = os.path.join(buffer_folder, f"{random.randint(0, int(1e8))}.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(self.buffer, f)
        self.buffer.clear()

    def load_buffers(self, buffer_folder):
        buffer_paths = [os.path.join(buffer_folder, fname) for fname in os.listdir(buffer_folder) if fname.endswith(".pkl")]
        for buffer_path in buffer_paths:
            with open(buffer_path, "rb") as f:
                self.buffer = self.buffer + pickle.load(f)
            os.remove(buffer_path)

    def load_random_buffer(self, buffer_folder):
        self.buffer.clear()
        buffer_paths = [os.path.join(buffer_folder, fname) for fname in os.listdir(buffer_folder) if fname.endswith(".pkl")]
        with open(random.choice(buffer_paths), "rb") as f:
            self.buffer = pickle.load(f)

    def remove_buffer_files(self, buffer_folder):
        buffer_paths = [os.path.join(buffer_folder, fname) for fname in os.listdir(buffer_folder) if fname.endswith(".pkl")]
        for buffer_path in buffer_paths:
            os.remove(buffer_path)


####################################################################################

async def run_episode(ppo_agent, envs, env_id, per_episode_kl_budget, max_ep_len=10000):
    env = envs[env_id]
    state, base_pol_logprobs = env.reset(out_device=device)
    # running reward
    current_ep_reward = 0
    kls = []
    log_ratios = []
    rewards = []
    base_pol_ents = []
    ppo_agent.set_kl_budget_left(env_id, per_episode_kl_budget)

    for t in range(1, max_ep_len+1):
        # select action with policy
        action, info = ppo_agent.select_action(state, base_pol_logprobs, env_id)
        kls.append(info["kl"])
        log_ratios.append(info["action_logprob"] - info["base_pol_logprob"])
        state, reward, done, truncated, info = await env.step(action)
        rewards.append(reward)
        # if reward > 0:
        #     print(reward)
        base_pol_logprobs = info['base_pol_logprobs']
        base_pol_ents.append(torch.sum(base_pol_logprobs * torch.exp(base_pol_logprobs)).cpu().item())

        # saving reward and is_terminals
        ppo_agent.buffers[env_id].rewards.append(reward)
        ppo_agent.buffers[env_id].is_terminals.append(done)

        current_ep_reward += reward

        # break; if the episode is over
        if done:
            break
    ppo_agent.del_kl_budget_left(env_id)

    kls = torch.tensor(kls, device=env.speaking_record.device)
    kls_by_token = torch.zeros(env.chat.shape[0], dtype=kls.dtype, device=kls.device)
    kls_by_token[env.speaking_record] = kls
    log_ratios = torch.tensor(log_ratios, device=env.speaking_record.device)
    log_ratios_by_token = torch.zeros(env.chat.shape[0], dtype=kls.dtype, device=kls.device)
    log_ratios_by_token[env.speaking_record] = log_ratios
    rewards = torch.tensor(rewards, device=kls.device, dtype=kls.dtype)
    rewards_by_token = torch.zeros(env.chat.shape[0], dtype=rewards.dtype, device=rewards.device)
    rewards_by_token[env.speaking_record] = rewards
    # print(f"Total episode reward: {current_ep_reward}")
    return t, env.chat.cpu().numpy().tolist(), rewards_by_token.cpu().numpy().tolist(), kls_by_token.cpu().numpy().tolist(), kls.cpu().numpy().tolist(), log_ratios_by_token.cpu().numpy().tolist(), log_ratios.cpu().numpy().tolist(), env.speaking_record.cpu().numpy().tolist(), base_pol_ents

async def run_episodes(ppo_agent, envs, per_episode_kl_budget, verbose=False):
    tasks = [asyncio.create_task(run_episode(ppo_agent, envs, env_id, per_episode_kl_budget)) for env_id in range(len(envs))]
    out = []
    for task in tasks:
        out.append(await task)
    if verbose:
        print("********************Sample chat********************")
        envs[0].render()
        print("***************************************************")
    return out