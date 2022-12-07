# This file is based off of the PPO Implementation tutorial by Costa Huang
# Costa Huang: https://costa.sh/
# Video Source: https://www.youtube.com/watch?v=MEt6rrxH8W4

import argparse
import os
from distutils.util import strtobool
import time
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

#function to create a gym envirnoment
# inputs: gym_id (gym envirnoment id), seed (environmnent seed), idx (current index),
#         capture_video (bool to determine if we want to capture the video), run_name (string
#         value to name the video)
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 100 == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

#PPO specifies the weights are orthogonal and biases are constant in their intializations
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define an agent
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        # Critic network
        # calling nn.Sequential allows us to easily build a forward neural network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # logits are un-neuralized action probabilities
        logits = self.actor(x)
        # pass it through a Categorical (essentially softmax) to get action probability dirstribution
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
def parse_args():
    parser = argparse.ArgumentParser()
    # each variable in the receives: a name, type, default value, and help text
    parser.add_argument(
        '--exp-name', # name
        type=str, # type
        default=os.path.basename(__file__).rstrip(".py"), # default value
        help="The name of the experiment" # help text
        )
    parser.add_argument('--gym-id', type=str, default='CartPole-v1', 
        help="The id of the gym environment")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, 
        help="The learning rate of the optimizer")
    parser.add_argument('--seed', type=int, default=1, 
        help="seed of the experiment")
    parser.add_argument('--total-timesteps', type=int, default=25000, 
        help="Total timesteps of the experiment")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, 
        help="if toggled, the experiment will be tracked with Weights and Biases")
    parser.add_argument('--wandb-project-name', type=str, default='cleanRL', 
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None, 
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, 
        help="whether to capture videos of the agents performances (will be in the 'videos' folder)")
    
    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4, 
        help="number of parallel game envirnoments")
    # Controls how much data we will gather
    # in this case 4 envs * 128 steps => 512 data points for training per rollout
    # these data points are also referred to as the batch size
    parser.add_argument('--num-steps', type=int, default=128, 
        help="number of steps to run in each envirnoment per policy rollout")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="Use GAE for advantage computation")
    parser.add_argument('--gamma', type=float, default=0.99, 
        help="The discount factor gamma")
    parser.add_argument('--gae-lambda', type=float, default=0.95, 
        help="the lambda for general advantage estimation")
    parser.add_argument('--num-minibatches', type=int, default=4, 
        help="number of mini-batches")
    parser.add_argument('--update-epochs', type=int, default=4, 
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="toggles advantage normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2, 
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, 
        help="toggles whether or not to use a clipped loss for the value function, as per the paper")
    parser.add_argument('--ent-coef', type=float, default=0.01, 
        help="coefficient of entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5, 
        help="coefficient of entropy")
    parser.add_argument('--max-grad-norm', type=float, default=0.5, 
        help="the maximum norm for the gradient clipping")
    parser.add_argument('--target-kl', type=float, default=None, 
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


if __name__ == "__main__":
    args=parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed) # random seed for our experiment
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # one line of code to create a vectorized environment with a specific id and seed
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action spaces supported"
    #print("envs.single_observation_space.shape: ", envs.single_observation_space.shape)
    #print("envs.single_action_space.n: ", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Algorithm logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.Tensor(args.num_envs).to(device)
    # Calculate total number of updates for entirety of training
    num_updates = args.total_timesteps // args.batch_size
    #print(num_updates)
    #print("next_obs.shape: ", next_obs.shape)
    #print("agent.get_value(next_obs): ", agent.get_value(next_obs))
    #print("agent.get_value(next_obs).shape: ", agent.get_value(next_obs).shape)
    #print()
    #print("agent.get_action_and_value(next_obs): ", agent.get_action_and_value(next_obs))

    # update the current learning rate as the training runs
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # policy rollout
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # no need to chache gradients during rollouts
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Use tensorboard to log the episodic return and length
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # reward bootstrapping and GAE control
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # varaibles that store the flattened shape of our other vairables
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize the policy and the value network
        b_inds = np.arange(args.batch_size)
        # Clipped fractions variable
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # forward pass on minibatch observations
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # calculate and store kl
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio-1) - logratio).mean()
                    clipfracs += [((ratio-1.0).abs() > args.clip_coef).float().mean()]

                # calculate the advantages of the minibatch
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                #policy loss and clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value loss clipping
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                #entropy loss (level of chaos in action probability distribution)
                #maximizing entropy should make the agent explore more
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # global gradient clipping
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping implementation (stop updating if our KL divergence is too large)
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Explained variance tells us if the value function is a good indicator of the returns
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1-np.var(y_true-y_pred) / var_y

        # Use TensorBoard to record all of the metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac",np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

