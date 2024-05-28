import os
from typing import Optional, Sequence, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
import colorednoise as cn
MBPO_LOG_FORMAT = [
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

ROLLOUT_LOG_FORMAT = [("env_step", "S", "int"),
                      ("epoch", "E", "int"),
                      ("average_rollout_length", "AVG-RL", "float"),
                      ("minimum_rollout_length", "MIN-RL", "int"),
                      ("maximum_rollout_length", "MAX-RL", "int"),
                      ("added_transitions", "TR", "int")
                      ]


def get_masking_rate(current_rollout_length: int, max_rollout_length: int, default_masking_rate: float) -> float:
    if max_rollout_length == 1:
        return default_masking_rate
    else:
        return (max_rollout_length - current_rollout_length) / (2 * (max_rollout_length + 1))


def rollout_model_and_populate_sac_buffer(
        model_env: mbrl.models.ModelEnv,
        replay_buffer: mbrl.util.ReplayBuffer,
        agent: SACAgent,
        sac_buffer: mbrl.util.ReplayBuffer,
        sac_samples_action: bool,
        max_rollout_length: int,
        batch_size: int,
        default_masking_rate: float,
        model_error_penalty_coefficient: float,
        logger: mbrl.util.Logger,
        step: int,
        epoch: int
):
    """Generates rollouts to create simulated trainings data for sac agent. These rollouts are used to populate the
    SAC-buffer, from which the agent can learn cheaply how to behave optimal in the approximated environment

    Args:
        model_env (mbrl.models.ModelEnv): The learned model which was transformed to behave like an environment
        replay_buffer (mbrl.util.ReplayBuffer): Replay buffer with transitions experienced in the real environment
        Used in order to sample uniformly start states for model rollouts
        agent (SACAgent): Agent which has learned a policy is used to act to get the taken actions in the rollouts
        sac_samples_action (bool): True if the agents action should be sampled according to gaussian policy
        False if the agent should just choose the mean of the gaussian
        sac_buffer (mbrl.util.ReplayBuffer) : Here the transitions of the rollouts are stored
        max_rollout_length (int): How long can the rollouts be in maximum. The real length is decided
        by the masking algorithm
        So how many actions are taken by the agent in the approximated environment
        batch_size (int): Size of batch of initial states to start rollouts and
        thus there will be batch_size*rollout_horizon more transitions stored in the sac_buffer
        default_masking_rate(float): Proportion of transitions to keep with least uncertainty score
        (for maximum_rollout_length = 1) else the function get_masking_rate is used
    """
    batch = replay_buffer.sample(batch_size)
    # intial_obs ndarray batchsize x observation_size
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    # model_state tensor batchsize x observation_size
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    # accum_dones ndarray batchsize
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs

    # just eval -----
    transition_rollout_step = []
    rollout_tracker = np.zeros((batch_size,))
    for i in range(max_rollout_length):
        # action ndarray batchsize x actionsize
        action = agent.act(obs, sample=sac_samples_action, batched=True)

        # Make step in environment(ModelEnv->1DTransitionrewardModel->GaussianMLP->Ensemble) and get the
        # predicted obs and rewards. Also get dones and model_state.

        # For m2ac the means and vars of the next_obs+reward are needed so get them too.
        # chosen_means,chosen_stds are of size (batchsize) and are the mean and stds of the gaussian
        # that was chosen to sample pred_next_obs, pred_rewards
        # means_of_all_ensembles, stds_of_all_ensembles are (ensemble_size x batchsize) and are all means
        # and stds of all gaussians in the ensemble
        # model_indices is (batchsize) and is the chosen model_indices of the ensemebles [0,ensemble_size)
        (pred_next_obs, pred_rewards, pred_dones, model_state,
         chosen_means, chosen_stds, means_of_all_ensembles,
         stds_of_all_ensembles, model_indices) = model_env.step_plus_gaussians(action, model_state, sample=True)
        rollout_tracker[~accum_dones] += 1
        # just evaluating for loggin purposes the Geometric Jensen-Shannon Divergence-----------

        ensemble_size = model_env.dynamics_model.model.ensemble_size
        ensemble_rest_mean, ensemble_rest_var = mbrl.util.common.calc_rest_ensemble_mean_std_leave_out_model_indices(
            means_of_all_ensembles, stds_of_all_ensembles, model_indices, ensemble_size)
        chosen_vars = torch.pow(chosen_stds, 2)
        uncertainty_score = calc_uncertainty_score(chosen_means, chosen_vars, ensemble_rest_mean, ensemble_rest_var)
        uncertainty_score = uncertainty_score.cpu().numpy()
        sorted_indices_uncertainty = np.argsort(uncertainty_score)
        masking_rate = get_masking_rate(i, max_rollout_length, default_masking_rate)
        # calc how many most certain transitions shall be stored
        number_of_certain_transitions = int(batch_size * masking_rate)
        indices_of_certain_transitions = sorted_indices_uncertainty[:number_of_certain_transitions]
        certain_bool_map = np.zeros(initial_obs.shape[0], dtype=bool)
        # certain_bool_map contains true for storing transition if it is certain enough and false else
        certain_bool_map[indices_of_certain_transitions] = True

        penalize_rewards = pred_rewards[:, 0] - (model_error_penalty_coefficient * uncertainty_score)


        assert np.sum(np.isinf(pred_rewards[:, 0])) == 0
        assert np.sum(np.isnan(pred_rewards[:, 0])) == 0
        # pred_rewards and pred_done need to be of size batchsize not batchsize x 1
        sac_buffer.add_batch(
            obs[~accum_dones & certain_bool_map],
            action[~accum_dones & certain_bool_map],
            pred_next_obs[~accum_dones & certain_bool_map],
            penalize_rewards[~accum_dones & certain_bool_map],
            pred_dones[~accum_dones & certain_bool_map, 0],
        )
        obs = pred_next_obs
        transition_rollout_step.extend(list(rollout_tracker[~accum_dones & certain_bool_map]))
        # squeezing to transform pred_dones from batch_size x 1 to batchsize
        accum_dones |= pred_dones.squeeze()



def calc_uncertainty_score(chosen_means: torch.Tensor, chosen_stds: torch.Tensor,
                           ensemble_rest_mean: torch.Tensor, ensemble_rest_std: torch.Tensor,
                           device="cuda") -> torch.Tensor:
    """This function calculates the Kullback Leiber Divergence between two gaussians but for multiple pairs
    It is assumed that chosen_means, chosen_stds, ensemble_rest_mean, ensemble_rest_std
    are all of shape (batch_size, ).
    It is used the formula in docs/resources/formulas/kl-div.png
    P1 is N(chosen_means[0], chosen_stds[0]) and P2 is N(ensemble_rest_mean[0], ensemble_rest_std[0]) e.g
    Args:
        chosen_means(torch.Tensor):  is [batch_size] Tensor with
        the means of for sampling chosen Gaussian for each observation action pair
        chosen_stds(torch.Tensor):  is [batch_size] Tensor with
        the stds of for sampling chosen Gaussian for each observation action pair
        ensemble_rest_mean (torch.Tensor): [batch_size] sized Tensor with
        the means of merged not chosen Gaussian for each observation action pair
        ensemble_rest_std (torch.Tensor): is [batch_size] sized Tensor with
        the stds of merged not chosen Gaussian for each observation action pair
    Returns:
        torch.Tensor contains the uncertainty scores between all Gaussian pairs
    """

    # all tensors here are of shape[B, n], while B are the simultaneous model rollouts and
    # n is the size of next observation plus reward site of 1
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # !maybe change device of torch ones to cpu or cuda if not working automatically !
    n = torch.ones(chosen_means.size(dim=0)).fill_(chosen_means.size(dim=1)).to(device)
    # n is of size B and contains only values n
    # S1 is refering to Sigma1 so chosen_stds[0] e.g, S2 is refering to Sigma2 so ensemble_rest_std[0] e.g
    log_det_S1 = torch.sum(torch.log(chosen_stds), 1)
    log_det_S2 = torch.sum(torch.log(ensemble_rest_std), 1)
    log_term = log_det_S2 - log_det_S1
    tr_term = torch.sum(torch.div(chosen_stds, ensemble_rest_std), 1)

    mu2_sub_mu1 = ensemble_rest_mean - chosen_means

    last_big_term = torch.sum(mu2_sub_mu1 * mu2_sub_mu1 / ensemble_rest_std, 1)

    uncertainty_score = 1 / 2 * (log_term - n + tr_term + last_big_term)
    assert torch.sum(torch.isinf(uncertainty_score)) == 0
    assert torch.sum(torch.isnan(uncertainty_score)) == 0

    return uncertainty_score


def evaluate(
        env: gym.Env,
        agent: SACAgent,
        num_episodes: int,
        video_recorder: VideoRecorder,
) -> float:
    """We want to evaluate the agent.
    Uses agent to act in environemnt. Calculates the mean reward over the episodes.
    Also uses videorecorder to capture agents behaviour in environment.

    Args:
        env (gym.Env): The environment of the evaluations
        num_episodes (int): Number of episodes to evaluate the agent
        agent (SACAgent): Agent to evaluate
        video_recorder (VideoRecorder): Videorecorder which captures the actions in environment

    Returns:
        (float): The average reward of the num_episode episodes
    """
    avg_episode_reward = 0
    video_recorder.init(enabled=True)
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward

        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def change_capacity_replay_buffer(
        sac_buffer: Optional[mbrl.util.ReplayBuffer],
        obs_shape: Sequence[int],
        act_shape: Sequence[int],
        new_capacity: int,
        seed: int,
) -> mbrl.util.ReplayBuffer:
    """If the given sac_buffer is None, a new ReplayBuffer is created.
    Else the exisiting sac_buffers size is changed, exisiting data will be kept.

    Args:
        sac_buffer (mbrl.util.ReplayBuffer): Given replay_buffer which size should be changed to new_capacity.
        If None then new buffer will be created with obs_shape and act_shape as transition dimensions
        and new_capacity as number of transitions which can be stored
        obs_shape (Sequence[int]): Shape of observation and next_observation in transition
        act_shape (Sequence[int]): Shape of action in transition
        new_capacity (int): How many transitions can now be stored in buffer

    Returns:
        (float): The average reward of the num_episode episodes
    """
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        # sac buffer needs to be created
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
            new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
            return new_buffer
        # capacity needs to be increased
        else:
            rng = sac_buffer.rng
            new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
            obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
            new_buffer.add_batch(obs, action, next_obs, reward, done)
            return new_buffer
    return sac_buffer


def train(
        env: gym.Env,
        test_env: gym.Env,
        termination_fn: mbrl.types.TermFnType,
        cfg: omegaconf.DictConfig,
        silent: bool = False,
        work_dir: Optional[str] = None,
) -> np.float32:
    """ This is the starting point for the mbpo algorithm. We will learn on the env environment and test agents
    performance on test_env. We interchange model_training and agent_training. The model is trained using experienced
    trajectories in the real environment using the current agent. After that the agent is trained using artificial
    roulouts using the learned model.

    Args:
        env (gym.Env): The environment used to learn the model
        test_env (gym.Env): The environment used to evaluate the model and the agent after each epoch
        It seems to be the same es env.
        termination_fn (mbrl.types.TermFnType): Function which returns if state is terminal state or not
        cfg (omegaconf.DictConfig): Complete configuration of algorithm
        See mbpo_cfg_explained.txt for configuration details.
        silent (bool): True if the log should output something or false if not
        work_dir (Optional[str]) The current working directory

    Returns:
        (float): Best reward after evaluation
    """
    # ------------------- Initialization -------------------
    # ------------------------------------------------------
    if work_dir == None:
        print("Running Vanilla M2AC algorithm from a fresh start!")
        work_dir = os.getcwd()
        load_checkpoints = False
    else:
        print("Running Vanilla M2aC algorithm from a checkpoint!")
        print(f"Using checkpoints from folder {work_dir}")
        load_checkpoints = True

    debug_mode = cfg.get("debug_mode", False)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # ------------------- Create SAC Agent -------------------
    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    logger = mbrl.util.Logger(work_dir)#, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        mbrl.constants.MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    logger.register_group(
        "model_rollout",
        ROLLOUT_LOG_FORMAT,
        color="yellow",
        dump_frequency=1,
    )
    # ------------------- Create Viderecorder -------------------
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    # ------------------- Unify randomness -------------------
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial Model of environment (nothing learned yet) --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    # transform model to environment, just wrapper in order to enable sampling
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    # create a Model Trainer
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger
    )

    # -------------- Create Replay buffer storing transitions of agent in real environment --------------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer_real_env = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )

    # -------------- Gather initial data using random or random initialized SAC-Agent --------------
    # ---------------------------------------------------------
    # --------------------- Check for loading checkpoints---------------------
    # ---------------------------------------------------------
    if load_checkpoints:
        print("Loading ground truth replay buffer")
        replay_buffer_real_env.load(work_dir)
        print("Successfully loaded replay buffer!")
        print("Loading SAC agent networks")
        agent.sac_agent.load_checkpoint(f"{work_dir}/sac.pth")
        print("Successfully loaded SAC agent networks")
        print("Loading model network")
        dynamics_model.load(work_dir)
        print("Successfully loaded model network")
    else:
        # -------------- Gather initial data using random or random initialized SAC-Agent --------------
        random_explore = cfg.algorithm.random_initial_explore
        mbrl.util.common.rollout_agent_trajectories(
            env,
            cfg.algorithm.initial_exploration_steps,
            mbrl.planning.RandomAgent(env) if random_explore else agent,
            {} if random_explore else {"sample": True, "batched": False},
            replay_buffer=replay_buffer_real_env,
        )
    # ---------------------------------------------------------
    # --------------------- Start Training---------------------
    # ---------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # --------------------- Initialization before training starts ---------------------
    # ---------------------------------------------------------------------------------
    # per environment step, but sac_buffer is filled only after each freq_train_model env_steps
    effective_model_rollouts_per_step = cfg.overrides.effective_model_rollouts_per_step
    freq_train_model = cfg.algorithm.freq_train_model
    epoch_length = cfg.overrides.epoch_length
    rollout_batch_size = effective_model_rollouts_per_step * freq_train_model
    trains_per_epoch = int(np.ceil(epoch_length / freq_train_model))
    num_epochs_to_retain_sac_buffer = cfg.overrides.num_epochs_to_retain_sac_buffer
    num_sac_updates_per_step = cfg.overrides.num_sac_updates_per_step
    sac_updates_every_steps = cfg.overrides.sac_updates_every_steps
    real_data_ratio = cfg.algorithm.real_data_ratio
    sac_batch_size = cfg.overrides.sac_batch_size
    masking_rate_default = cfg.algorithm.masking_rate_H1
    model_error_penalty_coefficient = cfg.algorithm.model_error_penalty_coefficient
    # Possibility to increase rollout length as function of epoch when editing cfg.overrides.rollout_schedule
    max_rollout_length = cfg.algorithm.max_rollout_length
    exploration_type_env = cfg.overrides.exploration_type_env

    sac_buffer = None
    best_eval_reward = -np.inf
    updates_made = 0
    # real steps taken in environment
    env_steps = 0
    # full model and agent training phase
    epoch = 0
    total_max_steps_in_environment = cfg.overrides.num_steps
    # we will stop training after we reach our final steps in environment counting over all epochs
    while env_steps < total_max_steps_in_environment:
        # ---------------------------------------------------------------------------------
        # --------------------- Initialization for new epoch ---------------------
        # ---------------------------------------------------------------------------------

        # sac_buffer_capacity can only change when rollout length changes, but we need to adapt it for every epoch
        # capacity for one epoch needed

        # In this calculation cfg.overrides.freq_train_model cancels out. Thus more intuitive but exactly the same:
        # sac_buffer_capacity = rollout_length * effective_model_rollouts_per_step * epoch_length
        sac_buffer_capacity = max_rollout_length * rollout_batch_size * trains_per_epoch
        assert sac_buffer_capacity == max_rollout_length * effective_model_rollouts_per_step * epoch_length
        # capacity is scaled by amount of epochs transitions shall be stored
        # On average only 0.25 of all model rollouts are put into sac buffer
        sac_buffer_capacity *= num_epochs_to_retain_sac_buffer * 0.25
        sac_buffer_capacity = int(sac_buffer_capacity)
        sac_buffer = change_capacity_replay_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )

        obs, done = None, False
        if exploration_type_env == "pink":
            action_noise = cn.powerlaw_psd_gaussian(1, (env.action_space.shape[0], cfg.overrides.epoch_length), random_state=rng)
        for steps_epoch in range(epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            if exploration_type_env == "pink":
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer_eps(
                    env, obs, agent, {}, replay_buffer_real_env, eps=action_noise[:, steps_epoch])
            elif exploration_type_env == "white":
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                    env, obs, agent, {"sample":True}, replay_buffer_real_env
                )
            elif exploration_type_env == "det":
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                    env, obs, agent, {}, replay_buffer_real_env
                )
            else:
                print("Pls choose pink, white or det as exploration_type_env!")
                raise NotImplementedError
            env_steps += 1

            # --------------- Model Training -----------------
            # in each epoch all cfg.overrides.freq_train_model the model is trained and the sac_buffer is filed
            if env_steps % freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model, model_trainer, cfg.overrides, replay_buffer_real_env, work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # generates rollout_length * rollout_batch_size(=freq_train_model * effective_model_rollouts_per_step)
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer_real_env,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    max_rollout_length,
                    rollout_batch_size,
                    masking_rate_default,
                    model_error_penalty_coefficient,
                    logger,
                    env_steps,
                    epoch
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {max_rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(num_sac_updates_per_step):
                use_real_data = rng.random() < real_data_ratio
                which_buffer = replay_buffer_real_env if use_real_data else sac_buffer
                if env_steps % sac_updates_every_steps != 0 or len(which_buffer) < sac_batch_size:
                    break  # only when buffer is full enough to batch start training

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % epoch_length == 0:
                print(f"Epoch ended - env-steps:{env_steps}")
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps - 1,
                        "episode_reward": avg_reward,
                        "rollout_length": max_rollout_length,
                    },
                )

                video_recorder.save(f"{epoch}.mp4")
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1

            obs = next_obs
    return np.float32(best_eval_reward)
