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
import mbrl.util.replay_buffer
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder

import mbrl.util.mujoco
import mbrl.util.distance_measures as dm
import colorednoise as cn

MBPO_LOG_FORMAT = [
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

def rollout_model_and_populate_sac_buffer(
        rng,
        model_env: mbrl.models.ModelEnv,
        replay_buffer: mbrl.util.ReplayBuffer,
        agent: SACAgent,
        sac_buffer: mbrl.util.ReplayBufferDynamicLifeTime,
        sac_samples_action: bool,
        max_rollout_length: int,
        batch_size: int,
        current_border_count,
        current_border_estimate: float,
        pink_noise_exploration_mod: bool=False,
        xi:float = 1.0,
        zeta: int = 95,

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
    """
    batch = replay_buffer.sample(batch_size)
    # intial_obs ndarray batchsize x observation_size
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    # model_state tensor batchsize x observation_size
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    obs = initial_obs
    uncertainty_scores_for_each_rollout_sorted = []
    number_of_certain_transitions_each_rollout = []

    new_sac_size = 0
    rollout_tracker = np.zeros((0,))
    certain_bool_map_over_all_rollouts = np.zeros(obs.shape[0], dtype=bool)

    if pink_noise_exploration_mod:
        action_noise = cn.powerlaw_psd_gaussian(1, (model_env.action_space.shape[0], max_rollout_length), random_state=rng)
    for i in range(max_rollout_length):
        # action is of type ndarray batchsize x actionsize, action is sampled of SAC Gaussian
        if pink_noise_exploration_mod:
            action = agent.act_eps(obs, eps=action_noise[:, i], sample=sac_samples_action, batched=True)
        else:
            action = agent.act(obs, sample=sac_samples_action, batched=True)

        # -------------------------------------------------------------------#

        # -------------------------------------------------------------------#
        # Calculate the transitions using the model
        # Make step in environment(ModelEnv->1DTransitionrewardModel->GaussianMLP->Ensemble) and get the
        # predicted obs and rewards. Also get dones and model_state.
        # For rm2ac the means and vars of the next_obs+reward are needed so get them too.
        # chosen_means,chosen_stds are of size (batchsize) and are the mean and stds of the gaussian
        # that was chosen to sample pred_next_obs, pred_rewards
        # means_of_all_ensembles, stds_of_all_ensembles are (ensemble_size x batchsize) and are all means
        # and stds of all gaussians in the ensemble
        # model_indices is (batchsize) and is the chosen model_indices of the ensemebles [0,ensemble_size)
        (pred_next_obs, pred_rewards, pred_dones, model_state,
         chosen_means, chosen_stds, means_of_all_ensembles,
         stds_of_all_ensembles, model_indices) = model_env.step_plus_gaussians(action, model_state, sample=True)

        ensemble_size = model_env.dynamics_model.model.ensemble_size

        vars_of_all_ensembles = torch.pow(stds_of_all_ensembles, 2)
        # -------------------------------------------------------------------#


        # -------------------------------------------------------------------#

        jsp = dm.calc_pairwise_symmetric_uncertainty_for_measure_function(means_of_all_ensembles,
                                                                              vars_of_all_ensembles,
                                                                              ensemble_size,
                                                                              dm.calc_uncertainty_score_genShen)
        uncertainty_score = jsp

        # -------------------------------------------------------------------#
        # Calculate the uncertainty threshhold. If some non zero uncertainty threshold was chosen it is used to filter
        # the generated transitions. For a zero threshold the current_border_estimate is used to filter the data,
        # it is the average over a fixed number of past border_for_this_rollout values and is given to the rollout
        # function

        if i == 0:
            zeta_percentile = np.percentile(uncertainty_score, zeta)
            border_for_this_rollout = zeta_percentile * xi
            threshold = 1 / (current_border_count + 1) * border_for_this_rollout + current_border_count / (
                        current_border_count + 1) * current_border_estimate
            print(f"Max Uncertainty of {zeta} percentile times {xi} factor: {border_for_this_rollout}")
            print(f"Updated Uncertainty threshhold is {threshold}")
            reduce_time = True
        else:
            reduce_time = False


        indices_of_certain_transitions = uncertainty_score < threshold

        # certain_bool_map contains true for storing transition if it is certain enough and false else
        if i ==0:
            certain_bool_map_over_all_rollouts[indices_of_certain_transitions] = True
        else:
            certain_bool_map_this_rollout = np.zeros(obs.shape[0], dtype=bool)
            certain_bool_map_this_rollout[indices_of_certain_transitions] = True
            certain_bool_map_over_all_rollouts = np.logical_and(certain_bool_map_this_rollout, certain_bool_map_over_all_rollouts)

        number_of_certain_transitions = certain_bool_map_over_all_rollouts.sum()

        rollout_tracker = np.append(rollout_tracker, np.full((obs.shape[0] - number_of_certain_transitions), i))
        new_sac_size = new_sac_size + number_of_certain_transitions
        if number_of_certain_transitions == 0:
            endOfRollout = i
            break

        ind_sort_un = np.argsort(uncertainty_score)
        uncertainty_scores_for_each_rollout_sorted.append(uncertainty_score[ind_sort_un])
        number_of_certain_transitions_each_rollout.append(number_of_certain_transitions)
        assert np.sum(np.isinf(pred_rewards[:, 0])) == 0
        assert np.sum(np.isnan(pred_rewards[:, 0])) == 0

        # Add the filtered rollouts to the SAC Replay Buffer
        sac_buffer.add_batch(
            obs[certain_bool_map_over_all_rollouts],
            action[certain_bool_map_over_all_rollouts],
            pred_next_obs[certain_bool_map_over_all_rollouts],
            pred_rewards[certain_bool_map_over_all_rollouts, 0],# pred_rewards and pred_done
            pred_dones[certain_bool_map_over_all_rollouts, 0],# need to be of size batchsize not batchsize x 1
            reduce_time=reduce_time #is true for i==0 and serves the purpose to reduce the lifetime of the stored items in replay buffer
        )
        # squeezing to transform pred_dones from batch_size x 1 to batchsize
        certain_bool_map_over_all_rollouts = np.logical_and(~(pred_dones.squeeze()),
                                                            certain_bool_map_over_all_rollouts)
        obs = pred_next_obs
        batch_size = obs.shape[0]
        model_state = model_env.reset(
            initial_obs_batch=cast(np.ndarray, obs),
            return_as_np=True,
        )

    return new_sac_size, border_for_this_rollout


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
        counter = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
            counter += 1

        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def change_capacity_replay_buffer(
        sac_buffer: Optional[mbrl.util.ReplayBufferDynamicLifeTime],
        obs_shape: Sequence[int],
        act_shape: Sequence[int],
        new_capacity: int,
        seed: int,
        lifetime: int,
) -> mbrl.util.ReplayBufferDynamicLifeTime:
    """If the given sac_buffer is None, a new ReplayBuffer is created.
    Else the existing sac_buffers size is changed, existing data will be kept.

    Args:
        sac_buffer (mbrl.util.ReplayBuffer): Given replay_buffer which size should be changed to new_capacity.
        If None then new buffer will be created with obs_shape and act_shape as transition dimensions
        and new_capacity as number of transitions which can be stored
        obs_shape (Sequence[int]): Shape of observation and next_observation in transition
        act_shape (Sequence[int]): Shape of action in transition
        new_capacity (int): How many transitions can now be stored in buffer
        lifetime (int): number of rollouts until data is deleted

    Returns:
        (float): The average reward of the num_episode episodes
    """
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        # sac buffer needs to be created
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
            new_buffer = mbrl.util.ReplayBufferDynamicLifeTime(new_capacity, obs_shape, act_shape, lifetime, rng=rng)
            return new_buffer
        # capacity needs to be increased
        else:
            rng = sac_buffer.rng
            new_buffer = mbrl.util.ReplayBufferDynamicLifeTime(new_capacity, obs_shape, act_shape, lifetime, rng=rng)
            obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
            new_buffer.add_batch(obs, action, next_obs, reward, done)
            return new_buffer
    return sac_buffer


def train(
        env: gym.Env,
        test_env: gym.Env,
        distance_env: gym.Env,
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
        distance_env (gym.Env): The environment used to track the real next states for rollouts
        termination_fn (mbrl.types.TermFnType): Function which returns if state is terminal state or not
        cfg (omegaconf.DictConfig): Complete configuration of algorithm
        See mbpo_cfg_explained.txt for configuration details.
        silent (bool): True if the log should output something or false if not
        work_dir (Optional[str]) The current working directory

    Returns:
        (float): Best reward after evaluation
    """


    # -------------------------------------------------------------------#

    # -------------------------------------------------------------------#
    # ------------------- Initialization -------------------

    if work_dir == None:
        print("Running MACURA algorithm from a fresh start!")
        work_dir = os.getcwd()

    max_rollout_length = cfg.algorithm.max_rollout_length
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # ------------------- Create SAC Agent -------------------
    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    # ------------------- Create Logger -------------------
    logger = mbrl.util.Logger(work_dir)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        mbrl.constants.MBPO_LOG_FORMAT,
        color="green",
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

    real_experienced_states_full = []

    # -------------- Gather initial data using random or random initialized SAC-Agent --------------
    random_explore = cfg.algorithm.random_initial_explore

    mbrl.util.common.rollout_agent_trajectories_Tracking_States(
        real_experienced_states_full,
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
    # number of effective added transitions per environment step, but sac_buffer is filled only after each freq_train_model env_steps
    effective_model_rollouts_per_step = cfg.overrides.effective_model_rollouts_per_step
    freq_train_model = cfg.algorithm.freq_train_model
    epoch_length = cfg.overrides.epoch_length
    rollout_batch_size = effective_model_rollouts_per_step * freq_train_model
    num_epochs_to_retain_sac_buffer = cfg.overrides.num_epochs_to_retain_sac_buffer
    num_sac_updates_per_step = cfg.overrides.num_sac_updates_per_step
    sac_updates_every_steps = cfg.overrides.sac_updates_every_steps
    real_data_ratio = cfg.algorithm.real_data_ratio
    sac_batch_size = cfg.overrides.sac_batch_size

    unc_tresh_run_avg_history = cfg.overrides.unc_tresh_run_avg_history
    pink_noise_exploration_mod = cfg.overrides.pink_noise_exploration_mod
    exploration_type_env = cfg.overrides.exploration_type_env
    xi = cfg.overrides.xi
    zeta = cfg.overrides.zeta


    sac_buffer = None
    best_eval_reward = -np.inf
    updates_made = 0
    # real steps taken in environment
    env_steps = 0
    # full model and agent training phase
    epoch = 0
    total_max_steps_in_environment = cfg.overrides.num_steps

    # we will stop training after we reach our final steps in environment counting over all epochs
    current_border_count_position = 0
    current_border_count = 0
    current_border_estimate = 0
    current_border_estimate_list_full = False
    # Number of maximum border_for_this_rollout_values to average to get the uncertainty threshold
    Max_Count = unc_tresh_run_avg_history
    # Max_Count = int(total_max_steps_in_environment / 5000 * (epoch_length / freq_train_model))
    # here are these values safed
    current_border_estimate_list = np.empty(Max_Count)
    while env_steps < total_max_steps_in_environment:
        # ---------------------------------------------------------------------------------
        # --------------------- Initialization for new epoch ---------------------
        # ---------------------------------------------------------------------------------
        # Because cfg.overrides.freq_train_model cancels out two ways to calculate the capacity
        # sac_buffer_capacity = max_rollout_length * effective_model_rollouts_per_step * epoch_length
        # sac_buffer_capacity = max_rollout_length * rollout_batch_size * trains_per_epoch

        trains_per_epoch = epoch_length / freq_train_model
        sac_buffer_capacity = max_rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity = sac_buffer_capacity * num_epochs_to_retain_sac_buffer
        sac_buffer_capacity = int(sac_buffer_capacity)
        lifetime = num_epochs_to_retain_sac_buffer * trains_per_epoch
        print("Sac Buffer Cap:")
        print(sac_buffer_capacity)
        sac_buffer_capacity = int(sac_buffer_capacity)
        sac_buffer = change_capacity_replay_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed, lifetime
        )

        obs, done = None, False
        if exploration_type_env == "pink":
            action_noise = cn.powerlaw_psd_gaussian(1, (env.action_space.shape[0], cfg.overrides.epoch_length),
                                                    random_state=rng)
        for steps_epoch in range(epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            if exploration_type_env == "pink":
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer_eps(
                    env, obs, agent, {}, replay_buffer_real_env, eps=action_noise[:, steps_epoch])
            elif exploration_type_env == "white":
                next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                    env, obs, agent, {"sample": True}, replay_buffer_real_env
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

                # Start Model Training
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model, model_trainer, cfg.overrides, replay_buffer_real_env, work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # generates maximally rollout_length * rollout_batch_size
                # (=freq_train_model * effective_model_rollouts_per_step) new transitions for SAC buffer
                new_sac_size, current_border_estimate_update = rollout_model_and_populate_sac_buffer(
                    rng,
                    model_env,
                    replay_buffer_real_env,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    max_rollout_length,
                    rollout_batch_size,
                    current_border_count,
                    current_border_estimate,
                    pink_noise_exploration_mod,
                    xi,
                    zeta
                )

                current_border_estimate_list[current_border_count_position] = current_border_estimate_update
                if current_border_count_position == Max_Count - 1:
                    current_border_estimate_list_full = True
                    current_border_count = Max_Count - 1
                else:
                    current_border_count = current_border_count_position + 1
                current_border_count_position = (current_border_count_position + 1) % Max_Count
                if current_border_estimate_list_full:
                    current_border_estimate = np.mean(current_border_estimate_list)
                else:
                    current_border_estimate = np.mean(current_border_estimate_list[0:current_border_count_position])

            # --------------- Agent Training -----------------
            # here is a formula which controlls learning steps proportionally to the filling of the SAC buffer
            dynamic_updates_per_step = int(
                ((sac_buffer.num_stored * 2) / sac_buffer.capacity) * num_sac_updates_per_step)
            for _ in range(dynamic_updates_per_step):
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

            if env_steps % epoch_length == 0:
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
                if save_video:
                    video_recorder.save(f"{epoch}.mp4")
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1
            obs = next_obs
    return np.float32(best_eval_reward)
