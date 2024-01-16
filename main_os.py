import gym
from Brain import DSACAgent
from Common import Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = DSACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    agent.logger = logger

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):

                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state, reward)
                episode_reward += reward
                state = next_state
                if done:
                    break

            # OSINAYN does not add diversity reward UNTIL after the policy is close to the optimal
            train_with_diversity = (episode_reward > params["reward_epsilon"])
            # alternatively using logger.max_episode_reward ?
            losses_list = []
            for step in range(1, 1 + max_n_steps):
                losses = agent.train(diversity_reward=train_with_diversity)
                if losses is not None: losses_list.append(losses)

            if len(losses_list):
                loss_dict = {k: np.average([dic[k] for dic in losses_list]) for k in losses_list[0].keys()}

                logger.log(episode,
                           episode_reward,
                           z,
                           -loss_dict['discriminator_loss'], # log q(z|s)
                           step,
                           np.random.get_state(),
                           env.np_random.get_state(),
                           env.observation_space.np_random.get_state(),
                           env.action_space.np_random.get_state(),
                           *agent.get_rng_states(),
                           )
                logger.log_train(loss_dict, episode)

    else:
        from Common import Play
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        env_name=params["env_name"]
        agent_name=params["agent_name"]
        player.evaluate(folder_name=f"Vid/{env_name}/{agent_name}")
