import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="BipedalWalker-v3", type=str, help="Name of the environment.")
    parser.add_argument("--agent_name", default="", type=str, help="Name of the agent.")
    parser.add_argument("--interval", default=100, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--train_from_scratch", action="store_false",
                        help="The flag determines whether to train from scratch or continue previous tries.")
    parser.add_argument("--omit_env_rewards", action="store_true",
                        help="Omit environment default rewards.")
    parser.add_argument("--mem_size", default=int(1e+6), type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=50, type=int, help="The number of skills to learn.")
    parser.add_argument("--alpha", default=0.1, type=float, help="The SAC temperature. Alternatively, if auto entropy tuning, set the target entropy.")
    parser.add_argument("--auto_entropy_tuning", action="store_true", help="Enable automatic tuning of the SAC temperature.")
    parser.add_argument("--reward_scale", default=1, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--reward_balance", default=0.4, type=float, help="The reward balancing factor introduced in OneSolutionIsNotAllYouNeed.")
    parser.add_argument("--reward_epsilon", default=-1000, type=float, help="When to kick in diversity, as described in OneSolutionIsNotAllYouNeed.")
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")

    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    # region default parameters
    default_params = {"lr": 3e-4,
                      "batch_size": 256,
                      "max_n_episodes": 5000,
                      "max_episode_len": 1000,
                      "gamma": 0.99,
                      "tau": 0.005,
                      "n_hiddens": 300
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    return total_params
