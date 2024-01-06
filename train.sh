python main_os.py --agent_name SAC --reward_epsilon 100000 --mem_size=100000 --env_name="BipedalWalker-v3" --interval=200 --n_skills=10 --do_train
python main_os.py --agent_name OS --reward_epsilon -50 --reward_balance 0.3 --mem_size=100000 --env_name="BipedalWalker-v3" --interval=200 --n_skills=10 --do_train
python main_os.py --agent_name DIAYN --reward_epsilon -10000 --reward_balance 1.0 --mem_size=100000 --env_name="BipedalWalker-v3" --interval=200 --n_skills=10 --do_train

