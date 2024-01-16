# Train
#python main_os.py --agent_name SACa --auto_entropy_tuning --alpha 0.0 --reward_epsilon 10000 --mem_size=100000 --env_name="MountainCarContinuous-v0" --interval=200 --n_skills=1 --do_train
python main_os.py --agent_name OSa --auto_entropy_tuning --alpha 0.0 --reward_epsilon -50 --reward_balance 0.3 --mem_size=100000 --env_name="MountainCarContinuous-v0" --interval=200 --n_skills=5 --do_train
#python main_os.py --agent_name DIAYN --reward_epsilon -10000 --reward_balance 1.0 --mem_size=100000 --env_name="MountainCarContinuous-v0" --interval=200 --n_skills=5 --do_train

# Eval
#python main_os.py --agent_name SACa --auto_entropy_tuning --alpha 0.0  --reward_epsilon -10000 --mem_size=100000 --env_name="MountainCarContinuous-v0" --interval=200 --n_skills=1
#python main_os.py --agent_name OSa --auto_entropy_tuning --alpha 0.0  --reward_epsilon -10000 --reward_balance 0.3 --mem_size=100000 --env_name="MountainCarContinuous-v0" --interval=200 --n_skills=5
#python main_os.py --agent_name DIAYN --reward_epsilon -10000 --reward_balance 1.0 --mem_size=100000 --env_name="MountainCarContinuous-v0" --interval=200 --n_skills=5

