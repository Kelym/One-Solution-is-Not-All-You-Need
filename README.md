[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  

# One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL

**This repository is a PyTorch implementation of _One Solution is Not All You Need_**

**The DIAYN part of the code is based on [this repo](https://github.com/alirezakazemipour/DIAYN-Pytorch).**

Changes:
- Automatic tuning of entropy alpha
- Follow One Solution paper to consider env rewards when training the policy

## Dependencies
- gym == 0.21
- mujoco-py == 2.1.2.14
- numpy == 1.23.3
- opencv_contrib_python == 4.6.0
- psutil == 5.9.2
- torch == 1.12.1
- tqdm == 4.64.1

## Installation
```bash
pip3 install -r requirements.txt
```

## Usage

`train.sh MountainCarContinuous-v0`:

```bash
python main_os.py --agent_name SACa --reward_epsilon 10000 --mem_size=100000 --env_name="$1" --n_skills=1 --do_train --auto_entropy_tuning --alpha 0.0
```

## Reference

1. [_One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL_, Kumar, 2020](https://arxiv.org/abs/2010.14484)
2. [_Diversity is All You Need: Learning Skills without a Reward Function_, Eysenbach, 2018](https://arxiv.org/abs/1802.06070)

## Acknowledgment

Most of the repo is based on [@alirezakazemipour](https://github.com/alirezakazemipour) implementation of [DIAYN](https://github.com/alirezakazemipour/DIAYN-PyTorch)

1. [@ben-eysenbach ](https://github.com/ben-eysenbach) for [sac](https://github.com/ben-eysenbach/sac).
2. [@p-christ](https://github.com/p-christ) for [DIAYN.py](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/hierarchical_agents/DIAYN.py).
3. [@johnlime](https://github.com/johnlime) for [RlkitExtension](https://github.com/johnlime/RlkitExtension).
4. [@Dolokhow](https://github.com/Dolokhow) for [rl-algos-tf2 ](https://github.com/Dolokhow/rl-algos-tf2).
