import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class DSACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

        # Tunable Entropy
        self.auto_entropy_tuning = config["auto_entropy_tuning"]
        if self.auto_entropy_tuning:
            self.init_entropy()
            print("Enable auto entropy tuning")

    def init_entropy(self, value=0.):
        self.target_entropy = (
            self.config["alpha"] or -self.config["n_actions"])  # heuristic target entropy
        self.target_entropy = torch.tensor(self.target_entropy, requires_grad=False, device=self.device)
        self.log_alpha = torch.tensor(value, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam(
            [self.log_alpha],
            lr=self.config["lr"],
        )

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state, reward, device="cpu"):
        state = from_numpy(state).float().to(device)
        z = torch.ByteTensor([z]).to(device)
        done = torch.BoolTensor([done]).to(device)
        action = from_numpy(np.asarray([action])).to(device)
        next_state = from_numpy(next_state).float().to(device)
        reward = torch.Tensor([reward]).float().to(device)
        self.memory.add(state, z, done, action, next_state, reward)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        return states, zs, dones, actions, next_states, rewards

    def train(self, diversity_reward=True):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states, env_rewards = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            # Standard SAC: J_V = V(s) - E_{a \sim policy}[Q(s,a) - alpha * log pi(a|s)]
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()
            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            # Calculating the Q-Value target
            # Standard SAC: J_Q = Q(s,a) - (r + gamma * V(s+1))
            # DIAYN: define r = H(A|S,Z) + E_{z \sim p(z), s \sim pi(z)}[log discriminator(z|s) - log p(z)]
            rewards = env_rewards if not self.config["omit_env_rewards"] else 0

            if diversity_reward:
                next_logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
                logq_z_ns = log_softmax(next_logits, dim=-1)
                disciminator_reward = (logq_z_ns.gather(-1, zs).detach() - torch.log(p_z.gather(-1, zs) + 1e-6)).float()
                rewards += self.config["reward_balance"] * disciminator_reward

            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            # Entropy Loss
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = self.config["alpha"]

            # Calculate the policy loss
            # Standard SAC (with reparam trick a \sim policy): J_pi = log pi(a|s) - q(s,a)
            policy_loss = (alpha * log_probs - q).mean()

            # Discriminator loss
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return {
                'policy_loss':policy_loss.item(),
                'value_loss':value_loss.item(),
                'q_loss': 0.5 * (q1_loss + q2_loss).item(),
                'q1_loss':q1_loss.item(),
                'q2_loss':q2_loss.item(),
                'discriminator_loss':discriminator_loss.item(),
                'alpha': alpha.item() if self.auto_entropy_tuning else alpha,
                'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else alpha_loss,
                }

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
