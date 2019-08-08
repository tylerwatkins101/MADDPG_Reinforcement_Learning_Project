## Learning Algorithm

The learning algorithm used to train the agent was a Multi-Agent Deep Deterministic Policy Gradient (MADDPG).

Deep deterministic policy gradient (DDPG) is a variant of DPG where the policy and critic Q are approximated with deep neural networks. DDPG is an off-policy algorithm, and samples trajectories from a replay buffer of experiences that are stored throughout training. DDPG also makes use of a target network, as in DQN.

MADDPG extends DDPG even further toinclude multiple agents. Uses multiple agents to develop the critic network while relying on it's own actions to build the Actor network.



#### The Model Architecture for the Actor Network:

- Inputs = State Space Size (33)
- Outputs = Action Space Size (4)

- Linear Layer 1 (inputs = 33, outputs = 400)
- Relu Activation Function
- Batch Normalization Layer
- Linear Layer 2 (inputs = 400, outputs = 300)
- Relu Activation Function
- Linear Layer 3 (inputs = 300, outputs = 4)
- Tanh Activation Function

![Actor](photos/Actor.png)

#### The Model Architecture for the Critic Network:

- Inputs = State Space Size (33)
- Outputs = Action Space Size (4)

- Linear Layer 1 (inputs = 33, outputs = 400)
- Relu Activation Function
- Batch Normalization Layer
- Concatenation Layer(layers = 400 + 4 = 404)
- Linear Layer 2 (inputs = 404, outputs = 300)
- Relu Activation Function
- Linear Layer 3 (inputs = 300, outputs = 4)

![Critic](photos/Critic.png)

#### The Hyperparameters:

- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-5         # learning rate of the actor
- LR_CRITIC = 1e-4        # learning rate of the critic
- WEIGHT_DECAY = 0        # L2 weight decay
- UPDATE_NETWORK_FREQUENCY = 3      # how many steps the network is updated
- THETA: 0.15             # revert speed to mean
- SIGMA: 0.2              # degree of volatility

## Plot of Rewards

Here we see a plot of rewards per training episode to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +0.5. The environment was solved in 2311 episodes.


![reward_plot](photos/MADDPG_Plot.png)

## Ideas for Future Work

There are many ways that this agent could be improved in the future. I will list a few ideas here that could be tested.

1. Experiment more with the actor and critic network architectures. More layers, more nodes, batchnorm, dropout.
2. Experiment more with the model training hyperparameters. Batch_size, LR, sigma, weight decay.
3. Attempt to improve upon the MADDPG algorithm as suggested in the paper, "Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments" in the following way:

"We have proposed a multi-agent policy gradient algorithm where agents learn a centralized critic
based on the observations and actions of all agents. Empirically, our method outperforms traditional
RL algorithms on a variety of cooperative and competitive multi-agent environments. We can further
improve the performance of our method by training agents with an ensemble of policies, an approach
we believe to be generally applicable to any multi-agent algorithm.
One downside to our approach is that the input space of Q grows linearly (depending on what
information is contained in x) with the number of agents N. This could be remedied in practice by,
for example, having a modular Q function that only considers agents in a certain neighborhood of a
given agent. We leave this investigation to future work."

## References

[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
