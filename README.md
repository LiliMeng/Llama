# Llama
Overview of Llama for LLMs

## Llama 2: Open Foundation and Fine-Tuned Chat Models
[Paper](https://scontent.fyvr1-1.fna.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=gz4k9p3GxPgQ7kNvgGOjnLZ&_nc_ht=scontent.fyvr1-1.fna&oh=00_AYDtURHKq7Q8GC8LG8JkNTn7SMsPEgETbbeQ-GfakkFd8Q&oe=66ADAA7F)

### Training
<img width="1337" alt="Screenshot 2024-07-30 at 9 53 00 AM" src="https://github.com/user-attachments/assets/e984ed06-5017-4aae-99b9-6af8c12e4759">

### 3.1 Supervised Fine-tuning (SFT)
To ensure the model sequence length is properly filled, we concatenate all the prompts and answers from the training set. A special token is utilized to separate the prompt and answer segments. We use
autoregressive objective and zero-out the loss on tokends from the user prompt, as a result, only backpropage answer tokens. Fine-tune the model for 2 epochs.

### 3.2 Reinforcement Learning with Human Feedback (RLHF)
RLHF is a model training procedure that is applied to a fine-tuned language model to further align model behavior with human preferences and instruction following. We collect data that represents empirically
sampled human preferences, whereby human annotators select which of two model ouputs they prefer. This human feedback is subsequently used to train a reward model, which learns patterns in the preferences of the
human annotators and cna then automate preference decisions.

#### 3.2.1 Human Preference Data Collection

#### 3.2.2 Reward Modeling
The reward model takes a model response and its corresponding prompt (including contexts from previous turns) as inputs and outputs a scalar score to indicate the quality (e.g. helpfulness and safety) of the model generation. Leveraging such response scores as rewards, we can optimize LLAMA2-Chat during RLHF for better human preference alignment and improved helpfulness and safety.

The model architecture and hyper-parameters are identical to those of the pretrained language models, except that the classification head for next-token prediction is replaced with a regression head for outputing a scalar reward.
<img width="1339" alt="Screenshot 2024-07-29 at 5 25 08 PM" src="https://github.com/user-attachments/assets/7500897a-f1cf-4f94-a2f8-49f38c3a0c72">

#### 3.2.3 Iterative Fine-tuning
As we received more batches of human preference data annotation, we were able to train better reward models and collect more prompts. We therefore trained successive versions for RLHF models, referred to RLHF-V1,
...,RLHF-V5.

Two main methds:

**Proximal Policy Optimization (PPO)**

**Rejection Sampling fine-tuning**


### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that aims to improve the training of policies by optimizing a surrogate objective function. It addresses some of the challenges found in other policy optimization methods, such as the complexity and instability of training. PPO is known for its simplicity and effectiveness, making it one of the most popular algorithms in the field of reinforcement learning.

Here’s a detailed explanation of PPO:

### Key Concepts

1. **Policy**:
   - A policy defines the behavior of an agent by mapping states of the environment to actions. In reinforcement learning, the policy is often represented as a neural network that outputs a probability distribution over actions given a state.

2. **Objective Function**:
   - PPO optimizes a surrogate objective function to improve the policy. The goal is to maximize the expected reward by adjusting the policy parameters.

3. **Clipped Surrogate Objective**:
   <img width="684" alt="Screenshot 2024-07-30 at 10 24 31 AM" src="https://github.com/user-attachments/assets/4cac8b4b-b143-41ce-b658-52167b0ba104">


4. **Advantage Function**:
   - The advantage function \(\hat{A}_t\) is used to estimate the relative value of an action compared to the average action at a given state. It helps in deciding which actions are better than the average.

5. **Trust Region**:
   - By clipping the objective, PPO ensures that the new policy does not deviate too much from the old policy. This creates a "trust region" that helps maintain stable updates.

### Algorithm

1. **Initialize**:
   - Start with an initial policy \(\pi_{\theta_{\text{old}}}\) with parameters \(\theta_{\text{old}}\).

2. **Collect Data**:
   - Interact with the environment using the current policy to collect trajectories (states, actions, rewards).

3. **Compute Advantage Estimates**:
   - Use the collected data to compute the advantage estimates \(\hat{A}_t\) for each time step \(t\).

4. **Update Policy**:
   - Optimize the policy by maximizing the clipped surrogate objective \(L^{CLIP}\). Update the policy parameters \(\theta\) using gradient ascent.

5. **Repeat**:
   - Repeat the process by collecting new data with the updated policy and iterating through the optimization steps.

### Advantages of PPO

1. **Stability and Reliability**:
   - The clipping mechanism prevents large updates, ensuring stable learning and avoiding performance collapse.

2. **Simplicity**:
   - PPO is relatively simple to implement compared to other advanced policy optimization methods like Trust Region Policy Optimization (TRPO).

3. **Sample Efficiency**:
   - PPO can be more sample-efficient due to its ability to reuse data for multiple epochs of optimization.

### Applications

PPO is widely used in various applications of reinforcement learning, including:
- Robotics: Training robots for tasks like manipulation, navigation, and locomotion.
- Game Playing: Developing agents to play complex games like Dota 2, chess, and Go.
- Autonomous Vehicles: Training policies for self-driving cars to navigate safely.

### Example

Here’s a simplified code snippet to give an idea of how PPO might be implemented:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

def ppo_update(policy, optimizer, trajectories, clip_epsilon=0.2, epochs=10):
    for _ in range(epochs):
        for state, action, old_log_prob, advantage in trajectories:
            new_log_prob = policy(state).log_prob(action)
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            loss = -torch.min(surr1, surr2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Example usage
state_dim = 4
action_dim = 2
policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Assuming `trajectories` is a list of (state, action, old_log_prob, advantage) tuples
trajectories = [...]
ppo_update(policy, optimizer, trajectories)
```

In this snippet:
- A simple policy network is defined.
- The `ppo_update` function performs the PPO update using the clipped objective.

PPO is a powerful algorithm that balances simplicity and performance, making it a go-to choice for many reinforcement learning tasks.
