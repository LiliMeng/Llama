# Llama
Overview of Llama for LLMs

Here is a timeline showing the development of important large language models (LLMs):

1. **GPT-1**
   [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): June 2018
   
2. **GPT-2**
   [Language Models are Few-Shot Learners](https://openai.com/blog/better-language-models/): February 2019
   
3. **GPT-3**
   "Language Models are Few-Shot Learners" [GPT-3](https://arxiv.org/abs/2005.14165) June 2020
   

4. **ChatGPT** (based on GPT-3.5) [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/research/chatgpt) November 2022
   
5. **GPT-4** [Introducing GPT-4](https://openai.com/research/gpt-4) March 2023
  

6. **LLaMA 1, LLaMA 2, LLaMA 3**
- **LLaMA 1 (2023 Feb)**
- **LLaMA 2 (2023 July)**
- **LLaMA 3 (2024 July)**

For GPT Series, please refer to https://github.com/LiliMeng/LLMs/blob/main/README.md

## Llama 2: Open Foundation and Fine-Tuned Chat Models
[Paper](https://scontent.fyvr1-1.fna.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=gz4k9p3GxPgQ7kNvgGOjnLZ&_nc_ht=scontent.fyvr1-1.fna&oh=00_AYDtURHKq7Q8GC8LG8JkNTn7SMsPEgETbbeQ-GfakkFd8Q&oe=66ADAA7F)

Google Colab example LLAMA Superviesed fine tuning [link](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=x-xPb-_qB0dz)

Google Colab example on RLHF ([link1 from UC Bekeley homework, some cannot run](https://colab.research.google.com/drive/1NqsWBgl7rJsYhP1AsmBgGwVYj-6_Pp1P)) ([link2 some cannot run](https://colab.research.google.com/github/heartexlabs/RLHF/blob/master/tutorials/RLHF_with_Custom_Datasets.ipynb))([Fine-tuning and evaluating GPT-3.5 with human feedback for RAG](https://colab.research.google.com/github/argilla-io/argilla/blob/main/docs/_source/tutorials_and_integrations/tutorials/feedback/fine-tuning-openai-rag-feedback.ipynb))

#### LLaMA 2:
- **General-Purpose Model**: Designed for a wide range of natural language processing tasks such as text generation, summarization, translation, and more.
- **Pre-training**: Trained on diverse datasets without specific focus on conversational data.

#### LLaMA 2-Chat:
- **Fine-Tuned for Dialogue**: Specifically optimized for chat and dialogue-based applications.
- **Reinforcement Learning with Human Feedback (RLHF)**: Fine-tuned using human feedback to enhance conversational abilities, coherence, and relevance in dialogue.

In summary, LLaMA 2 is a versatile model for general NLP tasks, while LLaMA 2-Chat is specialized for generating high-quality conversational responses.

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
Rejection sampling is only with largest 70B LLAMA 2-Chat. All smaller models are fine-tuned on rejection sampled data from the larger model, thus distilling the large-model capabilities into the smaller ones.

Until RLHF(V4), only Rejection Sampling fine-tuning is used, after that, we combined the two sequentially, applying PPO on top of the resulted Rejection Sampling checkpoint before sampling again.

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
<img width="718" alt="Screenshot 2024-07-30 at 11 11 23 AM" src="https://github.com/user-attachments/assets/b69fbf50-d2b3-42ea-be72-6bce69a8ee49">


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

### Rejection Sampling fine-tuning
Rejection Sampling fine-tuning in Reinforcement Learning with Human Feedback (RLHF) is a technique to enhance model training by leveraging human judgments to select high-quality outputs. Here's a step-by-step explanation:

1. **Generate Outputs**: The model generates multiple outputs for a given input.
2. **Human Evaluation**: Human evaluators rank these outputs based on quality, relevance, and appropriateness.
3. **Select Best Outputs**: The highest-ranked outputs are accepted, while lower-ranked outputs are rejected.
4. **Update Model**: The accepted outputs are used to fine-tune the model, reinforcing desirable behaviors and improving overall performance.

This process ensures that the model learns from high-quality examples, aligning its behavior with human preferences.

### Example Code

```python
import random

class SimpleModel:
    def __init__(self):
        self.parameters = {"weight": 1.0}

    def generate_output(self, input_data):
        # Simple linear model for demonstration
        return self.parameters["weight"] * input_data

    def update_parameters(self, accepted_samples):
        # Update parameters based on accepted samples
        total = sum(accepted_samples)
        self.parameters["weight"] = total / len(accepted_samples)

def generate_samples(model, input_data, num_samples=10):
    samples = []
    for _ in range(num_samples):
        output = model.generate_output(input_data)
        samples.append(output + random.uniform(-1, 1))  # Add some noise
    return samples

def evaluate_samples(samples):
    # Simulate human evaluation with random scores
    scores = [random.uniform(0, 1) for _ in samples]
    ranked_samples = sorted(zip(samples, scores), key=lambda x: x[1], reverse=True)
    return ranked_samples

def rejection_sampling_fine_tuning(model, input_data, num_samples=10, acceptance_ratio=0.3):
    samples = generate_samples(model, input_data, num_samples)
    ranked_samples = evaluate_samples(samples)
    
    # Select top N% of samples
    num_accepted = int(acceptance_ratio * num_samples)
    accepted_samples = [sample for sample, score in ranked_samples[:num_accepted]]
    
    # Update the model with accepted samples
    model.update_parameters(accepted_samples)
    return model

# Example usage
model = SimpleModel()
input_data = 5.0  # Example input
model = rejection_sampling_fine_tuning(model, input_data)

print(f"Updated model parameters: {model.parameters}")
```

### Explanation

1. **SimpleModel**: A basic model that generates outputs based on input data.
2. **generate_samples**: Generates multiple noisy outputs for a given input.
3. **evaluate_samples**: Simulates human evaluation by assigning random scores and ranking the samples.
4. **rejection_sampling_fine_tuning**: Uses the top-ranked samples to fine-tune the model.

This code provides a basic framework for rejection sampling fine-tuning, demonstrating how to incorporate human feedback into model training. In a real-world scenario, the evaluation would be done by actual human annotators, and the model would be more complex.

## The Llama 3 Herd of Models
[Paper](https://scontent.fyvr1-1.fna.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=DTS7hDTcxZoQ7kNvgGa2Q8D&_nc_ht=scontent.fyvr1-1.fna&gid=Alq3bgijlK6y7Ed9LYFJBNZ&oh=00_AYAPD1I37SucEQSalgQevVvcfqKSjqJQAJYnJIVxAJiF8g&oe=66B0260D)
### Three key levers in the development of high-quality foundation models:
1. **Data**:
   Compared to prior versions of Llama, both quantity and quality of the data used for pre-training and post-training are improved. Include: the development of more careful pre-processing and curation
   pipelines for pre-training and more rigorous quality assurance and filtering approaches for post-training data. We pre-train Llama 3 on 15T multilingual tokens, compared to 1.8T tokens for Llama2.
2. **Scale**:
   We train a model at far larger scale than previous Llama models: our flagship language model was pre-trained using 3.8x10^25 FLOPs, almost 50X more than largest version of Llama2. Pre-train a flagship
   model with 405B params on 15.6T tokens. As scaling laws for foundation models, the flagship model outperforms smaller models trained using the same procedure.
3. **Managing complexity**:
   We make design choices that seek to maximize the ability to scale the model development process. Use a standard dense Transformer with minor adaptations, rather than a mixture-of-experts to maximize training      stability. Post-training procedure based on supervised fine-tuning (SFT), rejection sampling (RS), and direct preference optimization (DPO) as opposed to more complex reinforcement learning alogrithms that tend    to be less stabel and harder to scale.
