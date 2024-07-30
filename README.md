# Llama
Overview of Llama for LLMs

## Llama 2: Open Foundation and Fine-Tuned Chat Models
[Paper](https://scontent.fyvr1-1.fna.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=gz4k9p3GxPgQ7kNvgGOjnLZ&_nc_ht=scontent.fyvr1-1.fna&oh=00_AYDtURHKq7Q8GC8LG8JkNTn7SMsPEgETbbeQ-GfakkFd8Q&oe=66ADAA7F)


### Reward Modeling
The model architecture and hyper-parameters are identical to those of the pretrained language models, except that the classification head for next-token prediction is replaced with a regression head for outputing a scalar reward.
<img width="1339" alt="Screenshot 2024-07-29 at 5 25 08 PM" src="https://github.com/user-attachments/assets/7500897a-f1cf-4f94-a2f8-49f38c3a0c72">
