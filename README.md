# SwarmAttention
Experiments on merging swarm dynamics with attention mechanisms in transformers.

## gpt_swarmAtt.py
Trains a custom GPT model that generates text based on the tiny Shakespeare dataset.

Outperforms the [nanoGPT model](https://github.com/karpathy/nanoGPT) (implemented in gpt_original.py):
- lower validation loss
- lower parameter count
- shorter training time (training on Apple silicon GPU with MPS backend)


  
