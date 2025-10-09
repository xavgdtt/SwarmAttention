# SwarmAttention
Experiments on merging swarm dynamics with attention mechanisms in transformers.

## gpt_swarmAtt.py
Trains a custom GPT model that generates text based on the tiny Shakespeare dataset.

Outperforms the [nanoGPT model](https://github.com/karpathy/nanoGPT) (implemented in gpt_original.py):
- lower validation loss (original: 1.4816, training loss 1.1199)
- lower parameter count (original: 10.8M)
- shorter training time (training on Apple silicon GPU with MPS backend) (original: 2714.54s for min loss)


  
