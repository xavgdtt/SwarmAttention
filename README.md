# SwarmAttention
Experiments on merging swarm dynamics with attention mechanisms in transformers.

```
⚠️ This is open work, more developments and documentation to come!
```

## Basic Swarm Attention
[gpt_swarmAtt.py](gpt_swarmAtt.py) trains a custom swarm-based GPT model that generates text based on the tiny Shakespeare dataset.

Outperforms the [nanoGPT model](https://github.com/karpathy/nanoGPT) (implemented in [gpt_original.py](gpt_original.py)):
- lower validation loss
- lower parameter count
- shorter training time (training on Apple silicon GPU with MPS backend)

|  | SwarmAttention | nanoGPT |
|---|---|---|
| Parameters | 10.2M | 10.8M |
| Training Time | 2191 s | t.b.d. (longer!) |
| Validation Loss | 1.4646 | 1.4697 |
| Corresp. Training Loss | 1.1717 | t.b.d. |


  
