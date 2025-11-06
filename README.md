# SwarmAttention
Experiments on merging swarm dynamics with attention mechanisms in transformers.
Best results currently produced by the forked version with swarm loss. Work in progress.

```
⚠️ This is open work, more developments and documentation to come!
```

## Swarm Attention
[gpt_swarmAtt.py](gpt_swarmAtt.py) trains a custom swarm-based GPT model that generates text based on the tiny Shakespeare dataset.

Outperforms the [nanoGPT model](https://github.com/karpathy/nanoGPT) (implemented in [gpt_original.py](gpt_original.py)):
- lower validation loss **-2%**
- lower parameter count **-5%**
- shorter training time (training on Apple silicon GPU with MPS backend) **-50%**

|  | SwarmAttention | nanoGPT |
|---|---|---|
| Parameters | 10.2M | 10.8M |
| Training Time | 2371 s | 4000+ s |
| Validation Loss | 1.4625 | 1.4920 |
| Corresp. Training Loss | 1.1684 | 1.1718 |


  
