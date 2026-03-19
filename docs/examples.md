# Examples

## CNN architecture search (FashionMNIST)

Full search over 14 parameters — architecture and training hyperparameters.

```bash
neuropt run examples/train_fashion.py --backend claude
```

Source: [`examples/train_fashion.py`](https://github.com/loevlie/neuropt/blob/main/examples/train_fashion.py)

Searches over: block count, channel width, channel growth, kernel size, activation, residual connections, batch norm, dropout, pooling strategy, FC head size, learning rate, weight decay, optimizer.

## ResNet tuning with model introspection

Give it a ResNet, it figures out what to tune.

```bash
neuropt run examples/train_resnet.py --backend claude
```

Source: [`examples/train_resnet.py`](https://github.com/loevlie/neuropt/blob/main/examples/train_resnet.py)

Automatically discovers 9 ReLU layers and 20 BatchNorm layers, generates a search space of 5 parameters.

## Overnight results

We ran `train_fashion.py` for 8 hours on an M1 MacBook with the Claude backend (Haiku). The LLM:

- Started from completely random architectures
- Tried SiLU first, then settled on GELU after seeing faster early-epoch convergence
- Progressively reduced dropout from 0.15 → 0.08 → 0.01 → 0.0 as it noticed the train-val gap was already small
- Locked in on AdamW with lr ~8.8e-4 after seeing higher LRs cause oscillation
- All top 10 final configs converged on the same pattern: 7 blocks, ~26 base channels, GELU, residual, BatchNorm, no dropout, AdamW

The key insight: the LLM made the same decisions a human researcher would, but it ran 1,239 experiments to get there instead of a handful.
