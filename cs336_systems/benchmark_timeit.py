import argparse
from dataclasses import dataclass, asdict
from enum import StrEnum

from einops import rearrange

import torch

import timeit

import cs336_basics

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

cs336_basics.model.scaled_dot_product_attention = cs336_basics.model.annotated_scaled_dot_product_attention
import torch.cuda.nvtx as nvtx


import tqdm


class ModelSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    TENB = "10B"


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    
    
@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    context_length: int
    warmup_steps: int
    timing_steps: int
    


MODEL_CONFIGS: dict[ModelSize, ModelConfig] = {
    ModelSize.SMALL:  ModelConfig(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    ModelSize.MEDIUM: ModelConfig(1024, 4096,  24, 16),
    ModelSize.LARGE:  ModelConfig(1280, 5120,  36, 20),
    ModelSize.XL:     ModelConfig(2560, 10240, 32, 32),
    ModelSize.TENB:   ModelConfig(4608, 12288, 50, 36),
}

OPTIMIZER_CONFIG: OptimizerConfig = OptimizerConfig(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

TRAIN_CONFIG: TrainConfig = TrainConfig(batch_size=4, context_length=256, warmup_steps=5, timing_steps=100)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", "-n", type=str, default=None,
                   help="Experiment name. All logs go to logs/<name>/.")
    p.add_argument("--size", "-s", type=ModelSize, default=ModelSize.SMALL,
                   choices=list(ModelSize),
                   help="Model size preset.")
    return p.parse_args()


def benchmark(model_size: ModelSize, experiment: str):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    vocab_size=10000
    model_cfg = MODEL_CONFIGS[model_size]
    optim_cfg = OPTIMIZER_CONFIG
    train_cfg = TRAIN_CONFIG
    
    model = BasicsTransformerLM(    # was: model = model(...)  ← bug
        vocab_size=vocab_size,
        context_length=256,
        rope_theta=10000.0,
        **asdict(model_cfg)
    ).to(device)
    print(f"Built {model_size} model with {sum(p.numel() for p in model.parameters()):,} params")
    
    
    optimizer = AdamW(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, betas=optim_cfg.betas)

    time_taken = {"forward": [], "backward": [], "optimizer": []}

    for i in tqdm.tqdm(range(train_cfg.warmup_steps + train_cfg.timing_steps)):
            
        optimizer.zero_grad() 

        x = torch.randint(0, vocab_size, (train_cfg.batch_size, train_cfg.context_length), device=device)
        target = torch.randint(0, vocab_size, (train_cfg.batch_size, train_cfg.context_length), device=device)

        torch.cuda.synchronize()
        start = timeit.default_timer()
        pred = model(x)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        time_taken["forward"].append(end - start)
        
        pred = rearrange(pred, "B T V -> (B T) V")            # (B*T, V)
        target = rearrange(target, "B T -> (B T)")            # (B
        
        loss = cross_entropy(pred, target)
        
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        
        time_taken["backward"].append(end - start)

        torch.cuda.synchronize()
        start = timeit.default_timer()
        optimizer.step()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        
        time_taken["optimizer"].append(end - start)
    
    
    
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Phase & Mean (ms) & Std (ms) & Min (ms) & Median (ms) \\")
    print(r"\midrule")

    for part in time_taken.keys():
        times = torch.tensor(time_taken[part][train_cfg.warmup_steps:]) * 1000
        print(f"{part.capitalize()} & "
            f"{times.mean():.3f} & "
            f"{times.std():.3f} & "
            f"{times.min():.3f} & "
            f"{times.median():.3f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
        

  

    

        
    
    
    


if __name__ == "__main__":
    args = parse_args()
    benchmark(args.size, experiment=args.name)