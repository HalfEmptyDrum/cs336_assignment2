import argparse
from dataclasses import dataclass, asdict
from enum import StrEnum

from einops import rearrange

import torch

import cs336_basics

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


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

TRAIN_CONFIG: TrainConfig = TrainConfig(batch_size=32, context_length=256, warmup_steps=100, timing_steps=100)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", "-n", type=str, default=None,
                   help="Experiment name. All logs go to logs/<name>/.")
    p.add_argument("--size", "-s", type=ModelSize, default=ModelSize.SMALL,
                   choices=list(ModelSize),
                   help="Model size preset.")
    return p.parse_args()


def run_model(model, optimizer, model_cfg, optim_cfg, train_cfg):
    optimizer.zero_grad()
        
    x = torch.randn(train_cfg.batch_size, train_cfg.context_length, model_cfg.d_model) # (B, T, d_model)
    target = torch.randn(train_cfg.batch_size, train_cfg.context_length)
    
    print(x.shape)
    
    pred = model(x)
    pred = rearrange(pred, "B T V -> (B T) V")
    
    loss = cross_entropy(pred, target)
    loss.backward()
    optimizer.step()

def benchmark(model_size: ModelSize, experiment: str):
    model_cfg = MODEL_CONFIGS[model_size]
    optim_cfg = OPTIMIZER_CONFIG
    train_cfg = TRAIN_CONFIG
    model = BasicsTransformerLM(    # was: model = model(...)  ← bug
        vocab_size=10_000,
        context_length=256,
        rope_theta=10_000.0,
        **asdict(model_cfg),
    )
    print(f"Built {model_size} model with {sum(p.numel() for p in model.parameters()):,} params")
    
    
    optimizer = AdamW(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, betas=optim_cfg.betas)
    
    
    print("warmup...")
    for _ in range(train_cfg.warmup_steps):
        run_model(model, optimizer, model_cfg, optim_cfg, train_cfg)
    print("done with warmup.")
        
    print("timing...")
    for _ in range(train_cfg.timing_steps):
        run_model(model, optimizer, model_cfg, optim_cfg, train_cfg)
        
    print("done with timing.")

        
    
    
    


if __name__ == "__main__":
    args = parse_args()
    benchmark(args.size, experiment=args.name)