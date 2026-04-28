import yaml
import argparse

import cs336_basics.model


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c",
                   default="cs336_systems/config.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--name", "-n", type=str, default=None,
                   help="Experiment name. All logs go to logs/<name>/. "
                        "Folder is created if it doesn't exist.")
    
    return p.parse_args()

def benchmark():
    cs336_basics.model
    
    
if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    print(cfg)