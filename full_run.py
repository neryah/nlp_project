import wandb
import subprocess
from consts import *

wandb.login()


def main():
    wandb.init(project="final_run")
    subprocess.run(["python", "StrategyTransfer.py", wandb.config['func_name'], f'--seed={wandb.config["seeds"]}'])


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "AUC.test.max"},
    "parameters": {
        "seeds": {"values": [1, 2, 3, 4, 5, 6]},
        "func_name": {"values": WINNING_MODELS},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="final_run")
wandb.agent(sweep_id, function=main)
