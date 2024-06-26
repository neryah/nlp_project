import wandb
import subprocess
from consts import *

wandb.login()


def main():
    wandb.init(project="final_full_run")
    subprocess.run(["python", "StrategyTransfer.py", wandb.config['func_name'], f'--seed={wandb.config["seeds"]}'])


sweep_configuration_1 = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "AUC.test.max"},
    "parameters": {
        "seeds": {"values": [7, 8, 9, 10, 11, 12]},
        "func_name": {"values": BEST_8_MODELS},
    },
}

# 3: Start the sweep
sweep_id_1 = wandb.sweep(sweep=sweep_configuration_1, project="final_full_run")
wandb.agent(sweep_id_1, function=main)

