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
        "seeds": {"values": [5, 6]},
        "func_name": {"values": COMB_2_MACHINE},
    },
}

# 3: Start the sweep
sweep_id_1 = wandb.sweep(sweep=sweep_configuration_1, project="final_full_run")
wandb.agent(sweep_id_1, function=main)

