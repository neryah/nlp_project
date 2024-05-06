import wandb
import subprocess

wandb.login()


def main():
    wandb.init(project="try_seed_saves")
    subprocess.run(["python", "StrategyTransferForSeedSave.py", f'--seed={wandb.config["seeds"]}'])


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "ENV_None_test_accuracy"},
    "parameters": {
        "seeds": {"values": [420, 51, 942, 14, 948, 57, 663, 178, 543, 329]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="try_seed_saves")

wandb.agent(sweep_id, function=main)
