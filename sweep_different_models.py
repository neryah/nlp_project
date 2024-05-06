import wandb
import subprocess

wandb.login()


def main():
    wandb.init(project="Final_Model_Choice_2")
    subprocess.run(["python", "StrategyTransferForCheck.py", wandb.config['func_name'], f'--seed={wandb.config["seeds"]}'])


# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "ENV_None_test_accuracy"},
    "parameters": {
        "seeds": {"values": [420, 51, 942, 14, 948, 57, 663, 178, 543, 329]},
        "func_name": {"values": ["LinearRegressionTrainer", "LogisticRegressionTrainer", "RandomForestRegressorTrainer",
                                 "RandomForestClassifierTrainer", "SVRTrainer", "SVCTrainer", "MLPRegressorTrainer",
                                 "MLPClassifierTrainer", "GradientBoostingRegressorTrainer",
                                 "GradientBoostingClassifierTrainer", "AdaBoostRegressorTrainer",
                                 "AdaBoostClassifierTrainer", "BaggingRegressorTrainer", "BaggingClassifierTrainer"]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Final_Model_Choice_2")

wandb.agent(sweep_id, function=main)
