import pandas as pd
from consts import *
import wandb

api = wandb.Api()


def fetch_run_data(runs, sweep_id):
    """Fetches data for each run in the given sweep."""
    run_data = []
    for run in runs:
        data = {
            'run_id': run.id,
            'config': run.config,
            'metrics': run.summary,
            'sweep_id': sweep_id
        }
        run_data.append(data)
    return run_data


def gather_all_runs_data():
    """Gathers data for all runs across all sweeps."""
    all_runs_data = []
    for sweep_id in SWEEP_IDS:
        sweep = api.sweep(f'{TEAM_NAME}/{PROJECT_NAME}/{sweep_id}')
        all_runs_data.extend(fetch_run_data(sweep.runs, sweep_id))
    df = pd.DataFrame(all_runs_data)
    df.to_csv('all_runs_data.csv', index=False)


if __name__ == "__main__":
    gather_all_runs_data()
