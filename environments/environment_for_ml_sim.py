from utils.samplers import NewUserBatchSampler
from torch.utils.data import DataLoader
from utils import *
import pickle
import Simulation.smart_dm as smart_dm
from utils.datasets_for_ml_sim import OfflineDataSet
import warnings
from Simulation import smart_dm_for_check
import os
import matplotlib.pyplot as plt


def extract_x_and_y(dataloader):
    x = []
    y = []
    for x_load, y_load in dataloader:
        x_hat = [t.tolist() for t in x_load]
        y_hat = [t.tolist() for t in y_load]
        y_all = []
        x_all = []
        for x_i in x_hat:
            x_all += x_i
        for y_i in y_hat:
            y_all += y_i
        x += [x_all[i] for i in range(len(x_all)) if y_all[i] != -100]
        y += [y_all[i] for i in range(len(y_all)) if y_all[i] != -100]

    return x, y


def get_models_scores():
    model_scores = []
    for model in ALL_CLASSIFIERS:
        print(f"Training {model}...")
        model_score = getattr(smart_dm_for_check, model)()
        model_scores.append(model_score)
    return model_scores


def save_scores(model_scores):
    file = 'classifier_scores.pkl'
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'wb') as f:
        pickle.dump((ALL_CLASSIFIERS, model_scores), f)


def save_winning_models(without=0):
    with open(f'classifier_scores.pkl', 'rb') as f:
        names, scores = pickle.load(f)
    names = [x for _, x in sorted(zip(scores, names))][without:]
    for model in names:
        print(f"Training {model}...")
        trained_model = getattr(smart_dm, model)()
        file = f'trained_models/{model}.pkl'
        if os.path.exists(file):
            os.remove(file)
        with open(file, 'wb') as f:
            pickle.dump(trained_model, f)


def plot(without=0):
    with open(f'classifier_scores.pkl', 'rb') as f:
        names, scores = pickle.load(f)

    names = [x for _, x in sorted(zip(scores, names))][without:]
    names = [CLASSIFIER_NAMES[name] for name in names]
    scores = sorted(scores)[without:]
    plt.figure(figsize=(13, 7))
    plt.bar(names, scores, color=plt.cm.viridis(np.linspace(0, 0.7, len(scores))))
    plt.title(f'Cross Validation Average Accuracy for Top {len(names)} Models', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=15)
    plt.xlabel('Model Name', fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # adding text to bars
    ax = plt.gca()
    for i in range(len(scores)):
        if scores[i] < 0.67:
            ax.text(i, 0.676, f'{round(scores[i], 3)}', ha='center', fontsize=10)
        ax.text(i, scores[i] + 0.0001, f'{round(scores[i], 3)}', ha='center', fontsize=10)
    plt.ylim(0.675, max(scores) + 0.01)
    plt.tight_layout()
    plt.savefig(f'plots/ml_sim_top_{len(names)}.png')
    plt.show()


class Environment:
    def __init__(self, config, mode='save_scores', without=0):
        self.config = config
        warnings.filterwarnings("ignore")
        if mode == 'save_models' or mode == 'save_scores':
            dataloader = self.get_dataloader()
            x, y = extract_x_and_y(dataloader)
            smart_dm.MLTrainer.initialize_samples(x, y)
        if mode == 'save_scores':
            model_scores = get_models_scores()
            save_scores(model_scores)
        if mode == 'plot':
            plot(without)
        if mode == 'save_models':
            save_winning_models(without)

    def get_dataloader(self):
        train_dataset = OfflineDataSet(user_groups="X", weight_type=self.config.loss_weight_type,
                                               config=self.config)
        train_sampler = NewUserBatchSampler(train_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
        return DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)
        # return DataLoader(test_dataset, batch_sampler=test_sampler, shuffle=False)
