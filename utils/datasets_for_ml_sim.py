import numpy

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from consts import *
from transformers import BertTokenizer
import os
from collections import defaultdict
from utils.functions import learn_sigmoid_weighting_by_reaction_time, get_model_name, move_to
import Simulation.strategies_code as bot_strategies
import Simulation.dm_strategies as user_strategies
import random
import utils.basic_nature_options
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import trange
import pandas as pd
from consts import *
from utils import personas


class OfflineDataSet(Dataset):
    def __init__(self, user_groups, config, weight_type, strategies=None, users=None):
        self.config = config
        reviews_path = DATA_GAME_REVIEWS_PATH
        x_path = DATA_CLEAN_ACTION_PATH_X
        y_path = DATA_CLEAN_ACTION_PATH_Y
        self.actions_df = None
        if "X" in user_groups:  # Offline Human - training groups (E_A)
            self.actions_df = pd.read_csv(x_path)
            assert self.actions_df.user_id.max() + 1 == DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
        if "Y" in user_groups:  # Offline Human - testing groups (E_B)
            Y_dataset = pd.read_csv(y_path)
            assert Y_dataset.user_id.max() + 1 == DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS
            Y_dataset.user_id += DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
            if self.actions_df is None:
                self.actions_df = Y_dataset
            else:
                self.actions_df = pd.concat([self.actions_df, Y_dataset])
        if "L" in user_groups:  # LLM Simulated users
            self.actions_df = pd.read_csv(config["OFFLINE_SIM_DATA_PATH"])
            if self.config.personas_group_number > -1:
                g = personas.get_personas_in_group(self.config.personas_group_number)
                print("In this run, we using data from personas", g)
                self.actions_df = self.actions_df[self.actions_df["persona"].isin(g)]

        if strategies is not None:
            self.actions_df = self.actions_df[self.actions_df["strategy_id"].isin(strategies)]
            strategies_in_data = self.actions_df["strategy_id"].drop_duplicates().tolist()
            for strategy in strategies:
                assert strategy in strategies_in_data, f"You have no games against strategy #{strategy} " \
                                                       f"in the entire dataset!"

        if users is not None:
            self.actions_df = self.actions_df[self.actions_df["user_id"].isin(users.tolist())]
            assert self.actions_df["user_id"].nunique() == len(users.tolist()), "some of the users that chosen to used" \
                                                                                "are not exists in the dataset!"

        if "persona" in self.actions_df.columns:
            print("user per persona:",
                  self.actions_df[["persona", "user_id"]].drop_duplicates().groupby("persona").count())

        grouped_counts = self.actions_df.groupby(["user_id", "strategy_id"]).size().reset_index(name="N")
        self.actions_df = self.actions_df.merge(grouped_counts, on=["user_id", "strategy_id"], how="left")
        number_of_groups = len(grouped_counts)
        total_samples = len(self.actions_df)
        self.actions_df["weight"] = 1
        if weight_type == "sender_receiver" or weight_type == "both":
            self.actions_df["weight"] *= total_samples / (self.actions_df["N"] * number_of_groups)
        if weight_type == "didGo" or weight_type == "both":
            p = self.actions_df["didGo"].mean()
            p_weight = (1 - p) / p
            q_weight = p / (1 - p)
            self.actions_df["weight"] *= np.where(self.actions_df["didGo"].to_numpy(), p_weight, q_weight)

        self.actions_df = self.actions_df.drop("N", axis=1)

        self.actions_df = self.actions_df.groupby(["user_id", "gameId"])

        self.idx_to_group = list(self.actions_df.indices.keys())
        self.group_to_idx = {g: i for i, g in enumerate(self.idx_to_group)}
        self.n_groups_by_user_id = defaultdict(list)
        for u, i in sorted(self.actions_df.indices.keys()):
            self.n_groups_by_user_id[u].append(i)

        self.review_reduced = pd.read_csv(config['FEATURES_PATH'], index_col=0).T.astype(int).to_dict(orient='list')
        self.review_reduced = {int(rid): torch.Tensor(vec) for rid, vec in self.review_reduced.items()}
        self.review_reduced[-1] = torch.zeros(config["REVIEW_DIM"])

        self.reviews = {}
        for h in range(1, N_HOTELS + 1):
            hotel_df = pd.read_csv(os.path.join(reviews_path, f"{h}.csv"),
                                   header=None)
            for review in hotel_df.iterrows():
                self.reviews[review[1][0]] = {"positive": review[1][2],
                                              "negative": review[1][3],
                                              "score": review[1][4]}
            self.reviews[-1] = {"positive": "",
                                "negative": "",
                                "score": 8}

    def __len__(self):
        return len(self.idx_to_group)

    def __getitem__(self, item):
        if isinstance(item, int):
            group = self.idx_to_group[item]
        else:
            group = item
        game = self.actions_df.get_group(group).reset_index()
        n_rounds = len(game)

        game["is_sample"] = np.ones(n_rounds).astype(bool)
        if n_rounds < DATA_ROUNDS_PER_GAME:
            game = pd.concat([game] + [DATA_BLANK_ROW_DF(game["strategy_id"][0])] * (DATA_ROUNDS_PER_GAME - n_rounds),
                             ignore_index=True)

        action_taken = game["didGo"].to_numpy().astype(np.int64)

        reviewId = game["reviewId"]
        round_num = np.full(10, -1)
        round_num[:n_rounds] = np.arange(n_rounds)

        sample = {"action_taken": action_taken,
                  "review_vector": reviewId.apply(lambda r: self.review_reduced[r]).tolist()}

        x = []
        y = []
        for review_vec, action in zip(sample["review_vector"], sample["action_taken"]):
            x += [review_vec]
            y += [action]

        return x, y