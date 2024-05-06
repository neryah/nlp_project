from utils.datasets import OfflineDataSet, OnlineSimulationDataSet, ConcatDatasets
from utils.samplers import NewUserBatchSampler, SimulationSampler
from torch.utils.data import DataLoader
from torch import nn
from itertools import chain
from utils import *
import pickle
from utils import personas
import Simulation.smart_dm as smart_dm
import Simulation.smart_dm_for_check as smart_dm_for_check
import wandb


class Environment:
    def __init__(self, model_path, config, model_function):
        self.training_mode = True
        self.hidden_dim = config["hidden_dim"]
        self.use_user_vector = config["use_user_vector"]
        self.n_layers = config["layers"]
        self.env_learning_rate = config["ENV_LEARNING_RATE"]
        self.model_path = model_path
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.currentDM = None
        self.currentGame = None
        self.config = config
        self.model_function = model_function

    def __call__(self, *args, **kwargs):
        raise NotImplemented

    def train(self, do_eval=True):
        print("Start training the environment...")
        online_sim_type = self.config["online_sim_type"]
        assert online_sim_type in ["None", "mixed", "before_epoch", "init"]
        phases = []

        if online_sim_type == "init":
            raise NotImplementedError("The 'init' simulation type is not implemented yet.")

        elif self.config["task"] == "on_policy":
            test_size = ON_POLICY_TEST_SIZE
            llm_real_test = np.arange(test_size)

        if self.config["human_train_size"] != 0:
            if self.config["ENV_HPT_mode"]:
                all_users = np.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
                train_users = np.random.choice(all_users,
                                               int(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS * 0.8), replace=False)
                test_users = np.setdiff1d(all_users, train_users)
                train_dataset = OfflineDataSet(user_groups="X", strategies=[3, 0, 2, 5], users=train_users,
                                               weight_type=self.config.loss_weight_type, config=self.config)
            else:
                train_dataset = OfflineDataSet(user_groups="X", weight_type=self.config.loss_weight_type,
                                               config=self.config)

            train_sampler = NewUserBatchSampler(train_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)
            phases += [("Train", train_dataloader)]

        if self.config["offline_simulation_size"] != 0:
            if self.config.personas_group_number == -1:
                llm_users_options = range(TOTAL_LLM_USERS)
                llm_users = np.random.choice(llm_users_options, int(self.config["offline_simulation_size"]),
                                             replace=False)
            else:
                groups = personas.get_personas_in_group(self.config.personas_group_number)
                personas_df = pd.read_csv(self.config["OFFLINE_SIM_DATA_PATH"])
                if self.config["personas_balanced"]:
                    group_size = int(self.config["offline_simulation_size"]) // len(groups)
                    llm_users = []
                    for group in groups:
                        llm_users_options = personas_df[personas_df["persona"] == group]["user_id"].unique()
                        persona_users = np.random.choice(llm_users_options, group_size, replace=False)
                        llm_users += [persona_users]
                    llm_users = np.concatenate(llm_users)
                else:
                    llm_users_options = personas_df[personas_df["persona"].isin(groups)]["user_id"].unique()
                    llm_users = np.random.choice(llm_users_options, int(self.config["offline_simulation_size"]),
                                                 replace=False)
            offline_dataset = OfflineDataSet(user_groups="L", users=llm_users, config=self.config,
                                             weight_type=self.config.loss_weight_type,
                                             strategies=self.config.strategies)
            offline_sim_sampler = NewUserBatchSampler(offline_dataset, batch_size=ENV_BATCH_SIZE, shuffle=True)
            offline_sim_dataloader = DataLoader(offline_dataset, batch_sampler=offline_sim_sampler, shuffle=False)
            phases.insert(0, ("Offline Simulation", offline_sim_dataloader))

        if do_eval:
            if self.config["ENV_HPT_mode"]:
                test_dataset = OfflineDataSet(user_groups="X", users=test_users, strategies=[19, 59],
                                              weight_type=self.config.loss_weight_type, config=self.config)
            elif self.config["task"] == "off_policy":
                test_dataset = OfflineDataSet(user_groups="Y", strategies=self.config.strategies,
                                              weight_type="sender_receiver", config=self.config)
            else:
                assert self.config["task"] == "on_policy"
                test_dataset = OfflineDataSet(user_groups="X", users=llm_real_test, weight_type="sender_receiver",
                                              config=self.config,
                                              strategies=self.config.strategies)

            test_sampler = NewUserBatchSampler(test_dataset, batch_size=ENV_BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, shuffle=False)
            phases += [("Test", test_dataloader)]

        if self.config["online_simulation_size"] > 0 and online_sim_type == "before_epoch":
            phases.insert(0, ("Online Simulation", "sim_dataloader"))

        with open(f'ml_list_{self.config["seed"]}.pkl', 'rb') as f:
            ml_list = pickle.load(f)
        smart_dm.MLTrainer.initialize_samples(ml_list)
        train_rmses = []
        test_rmses = []
        train_accuracies = []
        test_accuracies = []
        metrics = Metrics("ENV")
        try:
            train_rmse, test_rmse, train_accuracy, test_accuracy = getattr(smart_dm_for_check, self.model_function)()
            metrics.write('train_rmse', train_rmse)
            metrics.write('test_rmse', test_rmse)
            metrics.write('train_accuracy', train_accuracy)
            metrics.write('test_accuracy', test_accuracy)
            train_rmses.append(train_rmse)
            test_rmses.append(test_rmse)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        except Exception as e:
            print(f"Error in {self.model_function}: {e}")
            train_rmses.append(0)
            test_rmses.append(0)
            train_accuracies.append(0)
            test_accuracies.append(0)
        wandb.log(metrics.all)
        return train_rmses, test_rmses, train_accuracies, test_accuracies

    def init_user_vector(self):
        raise NotImplemented

    def init_game_vector(self):
        raise NotImplemented

    def get_curr_vectors(self):
        raise NotImplemented
