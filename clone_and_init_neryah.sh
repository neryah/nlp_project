
# cloning our repository
git clone https://github.com/neryah/nlp_project.git

# going into the repository
# shellcheck disable=SC2164
cd nlp_project

# testing the accuracy of the different classifiers via cross validation and saving the average accuracy of each
python StrategyTransferForMLSim.py --env_mode=save_scores

# plotting the results
python StrategyTransferForMLSim.py --env_mode=plot

# saving the best classifiers
python StrategyTransferForMLSim.py --env_mode=save_models

# using the best classifiers as a strategy for the game
python full_run_stage_1.py

# using the best classifiers from stage 1 as a strategy for the game on more seeds to validate the results
python full_run_stage_2.py

# using all combinations of the best classifiers from stage 2 as strategies for the game
python full_run_stage_3.py

# using the best classifiers from stage 3 as a strategy for the game on more seeds to validate the results
python full_run_stage_4.py

# getting all sweeps data
python Notebooks_neryah/get_sweeps.py

# reading the sweeps data and plotting the results
python Notebooks_neryah/full_read.py
