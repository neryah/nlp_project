import matplotlib.pyplot as plt
import pickle
from consts import *
import ast
from typing import Tuple, List
import pandas as pd
import warnings


def extend_df(df: pd.DataFrame) -> None:
    """Adds function name and basic nature columns to the DataFrame."""
    df['func_name'] = df['config'].apply(lambda x: ast.literal_eval(x).get('func_name'))
    df['basic_nature'] = df['config'].apply(lambda x: ast.literal_eval(x).get('basic_nature'))
    df['seed'] = df['config'].apply(lambda x: ast.literal_eval(x).get('seed'))


def get_stages_func_names(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Returns lists of function names for each stage of the experiment."""
    all_func_names = df['func_name'].unique()
    func_names_stage_1 = [name for name in all_func_names if ',' not in name]
    func_names_stage_2 = [name for name in all_func_names if ','
                          not in name and df[df['func_name'] == name]['seed'].max() == 12]
    func_names_stage_3 = [name for name in all_func_names if ',' in name] + ['None']
    func_names_stage_4 = [name for name in all_func_names if ','
                          in name and df[df['func_name'] == name]['seed'].max() == 12] + ['None']
    return func_names_stage_1, func_names_stage_2, func_names_stage_3, func_names_stage_4


def group_by_func_name(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adds function name and basic nature columns and groups the DataFrame by function name."""
    func_names_stage_1, func_names_stage_2, func_names_stage_3, func_names_stage_4 = get_stages_func_names(df)
    df_stage_1 = df[df['seed'] < 7][df['func_name'].isin(func_names_stage_1)]
    df_stage_2 = df[df['func_name'].isin(func_names_stage_2)]
    df_stage_3 = df[df['seed'] < 7][df['func_name'].isin(func_names_stage_3)]
    df_stage_4 = df[df['func_name'].isin(func_names_stage_4)]
    return df_stage_1.groupby('func_name'), df_stage_2.groupby('func_name'), df_stage_3.groupby('func_name'), \
        df_stage_4.groupby('func_name')


def calculate_epoch_accuracies(grouped: pd.DataFrame) -> Tuple[List[List[float]], List[str], List[float]]:
    """Calculates mean accuracy for each function over all epochs."""
    all_func_accuracies = []
    names = []
    mean_accs = []

    for name, group in grouped:
        func_accuracies = []
        names.append(name)
        for i in range(25):
            epoch_accuracies = group['metrics'].apply(
                lambda x: ast.literal_eval(x).get(f'ENV_Test_accuracy_per_mean_user_and_bot_epoch{i}'))
            func_accuracies.append(epoch_accuracies.mean())
        all_func_accuracies.append(func_accuracies)
        mean_accs.append(max(func_accuracies))

    sorted_indices = sorted(range(len(mean_accs)), key=lambda k: mean_accs[k])
    names = [names[i] for i in sorted_indices]
    mean_accs = [mean_accs[i] for i in sorted_indices]

    return all_func_accuracies, names, mean_accs


def setup_plot(xlabel: str, ylabel: str, title: str, figsize: Tuple[int, int] = (10, 7)) -> None:
    """Sets up a matplotlib plot with the given labels and title."""
    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def add_text_to_bars(ax, heights: List[float], sorted_heights: List[float], offset: float = 0.0001,
                     fontsize: int = 10) -> None:
    """Adds text annotations to the bars in the plot."""
    for i, height in enumerate(heights):
        ax.text(i, height + offset, round(height, 4), ha='center', va='bottom', fontsize=fontsize)
        ax.text(i, height + offset, f'{sorted_heights.index(height) + 1}', ha='center', va='top', fontsize=fontsize,
                color='white')


def save_and_show_plot(filename: str) -> None:
    """Saves the current plot to a file and displays it."""
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.show()


def plot_accuracy_progress(all_func_accuracies: List[List[float]], names: List[str], stage: int) -> None:
    """Plots mean accuracy over epochs for each function."""
    setup_plot('Epoch', 'Mean Accuracy', f'Mean Accuracy vs Epoch Stage {stage + 1}', figsize=(7, 7))
    plt.gca().set_prop_cycle(COLOR_CYCLE)
    for func_accuracies, name in zip(all_func_accuracies, names):
        plt.plot(func_accuracies, label=get_acc_horizontal(name))
    plt.legend()
    save_and_show_plot(f'accuracy_progress_stage_{stage + 1}.png')


def get_acc(name):
    n_list = name.split(',')
    names = [CLASSIFIER_NAMES[n] for n in n_list]
    name = ',\n'.join(names)
    return name


def get_acc_horizontal(name):
    n_list = name.split(',')
    names = [CLASSIFIER_NAMES[n] for n in n_list]
    name = ', '.join(names)
    return name


def plot_max_accuracies(names: List[str], mean_accs: List[float], stage: int) -> None:
    """Plots bar chart of maximum accuracies for each function."""
    setup_plot('Model Type', 'Test Accuracy', f'Model Performance Comparison Stage {stage + 1}', figsize=(10, 7))
    names1 = [get_acc(name) for name in names]
    ax = plt.gca()
    ax.bar(names1, mean_accs, color=plt.cm.viridis(np.linspace(0, 0.7, len(mean_accs))))
    sorted_mean_accs = sorted(mean_accs)
    add_text_to_bars(ax, mean_accs, sorted_mean_accs, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(min(mean_accs) - 0.0001, max(mean_accs) + 0.0001)
    save_and_show_plot(f'final_performance_stage_{stage + 1}.png')


def load_scores() -> Tuple[List[str], List[float]]:
    """Loads and sorts original classifier scores from a pickle file."""
    with open('classifier_scores.pkl', 'rb') as f:
        names_orig, scores_orig = pickle.load(f)
    scores_orig, names_orig = zip(*sorted(zip(scores_orig, names_orig))[8:])
    return list(names_orig), list(scores_orig)


def remove_none(names: List[str], mean_accs: List[float]) -> Tuple[List[str], List[float]]:
    """Removes 'None' from names and corresponding accuracies."""
    filtered_names = [name for name in names if name != 'None']
    filtered_accs = [mean_accs[i] for i in range(len(mean_accs)) if names[i] != 'None']
    return filtered_names, filtered_accs


def match_scores(scores: List[float], names: List[str], filtered_names: List[str]) -> List[float]:
    """Matches scores to the filtered names list."""
    indices = [filtered_names.index(name) for name in names if name in filtered_names]
    return [scores[indices.index(i)] for i in range(len(filtered_names))]


def plot_comparison(filtered_names: List[str], filtered_accs: List[float], scores_orig: List[float], stage: int) \
        -> None:
    """Plots a comparison between two sets of scores."""
    fig, ax = plt.subplots(figsize=(10, 7))
    filtered_names = [get_acc(name) for name in filtered_names]
    ax.bar(filtered_names, filtered_accs, label='Filtered Scores')
    ax.bar(filtered_names, scores_orig, label='Original Scores')

    sorted_filtered = sorted(filtered_accs, reverse=True)
    sorted_orig = sorted(scores_orig, reverse=True)
    add_text_to_bars(ax, filtered_accs, sorted_filtered, fontsize=10)
    add_text_to_bars(ax, scores_orig, sorted_orig, fontsize=10)

    ax.set_ylabel('Scores')
    ax.set_title(f'Filtered vs Original Scores  Stage {stage + 1}')
    plt.xticks(fontsize=10)
    plt.ylim(min(scores_orig) - 0.01, max(filtered_accs) + 0.01)
    save_and_show_plot(f'final_performance_orig_stage_{stage + 1}.png')


def plot_ranking_diff(filtered_names: List[str], filtered_accs: List[float], scores_orig: List[float], stage: int) \
        -> None:
    """Plots the difference in ranking between two sets of scores."""
    sorted_filtered = sorted(filtered_accs, reverse=True)
    sorted_orig = sorted(scores_orig, reverse=True)
    ranking_diff = [sorted_filtered.index(acc) - sorted_orig.index(orig) for acc, orig in
                    zip(filtered_accs, scores_orig)]

    filtered_names = [get_acc(name) for name in filtered_names]

    setup_plot('Model Type', 'Difference', f'Difference in Ranking  Stage {stage + 1}', figsize=(10, 7))
    ax = plt.gca()
    ax.bar(filtered_names, ranking_diff, color=plt.cm.viridis(np.linspace(0, 0.7, len(ranking_diff))))

    for i, diff in enumerate(ranking_diff):
        ax.text(i, diff + 0.0001, diff, ha='center', va='top', fontsize=10)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-10, 10)
    save_and_show_plot(f'ranking_difference_stage_{stage + 1}.png')


def main():
    warnings.filterwarnings("ignore")
    df = pd.read_csv('all_runs_data.csv')
    extend_df(df)
    grouped_stages = group_by_func_name(df)
    for i, grouped in enumerate(grouped_stages):
        all_func_accuracies, names, mean_accs = calculate_epoch_accuracies(grouped)
        plot_accuracy_progress(all_func_accuracies, names, i)
        plot_max_accuracies(names, mean_accs, i)

        if i < 2:
            names_orig, scores_orig = load_scores()
            filtered_names, filtered_accs = remove_none(names, mean_accs)
            scores_orig = match_scores(scores_orig, names_orig, filtered_names)

            plot_comparison(filtered_names, filtered_accs, scores_orig, i)
            plot_ranking_diff(filtered_names, filtered_accs, scores_orig, i)


if __name__ == "__main__":
    main()
