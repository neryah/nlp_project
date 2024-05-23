import matplotlib.pyplot as plt
import pickle
from consts import *
import ast
from typing import Tuple, List
import pandas as pd


def group_by_func_name(df: pd.DataFrame) -> pd.DataFrame:
    """Adds function name and basic nature columns and groups the DataFrame by function name."""
    df['func_name'] = df['config'].apply(lambda x: ast.literal_eval(x).get('func_name'))
    df['basic_nature'] = df['config'].apply(lambda x: ast.literal_eval(x).get('basic_nature'))
    return df.groupby('func_name')


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


def add_text_to_bars(ax, heights: List[float], sorted_heights: List[float], offset: float = 0.0001, fontsize: int = 10) -> None:
    """Adds text annotations to the bars in the plot."""
    for i, height in enumerate(heights):
        ax.text(i, height + offset, round(height, 4), ha='center', va='bottom', fontsize=fontsize)
        ax.text(i, height + offset, f'{sorted_heights.index(height) + 1}', ha='center', va='top', fontsize=fontsize, color='white')


def save_and_show_plot(filename: str) -> None:
    """Saves the current plot to a file and displays it."""
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.show()


def plot_accuracy_progress(all_func_accuracies: List[List[float]], names: List[str]) -> None:
    """Plots mean accuracy over epochs for each function."""
    setup_plot('Epoch', 'Mean Accuracy', 'Mean Accuracy vs Epoch', figsize=(7, 7))
    plt.gca().set_prop_cycle(COLOR_CYCLE)
    for func_accuracies, name in zip(all_func_accuracies, names):
        plt.plot(func_accuracies, label=name)
    plt.legend()
    save_and_show_plot('accuracy_progress.png')


def plot_max_accuracies(names: List[str], mean_accs: List[float]) -> None:
    """Plots bar chart of maximum accuracies for each function."""
    setup_plot('Model Type', 'Test Accuracy', 'Model Performance Comparison', figsize=(10, 7))
    ax = plt.gca()
    ax.bar(names, mean_accs, color=plt.cm.viridis(np.linspace(0, 0.7, len(mean_accs))))
    sorted_mean_accs = sorted(mean_accs)
    add_text_to_bars(ax, mean_accs, sorted_mean_accs, fontsize=10)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.ylim(min(mean_accs) - 0.0001, max(mean_accs) + 0.0001)
    save_and_show_plot('final_performance.png')


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
    indices = [filtered_names.index(name) for name in names]
    return [scores[i] for i in indices]


def plot_comparison(filtered_names: List[str], filtered_accs: List[float], scores_orig: List[float]) -> None:
    """Plots a comparison between two sets of scores."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(filtered_names, filtered_accs, label='Filtered Scores')
    ax.bar(filtered_names, scores_orig, label='Original Scores')

    sorted_filtered = sorted(filtered_accs, reverse=True)
    sorted_orig = sorted(scores_orig, reverse=True)
    add_text_to_bars(ax, filtered_accs, sorted_filtered, fontsize=10)
    add_text_to_bars(ax, scores_orig, sorted_orig, fontsize=10)

    ax.set_ylabel('Scores')
    ax.set_title('Filtered vs Original Scores')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(min(scores_orig) - 0.01, max(filtered_accs) + 0.01)
    save_and_show_plot('final_performance_orig.png')


def plot_ranking_diff(filtered_names: List[str], filtered_accs: List[float], scores_orig: List[float]) -> None:
    """Plots the difference in ranking between two sets of scores."""
    sorted_filtered = sorted(filtered_accs, reverse=True)
    sorted_orig = sorted(scores_orig, reverse=True)
    ranking_diff = [sorted_filtered.index(acc) - sorted_orig.index(orig) for acc, orig in
                    zip(filtered_accs, scores_orig)]

    setup_plot('Model Type', 'Difference', 'Difference in Ranking', figsize=(10, 7))
    ax = plt.gca()
    ax.bar(filtered_names, ranking_diff, color=plt.cm.viridis(np.linspace(0, 0.7, len(ranking_diff))))

    for i, diff in enumerate(ranking_diff):
        ax.text(i, diff + 0.0001, diff, ha='center', va='top', fontsize=10)

    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.ylim(-10, 10)
    save_and_show_plot('ranking_difference.png')


def main():
    df = pd.read_csv('all_runs_data.csv')
    grouped = group_by_func_name(df)
    all_func_accuracies, names, mean_accs = calculate_epoch_accuracies(grouped)

    plot_accuracy_progress(all_func_accuracies, names)
    plot_max_accuracies(names, mean_accs)

    names_orig, scores_orig = load_scores()
    filtered_names, filtered_accs = remove_none(names, mean_accs)
    scores_orig = match_scores(scores_orig, names_orig, filtered_names)

    plot_comparison(filtered_names, filtered_accs, scores_orig)
    plot_ranking_diff(filtered_names, filtered_accs, scores_orig)


if __name__ == "__main__":
    main()
