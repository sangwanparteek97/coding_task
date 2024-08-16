# External Imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pandas as pd
import random


def load_files(path):
    available_files = os.listdir(path)
    file_contents = []
    for a_f in available_files:
        if not a_f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(path, a_f), delimiter=',')
        file_contents.append(df)
    return file_contents


def filter_files(files, iteration=0, last_scores=0, last_selections=None):
    """
    This function should select a subset of files from the list of all files.

    :param files: List of pandas DataFrames, each DataFrame represents a file
    :param iteration: The current iteration of the selection process
    :param last_scores: List of scores from the last iterations
    :param last_selections: List of selected files from the last iterations
    :return: List of selected files
    """

    # TODO: IMPLEMENT some logic here to select a subset of all files.
    #  The size of the subset does not need to be fixed, it can be anything and can change
    # Hint: the score of the `run_classification'-function gets higher the more homogeneous the contents of the selected files are
    # return random.sample(files, random.randint(1, len(files) - 1))

    assert isinstance(files, list), "files should be a list"
    assert all([isinstance(df, pd.DataFrame) for df in files]), "files should be a list of pandas DataFrames"
    assert isinstance(iteration, int), "iteration should be an integer"

    # Calculate variances for all files
    file_indices = list(range(len(files)))
    file_vars = [np.sum([df[f'col_{i}'].var() for i in range(1, 10)]) for df in files]
    file_selection_df = pd.DataFrame(
        {"file_idx": file_indices, "var": file_vars},
    )

    # Sort the variance im ascending order because a low variance indicates a high score
    file_selection_df = file_selection_df.sort_values(by='var', ascending=True)

    if iteration == 1:
        num_files_to_select = len(last_selections[iteration - 1]) + 1

    elif iteration > 1:
        if last_scores[-1] > last_scores[-2]:
            # Score improved, continue with a similar or slightly larger selection
            num_files_to_select = min(len(files), len(last_selections[-1]) + 10)  # This can be configured
        else:
            # Score did not improve, try a different subset size
            num_files_to_select = max(1, len(last_selections[-1]) - 1)  # This can be configured
    else:
        # We select atleast 50% of files for iteration 0 or a random number of files beyond 50%
        num_files_to_select = max(len(files) // 2, random.randint(1, file_selection_df.shape[0]))
        print(f"Selecting {num_files_to_select} files")

    selected_files = file_selection_df['file_idx'].head(num_files_to_select).tolist()
    return [files[i] for i in selected_files]


def filter_files_using_clustering(files, iteration=0, last_scores=0, last_selections=None):
    """
    This function should select a subset of files from the list of all files.
    :param files: List of pandas DataFrames, each DataFrame represents a file
    :param iteration: The current iteration of the selection process
    :param last_scores: List of scores from the last iterations
    :param last_selections: List of selected files from the last iterations
    :return: List of selected files
    """

    assert isinstance(files, list), "files should be a list"
    assert all([isinstance(df, pd.DataFrame) for df in files]), "files should be a list of pandas DataFrames"
    assert isinstance(iteration, int), "iteration should be an integer"

    # Calculate variances for all files
    file_indices = list(range(len(files)))
    file_vars_sum = [np.sum([df[f'col_{i}'].var() for i in range(1, 10)]) for df in files]
    file_selection_df = pd.DataFrame({"file_idx": file_indices, "var_sum": file_vars_sum})

    # Scale the summed up variances and cluster them (number of clusters are to be studied and optimized)
    file_selection_df['var_sum'] = StandardScaler().fit_transform(file_selection_df[['var_sum']])
    file_selection_df['cluster'] = KMeans(n_clusters=8, random_state=42).fit_predict(file_selection_df[['var_sum']])

    num_files_to_select = file_selection_df[file_selection_df['cluster'] <= iteration].shape[0]
    selected_files = file_selection_df['file_idx'].head(num_files_to_select).tolist()
    return [files[i] for i in selected_files]


def run_classification(file_selection):
    var_sums = [1 / np.sum([df[f'col_{i}'].var() for i in range(1, 10)]) for df in file_selection]
    score = 2000 * len(file_selection) * np.mean(var_sums)
    return score, 500


def main():
    # Do not modify this function unless you need to pass some additional data to the `filter_files` function that is not yet passed
    # The summation of cost and score should stay the same
    # modification for error handling / checking data is of course allowed here as well
    files = load_files("./data")
    total_cost = 0
    top_score = 0

    last_scores = []
    last_selections = []
    itr = 0
    while top_score < 88 and total_cost < 100000:
        # TODO: Please uncomment the approach to test the said file selection method
        # Approach 1: We select files based on how scores are improving
        # file_selection = filter_files(
        #     files,
        #     iteration=itr,
        #     last_scores=last_scores,
        #     last_selections=last_selections
        # )

        # Approach 2: We select files based on clustering
        file_selection = filter_files_using_clustering(
            files,
            iteration=itr,
            last_scores=last_scores,
            last_selections=last_selections
        )

        score, cost = run_classification(file_selection)
        top_score = max(top_score, score)
        total_cost += cost
        last_selections.append(file_selection)
        last_scores.append(score)
        itr += 1
        print(f"You reached a score of {top_score} at a cost of {total_cost}")
    print(f"You reached a score of {top_score} at a cost of {total_cost}")
    return


if __name__ == '__main__':
    main()
