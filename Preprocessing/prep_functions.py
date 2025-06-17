import pandas as pd
import numpy as np


def missing_info(df, id_column, sort_clmn, sort_ascending=False):
    """ 
    returns Dataframe which shows information about missing values
    for eacht column of the Dataframe

    id_column:  column which contains unique identifiers for each row
    sort_clmn:  column by which the resulting DataFrame should be sorted
    sort_ascending: whether to sort in ascending order (default is False)
    """

    missing_info = []

    for col in df.columns:
        missing_rows = df[df[col].isnull()]
        num_missing = missing_rows.shape[0]

        row_ids = missing_rows[id_column].tolist()
        percent_missing = round((num_missing/df.shape[0])*100, 2)
        dtype = df[col].dtype
        missing_info.append({
            'column': col,
            'dtype': str(dtype),
            'num_missing': num_missing,
            'percent_missing': percent_missing,
            'affected_row_ids': row_ids
        })

    missing_df = pd.DataFrame(missing_info)
    missing_df = missing_df.sort_values(by=sort_clmn, ascending=sort_ascending)
    missing_df = missing_df.reset_index(drop=True)
    return missing_df


def shared_values(df, column1, column2):
    clmn1_ids = df[df['column'] == column1]['affected_row_ids'].values[0]
    clmn2_ids = df[df['column'] == column2]['affected_row_ids'].values[0]

    shared = list(set(clmn1_ids) & set(clmn2_ids))
    return shared


def calc_distribution(fix_value, of_column, to_column, df):
    
    # print(f'Probabilities for {to_column} given {of_column}={fix_value}\n')

    # remove rows with null values
    filtered_df = df.dropna(subset=[of_column, to_column])
    filtered_df = filtered_df[filtered_df[of_column] == fix_value]
    # print(filtered_df)

    # frequency of each unique ratio
    frequency = filtered_df[to_column].value_counts()

    # calc probabilities
    tot = frequency.sum()
    probability = frequency / tot
    distribution_df = pd.DataFrame({to_column: frequency.index, 'Probability': probability.values})
    # print(distribution_df)
    return distribution_df


def distribution_imputation(row, of_column, to_column, distribution_dict):
    
    # if to_column has missing value, impute based on distribution of of_column
    if pd.isnull(row[to_column]):
        fix_value = row[of_column]
        distribution = distribution_dict.get(fix_value, None)
        if distribution is not None:
            return np.random.choice(distribution[to_column], p=distribution['Probability'])
    return row[to_column]