import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Any, Optional, Dict

from src.datasets.preprocess.utils import add_one_hot_encoding, add_mapping_encoding

"""
# metadata files include information essential for preprocessing, structure constraints and for inference time.
# The metadata corresponding dataframe is also updated after the preprocessing.
"""


def get_adult_dataset(data_file_path: str,
                      metadata_file_path: str,
                      encoding_method: str = None) -> Tuple[pd.DataFrame, List[str], str]:
    """
    :param data_file_path: path to raw CSV dataset of the bank.
    :param metadata_file_path: path to raw CSV dataset of the bank.
    :param encoding_method: whether to perform one-hot-encoding on categorical features in the dataset
    :returns: DataFrame after pre-processing (including encoding categorical features),
                and a list of the features-meta-data
    """

    assert encoding_method in ('one_hot_encoding', None)

    # 0. Read data
    raw_data_header = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status',
                       'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                       'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(data_file_path, names=raw_data_header)
    metadata_df = pd.read_csv(metadata_file_path)

    # 1. Set order and filtering of columns from metadata (from metadata file)
    df = df[metadata_df['name']].copy()

    # 2. Perform basic transformations (missing data, mapping columns, etc)
    for cat_feature in metadata_df[metadata_df.type == 'categorical'].name:
        # Missing: Missing val --> most common val
        missing_val_string = ' ?'
        most_common_val = df[cat_feature].mode()[0]
        df[cat_feature].replace(missing_val_string, most_common_val, inplace=True)

    label_col = metadata_df[metadata_df.type == 'label'].name.item()
    df[label_col] = (df[label_col] == ' >50K')  # we predict who _has_ high income

    # 3. Categorical encoding:
    if encoding_method == 'one_hot_encoding':
        # count number of unique values in each categorical feature
        df, metadata_df = add_one_hot_encoding(df, metadata_df)
    elif encoding_method is None:
        # Default encoding follows a simple mapping of categories to integers
        df, metadata_df = add_mapping_encoding(df, metadata_df)

    # 4. Update n_unique column in metadata (with for loop)
    for idx, row in metadata_df.iterrows():
        metadata_df.at[idx, 'n_values'] = df[row['name']].nunique()

    # 5. split to input and labels
    x_df, y_df = df.drop(columns=[label_col]), df[label_col]

    return x_df, y_df, metadata_df


def get_adult_from_dict_sample(
        samples_dict: List[Dict[str, Any]],
        input_file_path: str,
        metadata_file_path: str,
        encoding_method: str = None
):
    pass  # TODO: implement


if __name__ == '__main__':
    get_adult_df("data/adult/adult.data", "data/adult/adult.metadata.csv",
                 'one_hot_encoding')
