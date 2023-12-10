from typing import Tuple

import torch
from sklearn.model_selection import train_test_split

from src.datasets.preprocess.adult import get_adult_dataset

dataset_name_to_preprocess_func = {
    'adult': get_adult_dataset,
}


class TabularMetadata:
    def __init__(self, metadata_df):
        self.metadata_df = metadata_df
    # TODO add methods according to needs

    @property
    def n_classes(self):
        return int(self.metadata_df[self.metadata_df.type == 'label'].n_values.item())

    @property
    def n_features(self):
        return int(len(self.metadata_df) - 1)  # -1 for the label

    @property
    def summary(self):
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            # 'feature_names': self.feature_names, # TODO
            # 'label_name': self.label_name,  # TODO
            'train_test_split_ratio': '0.87',
            'train_test_split_random_seed': '42',
        }


def load_data(
        dataset_name: str,
        data_file_path: str,
        metadata_file_path: str,
        encoding_method: str = None
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, TabularMetadata]:

    # preprocess
    x_df, y_df, metadata_df = get_adult_dataset('data/adult/adult.data',
                                                metadata_file_path='data/adult/adult.metadata.csv')

    # split to train and test:
    df_train, df_test, _, _ = train_test_split(x_df, y_df,
                                               train_size=0.87, random_state=42)  # TODO split should be configurable

    # setup dataset object
    trainset = testset = torch.utils.data.TensorDataset(torch.tensor(x_df.values, dtype=torch.float32),
                                                        torch.tensor(y_df.values, dtype=torch.long))

    # create features metadata object, to be used in the attack
    features_metadata = TabularMetadata(metadata_df)

    return trainset, testset, features_metadata
