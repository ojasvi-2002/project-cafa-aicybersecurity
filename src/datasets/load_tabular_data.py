from typing import List, Dict, Type
import ast

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.datasets.preprocess.adult import get_adult_dataset
from src.datasets.preprocess.bank import get_bank_dataset
from src.datasets.preprocess.phishing import get_phishing_dataset

dataset_name_to_preprocess_func = {
    'adult': get_adult_dataset,
    'bank': get_bank_dataset,
    'phishing': get_phishing_dataset,
}


class TabularDataset:
    def __init__(
            self,
            dataset_name: str,
            data_file_path: str,
            metadata_file_path: str,
            encoding_method: str = 'one_hot_encoding',
            random_seed: int = 42,
            train_proportion: float = 0.87,
    ):
        self.dataparameters = {
            'dataset_name': dataset_name,
            'data_file_path': data_file_path,
            'metadata_file_path': metadata_file_path,
            'encoding_method': encoding_method,
            'random_seed': random_seed,
            'train_proportion': train_proportion,
        }
        # preprocess # TODO generalize
        self.x_df, self.y_df, self.metadata_df = dataset_name_to_preprocess_func[dataset_name](
            data_file_path=data_file_path,
            metadata_file_path=metadata_file_path,
            encoding_method=encoding_method
        )

        # Split to train and test:
        X_train, X_test, y_train, y_test = train_test_split(self.x_df, self.y_df,
                                                            train_size=train_proportion,
                                                            random_state=random_seed)  # TODO split should be configurable
        # Validate the processed input data
        self._validate_input()

        # Save numpy arrays
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train.values.astype(np.float32), X_test.values.astype(np.float32),
            y_train.values.astype(np.float32), y_test.values.astype(np.float32))

        # Create features metadata object, to be used in the attack
        self.feature_names = self.metadata_df[self.metadata_df.type != 'label'].feature_name.values
        self.label_name = self.metadata_df[self.metadata_df.type == 'label'].feature_name.item()
        self.cat_encoding_method = encoding_method

    @property
    def trainset(self):
        """
        :return:
        """
        trainset = torch.utils.data.TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long)
        )
        return trainset

    @property
    def testset(self):
        testset = torch.utils.data.TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long)
        )
        return testset

    @property
    def n_classes(self):
        return int(self.y_df.nunique())

    @property
    def n_features(self):
        return int(len(self.metadata_df) - 1)  # -1 for the label

    # Currently disabled
    # @property
    # def summary(self):
    #     return {
    #         'n_features': self.n_features,
    #         'n_classes': self.n_classes,
    #         # 'feature_names': self.feature_names, # TODO
    #         # 'label_name': self.label_name,  # TODO
    #         'train_test_split_ratio': '0.87',
    #         'train_test_split_random_seed': '42',
    #     }

    @property
    def feature_ranges(self):
        ranges = []
        for range in self.metadata_df[self.metadata_df.type != 'label'].range:
            range = ast.literal_eval(range)
            if range[0] == '-inf':
                range[0] = -np.inf
            if range[1] == 'inf':
                range[1] = np.inf
            # cast elements to float
            range[0], range[1] = float(range[0]), float(range[1])
            ranges.append(range)
        return np.array(ranges)

    @property
    def cat_indices(self):
        return self.metadata_df[self.metadata_df.type == 'categorical'].index.values

    @property
    def cont_indices(self):
        return self.metadata_df[self.metadata_df.type == 'continuous'].index.values

    @property
    def ordinal_indices(self):
        return self.metadata_df[self.metadata_df.type == 'ordinal'].index.values

    @property
    def one_hot_groups_dict(self) -> Dict[str, List[int]]:
        """
        :return: a dictionary mapping categorical feature names to the indices of the one-hot encoded categories
        """
        oh_groups_dict = {}
        for idx, row in self.metadata_df[self.metadata_df.type == 'categorical'].iterrows():
            cat_name = row.feature_name
            if cat_name not in oh_groups_dict:
                oh_groups_dict[cat_name] = []
            oh_groups_dict[cat_name].append(idx)
        return oh_groups_dict

    @property
    def one_hot_groups(self) -> List[np.ndarray]:
        """
        :return: a list of lists, each inner list contains the indices of the one-hot encoded categories
        """
        return [np.array(indices) for indices in self.one_hot_groups_dict.values()]

    @property
    def standard_factors(self) -> np.ndarray:
        robustness_gap = 0.0
        return (
                self.x_df.quantile(1 - robustness_gap) -
                self.x_df.quantile(0 + robustness_gap)
        ).values

    @property
    def is_feature_ordinal(self):
        return (self.metadata_df[self.metadata_df.type != 'label'].type == 'ordinal').tolist()

    @property
    def is_feature_continuous(self):
        return (self.metadata_df[self.metadata_df.type != 'label'].type == 'continuous').tolist()

    @property
    def feature_types(self) -> List[Type]:
        return self.metadata_df[self.metadata_df.type != 'label'].type.apply(
            lambda type: {'categorical': str, 'ordinal': int, 'continuous': float}[type]
        ).values.tolist()

    @property
    def feature_names_dcs_format(self) -> List[str]:
        """
        :return: a list of feature names, in the format used in the DCs mining algorithm (FastADC)
        """
        return self.metadata_df[self.metadata_df.type != 'label'].apply(
            lambda row: f"{row.feature_name}({row['dc-mining-type'].title()})", axis=1
        ).values.tolist()

    def _validate_input(self):
        f_names_from_metadata = self.metadata_df[self.metadata_df.type != 'label'].feature_name.values.tolist()
        f_names_from_data = [col.split('===')[0] for col in self.x_df.columns.tolist()]
        assert f_names_from_metadata == f_names_from_data, \
            "Order and names of features in metadata and data should be the same"
        assert self.y_df.name == self.metadata_df[self.metadata_df.type == 'label'].feature_name.item(), \
            "Label name in metadata and data should be the same"

        assert np.all(self.feature_ranges[:, 0] <= self.x_df.values), \
            "Feature ranges should be at least as wide as the data"
        assert np.all(self.feature_ranges[:, 1] >= self.x_df.values), \
            "Feature ranges should be at least as wide as the data"

    @staticmethod
    def cast_sample_format(sample: np.ndarray, from_dataset, to_dataset) -> np.ndarray:
        """
        Transform a sample from the original dataset to this dataset's format.
            * casting is done in terms of order features and re-encoding categorical features.
            * can be used to cast model-input to SAT solver input.
        :param sample: a sample from the original dataset
        :param from_dataset: the dataset object from which the sample was originated
        :param to_dataset: teh target dataset object, to which the sample should be cast
        :return: a sample in this dataset's format
        """
        # TODO assert origin this dataset is castable to this (same data, same feature names, same label name)
        assert len(sample) == len(from_dataset.feature_names), "`from_dataset` should be the object where sample from"

        sample_new = np.zeros_like(to_dataset.feature_names)
        for f_idx, row in to_dataset.metadata_df[to_dataset.metadata_df.type != 'label'].iterrows():
            feature_name = row.feature_name

            if (row.type == 'categorical' and
                    from_dataset.cat_encoding_method == 'one_hot_encoding' and to_dataset.cat_encoding_method is None):
                # Extract the category from the 'original' one-hot-encoding
                oh_group_indices = from_dataset.one_hot_groups_dict[feature_name]
                oh_category_idx = oh_group_indices[sample[oh_group_indices].argmax()]
                oh_category = from_dataset.metadata_df.one_hot_encoding[oh_category_idx]

                # Update the new sample with the new category-encoding
                cat_to_enc_label = {cat: enc_label for enc_label, cat in row.encoding_map.items()}
                sample_new[f_idx] = cat_to_enc_label[oh_category]

            elif (row.type == 'categorical' and
                  from_dataset.cat_encoding_method is None and to_dataset.cat_encoding_method == 'one_hot_encoding'):
                # Extract the category from the original label-encoded dataset
                encoding_map_origin = from_dataset.metadata_df[from_dataset.metadata_df.feature_name == feature_name].encoding_map.item()
                f_idx_in_origin_dataset = (from_dataset.feature_names == feature_name).argmax()
                encoded_category = encoding_map_origin[sample[f_idx_in_origin_dataset]]

                # Update the new one-hot-encoded sample
                sample_new[f_idx] = int(encoded_category == row.one_hot_encoding)

            else:  # for any other feature, we simply copy the value as is
                # Get the index where the feature name is in the original dataset
                f_idx_in_origin_dataset = (from_dataset.feature_names == feature_name).argmax()
                sample_new[f_idx] = sample[f_idx_in_origin_dataset]

        sample_new = sample_new.astype(np.float32)
        return sample_new
