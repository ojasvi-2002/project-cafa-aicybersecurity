import pandas as pd
from sklearn import preprocessing


def add_one_hot_encoding(df: pd.DataFrame, metadata_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
        return updated df and metadata_df, after one-hot-encoding all categorical features in df
    """

    # list the features to one-hot-encode
    cat_features = metadata_df[metadata_df.type == 'categorical'].name.to_list()
    cats_to_one_hot = []
    for cat_feature in cat_features:
        if df[cat_feature].nunique() > 2:
            cats_to_one_hot.append(cat_feature)

    # one-hot-columns will be of form "{original_categorical_feature}==={encoded_value}"
    df = pd.get_dummies(df, columns=cats_to_one_hot, prefix_sep="===", drop_first=False)

    # updated metadata df, with new one-hot coordinates
    new_metadata_rows = []

    for feature_name in df.columns:
        # add the row to the new_metadata_df, in the simplest way
        # get the dict of the row where 'metadata_df.name == feature_name'
        if feature_name.split('===')[0] in cats_to_one_hot:
            # if the feature was one-hot encoded, add a row for its one-hot-encoding coordinate
            feature_name, encoded_val = feature_name.split('===')  # split to original feature name and encoded val
            new_metadata_row = metadata_df[metadata_df.name == feature_name].to_dict(orient='records')[0]
            new_metadata_row.update({'range': '[0, 1]', 'one_hot_encoding': encoded_val})
        else:  # if the feature was not one-hot encoded, add it as is
            new_metadata_row = metadata_df[metadata_df.name == feature_name].to_dict(orient='records')[0]

        new_metadata_rows.append(new_metadata_row)

    return df, pd.DataFrame(new_metadata_rows)


def add_mapping_encoding(df: pd.DataFrame, metadata_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
        return updated df and metadata_df, after mapping all categorical features in df to integers
    """
    # list the features to map
    metadata_df['encoding_map'] = None
    for cat_feature in metadata_df[metadata_df.type == 'categorical'].name:
        # encode categories in integers
        le = preprocessing.LabelEncoder()
        df[cat_feature] = le.fit_transform(df[cat_feature])
        metadata_df.loc[metadata_df.name == cat_feature, 'encoding_map'] = [{idx: actual_val for idx, actual_val in
                                                                             enumerate(le.classes_.tolist())}]
    return df, metadata_df
