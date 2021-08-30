import pandas as pd
import os
from tqdm import tqdm
import glob

file_feature = ["mask", "incorrect_mask", "normal"]


def get_train_img_path(train_dir, img_path, feature=None):
    """
    Generate real path for img_path and featured.
    Return without extension. (jpg, png ...)
    """
    if feature is None:
        """
        Generate all path for img_path
        """
        path = []
        for feature in file_feature:
            result = get_train_img_path(train_dir, img_path, feature)
            if isinstance(result, list):
                path.extend(result)
            else:
                path.append(result)
    elif feature == "mask":
        path = [
            os.path.join(train_dir, img_path, f'{feature}{i}') for i in range(1, 6)
        ]
    else:
        path = os.path.join(train_dir, img_path, feature)
    return path


def get_test_img_path(test_pd, image_dir):
    """Return with extension. (jpg, png, ...) """
    return [
        os.path.join(image_dir, image_id) for image_id in test_pd["ImageID"]
    ]


class DataFrameModule:
    """
    Manage dataframe for mask database.
    """

    def __init__(self, data_df, images_dir):
        self.data_df = data_df
        print(self.data_df.size)
        self.images_dir = images_dir
        self.system_path_column = "system_path"

    def get_df_with_path(self, feature=None, train=True) -> pd.DataFrame:
        """
        Generate dataframe for given parmaeters and return.
            feature:
                If featuer is None, get all features.
        """
        new_column = list(self.data_df.columns) + [self.system_path_column]
        new_df = pd.DataFrame(columns=new_column)
        count = 0
        for idx in tqdm(range(self.data_df.shape[0])):
            # Generate all real path with given featuer
            for path in self.get_path(idx, feature):
                new_df.loc[count] = list(self.data_df.loc[idx]) + path
                count += 1
        return new_df

    def get_path(self, idx, feature) -> list:
        # merge path and feature
        base_path = self.data_df.iloc[idx, -1]

        # Get all possilbe path for base_path and feature
        target_path = get_train_img_path(self.images_dir, base_path, feature)

        # Append asterisk for using glob, cause all the images have different extension.
        if isinstance(target_path, list):
            target_path = [p + "*" for p in target_path]
            target_path = [glob.glob(p) for p in target_path]
        elif isinstance(target_path, str):
            target_path = target_path + "*"
            target_path = glob.glob(target_path)
        return target_path


def generate_csv(train_csv, train_dir, file_path):
    train_pd = pd.read_csv(train_csv)
    train_df_manager = DataFrameModule(train_pd, train_dir)
    csv_file = train_df_manager.get_df_with_path()
    csv_file.to_csv(file_path)
    print("Generate csv file!!")

