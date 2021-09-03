"""
mask
    wear: 0
    incorrect: 1
    not wear: 2
gender
    male: 0
    female: 1
age
    <30: 0
    >=30 and <60: 1
    >=60: 2
"""
import os
from itertools import product
class FileNameError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class Label:
    mask = [0, 1, 2]
    gender = [0, 1]
    age = [0, 1, 2]
    def __init__(self):
        self.feature_func = {
            "gender": self.gender_feature,
            "mask": self.mask_feature,
            "age": self.age_feature,
        }

        self.label_number = list(product(Label.mask, Label.gender, Label.age))
        # print(self.label_number)

    def get_class_num(self, feature) -> list:
        if feature == 'Merged feature':
            return 18
        else:
            return len(getattr(Label, feature))

    def merge_feature(self, path) -> int:
        return self.label_number.index((self.mask_feature(path), self.gender_feature(path), self.age_feature(path)))

    def mask_feature(self, path) -> int:
        file_name = path.split(os.sep)[-1]
        if file_name[:4] == "mask":
            return 0
        elif file_name[:14] == "incorrect_mask":
            return 1
        elif file_name[:6] == "normal":
            return 2
        else:
            raise FileNameError("Mask naming error")

    def gender_feature(self, path) -> int:
        gender = path.split(os.sep)[-2].split("_")[1]
        if gender == "male":
            return 0
        elif gender == "female":
            return 1
        else:
            raise FileNameError("Gender naming error")

    def age_feature(self, path) -> int:
        age = int(path.split(os.sep)[-2][-2:])
        if age < 29:
            return 0
        elif 29 <= age < 59:
            return 1
        elif age >= 59:
            return 2
        else:
            raise FileNameError("Age naming error")

    def get_label(self, path: str, feature: str) -> int:
        try:
            if feature == 'Merged feature':
                return self.merge_feature(path)
            else:
                return self.feature_func[feature](path)
        except FileNameError as e:
            print(e)
            exit()

