import os

def start(path):
    """get image names (mask, incorrect, normal)

    Args:
        path ([str]): path of image

    Returns:
        [list]: image names
    """
    imgs_path = []
    for _dir in os.listdir(path):
        if not _dir.startswith('.'): #숨긴 파일은 적용 X 하기 위해
            for img in os.listdir(path + '/' + _dir):
                if not img.startswith('.'): #숨긴 파일은 적용 X 하기 위해
                    imgs_path.append(_dir + '/' + img)

    return imgs_path
