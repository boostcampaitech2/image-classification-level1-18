
def start(images_name):
    """make labels by conditions(class)

    Args:
        images_name ([list]): image name

    Returns:
        [list]: class
    """
    labels = []
    for path in images_name:
        _dir, img = path.split('/')
        _, sex, _, age = _dir.split('_')
        age = int(age)
        if img.startswith('mask'):
            if sex == 'male':
                if age < 28:
                    labels.append(0)
                elif age >= 59:
                    labels.append(2)
                else:
                    labels.append(1)
            elif sex == 'female':
                if age < 28:
                    labels.append(3)
                elif age >= 59:
                    labels.append(5)
                else:
                    labels.append(4)
        elif img.startswith('incorrect'):
            if sex == 'male':
                if age < 28:
                    labels.append(6)
                elif age >= 59:
                    labels.append(8)
                else:
                    labels.append(7)
            elif sex == 'female':
                if age < 28:
                    labels.append(9)
                elif age >= 59:
                    labels.append(11)
                else:
                    labels.append(10)
        elif img.startswith('norm'):
            if sex == 'male':
                if age < 28:
                    labels.append(12)
                elif age >= 59:
                    labels.append(14)
                else:
                    labels.append(13)
            elif sex == 'female':
                if age < 28:
                    labels.append(15)
                elif age >= 59:
                    labels.append(17)
                else:
                    labels.append(16)

    return labels
