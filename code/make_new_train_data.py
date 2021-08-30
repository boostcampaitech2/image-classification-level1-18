import csv

def start(img_names,labels):
    """make new train_data(path, label) and save as csv file

    Args:
        img_names ([list]): image name(path)
        labels ([list]): class(label)
    """
    col_name=[['imageName','label']]

    for img_name, label in zip(img_names,labels):
        col_name.append([img_name,label])

    with open('labeled_train_data.csv','w') as file:
        write = csv.writer(file)
        write.writerows(col_name)
        
