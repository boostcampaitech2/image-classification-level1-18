import csv

def start(img_names,ids,labels):
    """make new train_data(path, label) and save as csv file

    Args:
        img_names ([list]): image name(path)
        labels ([list]): class(label)
    """
    col_name=[['imageName','id','label']]

    for img_name,id, label in zip(img_names,ids,labels):
        col_name.append([img_name,id,label])

    with open('labeled_train_data.csv','w') as file:
        write = csv.writer(file)
        write.writerows(col_name)
        
