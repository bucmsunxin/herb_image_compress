from util import *
import random
import math
import pdb

random.seed(100)
root_dir = '../data'
image_dir = '%s/images' % root_dir
train_db_file = '%s/train.pkl' % root_dir
val_db_file = '%s/val.pkl' % root_dir
train_db = []
val_db = []
ratio = 0.7

class_list = os.listdir(image_dir)
for cls_id, class_name in enumerate(class_list):
    label = int(cls_id)
    print('%d : %s' % (label, class_name))
    class_dir = '%s/%s' % (image_dir, class_name)
    image_list = os.listdir(class_dir)
    num_images = len(image_list)
    num_train = int(math.floor(num_images * ratio))
    random.shuffle(image_list)

    for idx, img_name in enumerate(image_list):
        entry = {'imfile': 'images/%s/%s' % (class_name, img_name),
                 'label' : label}
        if idx < num_train:
            train_db.append(entry)
        else:
            val_db.append(entry)

random.shuffle(train_db)
random.shuffle(val_db)
save_file(train_db, train_db_file)
save_file(val_db, val_db_file)
print('num_train : %d' % len(train_db))
print('num_test  : %d' % len(val_db))
