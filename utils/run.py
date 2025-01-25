from fleiss_kapa import FleissKappa
from final_label_images import finalize_image_labels
from clear_labels import clean_labels

'''
1 - run finalize_image_labels
2 - run clean_labels
'''

path = ['/home/ali/Desktop/python/posture/labels/ha_checked_side.csv', '/home/ali/Desktop/python/posture/labels/Labels 2 - side_fj.csv', '/home/ali/Desktop/python/posture/labels/Labels 2 - side_aa.csv']

clean_labels(images_dir='/home/ali/Desktop/python/posture/input/side', labels_path='./final_labels.csv')