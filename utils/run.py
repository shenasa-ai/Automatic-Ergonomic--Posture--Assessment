from fleiss_kapa import FleissKappa
from final_label_images import finalize_image_labels
from clear_labels import clean_labels
from Evaluation import calculate_accuracy

'''
1 - run finalize_image_labels
2 - run clean_labels
'''

# path = ['./../labels/aa.csv', './../labels/fj.csv', './../labels/ha.csv']
# image2remove = [123, 89, 96, 158, 159, 171, 172, 213, 232, 244, 245, 247, 248, 265, 284]
#
# finalize_image_labels(img_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/side', lbl_paths=path, save_path='./../labels', rem_img=True)
# clean_labels(lbl_paths='./../labels/final_labels.csv',rem_img=image2remove , img_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/input/main_input/side', save_path='./../labels')

# Evaluate
print(calculate_accuracy(pred_part='monitor', actual_part='monitor', pred_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/output/pred_Openpifpaf.csv', actual_path='/home/ali/Desktop/Automatic-Ergonomic-Posture-Assessment/labels/final_labels_cleared.csv'))
