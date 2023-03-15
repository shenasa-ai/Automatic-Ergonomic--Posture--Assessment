from pose_detector import PoseDetector
import cv2
import openpifpaf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

class OpenpifpafPoseDetector(PoseDetector):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16
    Background = 18

    def __init__(self):
        self.predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')
        self.im_skeleton = None
        #self.image = None
        
    def resize(self, image):
        r = 400.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 400)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image, 'RGB')
        return image, dim


    def calc_disply_joints(self, prediction):
        plt.imshow(self.image)
        xx = []
        yy = []
        zz = []
        # for p in range(len(prediction)):
        for i in range(17):
            # predictions[0].data[i]
            x = prediction[i].data[0]
            y = prediction[i].data[1]
            z = prediction[i].data[2]
            if x > 0 and y > 0:
                xx.append(x)
                yy.append(y)
                zz.append(z)
        im2 = self.image.copy()
        #plt.sca(axs.flatten()[0])
        plt.imshow(im2)
        plt.scatter(xx, yy, s=3)
        plt.title('Joints')
        plt.axis('off')

    def preprocess_image(self, image):
        self.image, dim = self.resize(image)
        predictions, gt_anns, image_meta = self.predictor.pil_image(self.image)
        
        pred = []
        for i in predictions:
            pred.append(i.json_data())
        for i in range(0, len(pred)):
            new_keypoint2 = []
            keypoints = [pred[i]['keypoints'][s:s + 3:] for s in range(0, len(pred[i]['keypoints']), 3)]
            keypoints = np.array(keypoints)
            self.calc_disply_joints(keypoints)
            new_keypoints = list(map(lambda x: tuple([int(x[0]), int(x[1])]), keypoints))
            prob = list(map(lambda x: x[2], keypoints))
            for i in range(len(prob)):
                if prob[i] > 0.1:  # th
                    new_keypoint2.append(new_keypoints[i])
                else:
                    new_keypoint2.append(None)

            self.model_output = new_keypoint2

        return np.array(self.image)

    def get_joint_points(self) -> []:
        return self.model_output

    '''def get_joint_points(self) -> []:

        height, width = self.image.size
        scale_x = width / self.model_output.shape[0]
        scale_y = height / self.model_output.shape[1]
        
        points = []
        threshold = 0.1
        
        for i in range(self.points_number):
            prob_map = model_output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            x = scale_x * point[0]
            y = scale_y * point[1]
            
            if prob > threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
                
                return points
    '''

