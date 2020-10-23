import cv2
from SimpleHigherHRNet import SimpleHigherHRNet
import numpy as np


if __name__ == "__main__":
    model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")
    image = cv2.imread("b8155f9f5c690dd8e8025d192d1609c96617-photo.jpg", cv2.IMREAD_COLOR)
    
    joints = model.predict(image)
    print('joints shape:{}'.format(joints.shape))
    # print(joints)
    joints = joints[0]
    joints = joints[:, [1, 0, 2]]
    
    color = np.random.randint(0, 255, size=3)
    color = [int(i) for i in color]
    res_img = SimpleHigherHRNet.vis_joints(image, joints, color, True)
    cv2.imwrite('result.jpg', res_img)
    print('test done')