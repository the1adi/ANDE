import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# Ensure images read in RGB format for consistency with moviepy
def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# Perform calibration using chessboard images
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

calibration_fnames = glob('camera_cal/calibration*.jpg')

calibration_images = []
objpoints = []
imgpoints = []

for fname in calibration_fnames:
    img = read_image(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    calibration_images.append(gray)
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        plt.subplot(4, 3, len(imgpoints))
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        plt.imshow(img)
        # plt.title(fname)
        plt.axis('off')
plt.show()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,calibration_images[0].shape[::-1], None, None)
calibration = [mtx,dist]
print('Corners were found on', str(len(imgpoints)), 'out of', str(len(calibration_images)), 'it is',    str(len(imgpoints)*100.0/len(calibration_images)),'% of calibration images')
print(calibration)
