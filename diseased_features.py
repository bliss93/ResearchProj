import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

for img in glob.glob('C:/Users/User/PycharmProjects/researchP/Diseased_L/*.jpg'):
    imgRead = cv2.imread(img, 1)

    cv2.namedWindow('resized_w01', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('resized_w01', 400, 750)

    z = imgRead.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]

    res01 = res.reshape((imgRead.shape))

    imgray = cv2.cvtColor(res01, cv2.COLOR_BGR2GRAY)

    # erosion
    kernel01 = np.ones((5, 5), np.uint8)
    erosion01 = cv2.erode(imgray, kernel01, iterations=1)

    ret, th_gray = cv2.threshold(erosion01, 127, 255, 0)

    bit_andImg = cv2.bitwise_and(res01, res01, mask=th_gray)

    _, cnts, _ = cv2.findContours(th_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) != 0:
        #        cv2.drawContours(bit_andImg,cnts,-1,[0,0,255],10)

        max_area = max(cnts, key=cv2.contourArea)



        cv2.drawContours(bit_andImg, max_area, -1, [0, 0, 255], 10)

        # Contour Approximation
        epsilon = 0.1 * cv2.arcLength(max_area, True)
        approx = cv2.approxPolyDP(max_area, epsilon, True)

        area = cv2.contourArea(max_area)

        # print  area

        print 'detected area in the spot -->>  ' + str(area)

        # print 'detected area in the  spot -->>  ' + str(np.countNonZero(max_area))

        equa_diameter = np.sqrt(4 * area / np.pi)

        print 'detected diameter in the spot -->>  ' + str(equa_diameter)

       #finalnpzero = np.zeros(imgRead.shape,np.uint8)
        #masknpzero = np.zeros(imgray.shape,np.uint8)
       #
       #for i in xrange(0,len(cnts)) :
       #    masknpzero[...] =0
       #    cv2.drawContours(masknpzero,cnts,i,255,-1)
       #    cv2.drawContours(finalnpzero,cnts,i,cv2.mean(imgRead,masknpzero),-1)
       #
       #    print 'average color of the detected spot-->>' +str(cv2.mean(imgRead,masknpzero))
       #
        







        #print 'detected spot area number of cols and num of rows -->>' +str(max_areashape)

        #spot_row = max_areashape[0]
        #spot_col = max_areashape[1]
        #
        #color = int(max_areashape[spot_row,spot_col])
        #print         'detected spot colour of diseased area -->>' + str(color)

        print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'

        cv2.imshow('resized_w01', bit_andImg)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

        #  print approx
        #  print '**************************************'




