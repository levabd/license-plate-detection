from __future__ import division

import os
import cPickle
import tools as tl
import cv2
import numpy as np

# Delete all files
folders = ["HBA"]
for folder in folders:
    for the_file in os.listdir("debug_imgs/" + folder):
        file_path = os.path.join("debug_imgs/" + folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

ranges = cPickle.load(open("Chars.p", "r"))

# setup some parameters
scale = 0.5

minPlateW = scale * 60
maxPlateW = scale * 180

minPlateH = minPlateW / 4.64
maxPlateH = maxPlateW / 4.64

framenum = 0

for (imagepath, lpranges) in ranges:
    print imagepath + ' ' + str(framenum)

    img = cv2.imread(imagepath)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    gray = cv2.fastNlMeansDenoising(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                    None,10,7,21)

    # get mask
    maskpath = imagepath.replace('.jpg', 'M.jpg')
    mask = cv2.imread(maskpath)
    mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Heuristic analysis and priority selection of number plate candidates
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    _verprojsBHA = np.zeros((1, img.shape[1]), dtype=np.uint8)

    '''''''''''''''''''''''''''
    Basic Heuristic Analysis
    '''''''''''''''''''''''''''
    basicHeuristicAnalysisRanges = []

    maxDensity = 0
    secondMaxDensity = 0
    thirdMaxDensity = 0
    globalFlagDetected = False

    for r in lpranges:

        h = r[1] - r[0]
        w = r[3] - r[2]

        _band = gray[r[0]:r[1], r[2]:r[3]]
        _bandSource = img[r[0]:r[1], r[2]:r[3]]
        rborder = r[2] + int(w / 6)
        bborder = r[0] + int(h / 2)
        _bandFlag = img[r[0]:bborder, r[2]:rborder]

        if w / h > 4.87 and w / h < 8.14 and w < gray.shape[1] and h > 13:
            edges = cv2.Canny(_band, 100, 100)

            maxVariance = np.var(_bandSource)
            edgesSum = np.sum(edges)
            density = edgesSum / (w * h)

            lower_flag = np.array([80, 96, 55], dtype=np.uint8)
            upper_flag = np.array([135, 140, 82], dtype=np.uint8)

            lower_flag2 = np.array([142, 130, 55], dtype=np.uint8)
            upper_flag2 = np.array([190, 170, 95], dtype=np.uint8)
            flagDetected = False

            if (np.sum(cv2.inRange(_bandFlag, lower_flag, upper_flag)) > #509 for 7
                    765) or (np.sum(cv2.inRange(_bandFlag, lower_flag2,
                                                upper_flag2)) > 765):
                flagDetected = True
                globalFlagDetected = True

            #Debug
            if framenum == -1:

                #print np.sum(cv2.inRange(_bandFlag, lower_flag, upper_flag))
                #print np.sum(cv2.inRange(_bandFlag, lower_flag2, upper_flag2))

                #cv2.imshow("Result", cv2.resize(cv2.bitwise_and(_bandFlag,
                # _bandFlag, mask = cv2.inRange(_bandFlag,
                #                lower_flag, upper_flag)),(0,0),fx=3,fy=3))
                #cv2.waitKey(0)

                print "Variance: " + str(maxVariance)
                print "Density: " + str(density)
                #print "Horisontal Density: " + str(edgesSum / h)
                #print "Vertical Density: " + str(edgesSum / w)

            if maxVariance > 260:
                if maxDensity < density:
                    if maxDensity > 0:
                        if secondMaxDensity > 0:
                            thirdMaxDensity = secondMaxDensity
                        secondMaxDensity = maxDensity
                    maxDensity = density

                # Skoree vsego niznih ciklov ne nado
                if (secondMaxDensity < density) and (density < maxDensity) :
                    if secondMaxDensity > 0:
                        thirdMaxDensity = secondMaxDensity
                    secondMaxDensity = density
                if (thirdMaxDensity < density) and (density < maxDensity) and (density < secondMaxDensity):
                    thirdMaxDensity = density
            basicHeuristicAnalysisRanges.append(
                (r[0], r[1], r[2], r[3], maxVariance, density,
                 flagDetected, _band, edges))
        else:
            _img = img[r[0]:r[1], r[2]:r[3]]
            _img = cv2.copyMakeBorder(_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                      value=(0, 0, 255))
            _verprojsBHA = tl.concat_ver2((_verprojsBHA, _img))

    '''''''''''''''''''''''''''
    Deep Heuristic Analysis
    '''''''''''''''''''''''''''

    for r in basicHeuristicAnalysisRanges:
        _band = gray[r[0]:r[1], r[2]:r[3]]
        if _band.shape[0] > 0 and _band.shape[1] > 0:

            _img = img[r[0]:r[1], r[2]:r[3]]

            h = r[1] - r[0]
            w = r[3] - r[2]

            _color = (255, 127, 0)

            if r[6]:
                _color = (0, 255, 0)

            if globalFlagDetected == False:
                if r[5] == maxDensity:
                    _color = (0, 255, 0)
                elif r[5] == secondMaxDensity:
                    _color = (212, 255, 127)
                elif r[5] == thirdMaxDensity:
                    _color = (255, 255, 0)

                if r[4] < 260:
                    _color = (0, 191, 255)

            if len(basicHeuristicAnalysisRanges) == 1:
                _color = (0, 255, 0)

            _img = tl.concat_hor((_img, r[7], r[8]))
            _img = cv2.copyMakeBorder(_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                      value=_color)
            _verprojsBHA = tl.concat_ver2((_verprojsBHA, _img))

    showImg = tl.concat_hor((img, _verprojsBHA))
    cv2.imwrite("debug_imgs/HBA/" + str(framenum) + ".jpg", showImg)
    framenum += 1