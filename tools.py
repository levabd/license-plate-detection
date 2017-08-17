import cv2
import numpy as np
import tools as tl

def findb0(verpConvolved, ybm, c):
    for i in range(ybm,-1,-1):
        if verpConvolved[i] <= c:
            return i

    return 0

def findb1(verpConvolved, ybm, c):
    for i in range(ybm,len(verpConvolved)):
        if verpConvolved[i] <= c:
            return i

    return len(verpConvolved)

def draw_points(img, points):
    for point in points:    
        cv2.circle(img, (point[0], point[1]), 2, (0, 0, 255))
    return img


def pcp2Convolution(horp, h):
    horpc = np.zeros(horp.shape[0])
    pref = np.zeros(h)
    horp = np.concatenate((pref, horp))

    for i in range(horpc.shape[0]):
        horpc[i] = (horp[i+h] - horp[i])/h
        
    return horpc

def rotate_plate(lpgray):
    '''
        Calculating accumulator threshold dynamically
        by dividing width of image by divider, 
        which starts from 2.5 and increments by 0.5
        while number of founded lines will be more than 10
    '''
    gradX = cv2.Sobel(lpgray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    ret, thres = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    divider = 2.5
    while True:
        accum_thres = int(thres.shape[1]/divider)
        divider += 0.5

        if divider > 10:
            break

        lines = cv2.HoughLines(thres,1,np.pi/180,accum_thres)

        if lines is None:
            continue

        if lines.shape[0] < 5:
            continue

        lines = np.reshape(lines,(lines.shape[0],2))

        c = 180/np.pi
        angles = lines[:,1]*c
        angles = angles.astype(np.int)

        bins = np.bincount(angles)
        return np.argmax(bins) - 90, thres

    return 0, thres

def concat_hor(imgs):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[0])
        s += img.shape[1]+2*bs


    image = np.zeros((m+2*bs, s, 3))

    x = 0
    for img in imgs:
        if len(img.shape) == 3:
            imgg = cv2.copyMakeBorder(img.copy(), bs, bs, bs, bs,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image[0:imgg.shape[0], x:x+imgg.shape[1], :] = imgg
        else:
            imgg = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                      bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image[0:imgg.shape[0], x:x+imgg.shape[1], :] = imgg
        x += img.shape[1]+2*bs

    return np.asarray(image, dtype = np.uint8)

def concat_ver(imgs):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[1])
        s += img.shape[0]+2*bs


    image = np.zeros((s, m+2*bs, 3))

    y = 0
    for img in imgs:
        if len(img.shape) == 3:
            imgg = cv2.copyMakeBorder(img.copy(), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image[y:y+imgg.shape[0], 0:imgg.shape[1], :] = imgg
        else:
            imgg = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                      bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image[y:y+imgg.shape[0], 0:imgg.shape[1], :] = imgg
        y += img.shape[0]+2*bs

    return np.asarray(image, dtype = np.uint8)

def concat_ver2(imgs):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[1])
        s += img.shape[0]+2*bs


    image = np.zeros((s, m+2*bs, 3))

    y = 0
    for img in imgs:
        if len(img.shape) == 3:
            image[y:y+img.shape[0], 0:img.shape[1], :] = img
        else:
            imgg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image[y:y+imgg.shape[0], 0:imgg.shape[1], :] = imgg
        y += img.shape[0]+2*bs

    return np.asarray(image, dtype = np.uint8)

def draw_graphic(lp, verp = True, horp = False):
    rimg = lp

    verLines = projectionVer(lp)
    horLines = projectionHor(lp)

    graphicHor = np.zeros((100, lp.shape[1]), dtype = "uint8")
    graphicVer = np.zeros((lp.shape[0], 100), dtype = "uint8")
    
    for i in range(len(horLines)):
        graphicHor[graphicHor.shape[0]-horLines[i]:graphicHor.shape[0], i] = 255

    for i in range(len(verLines)):
        graphicVer[i, 0:verLines[i]] = 255

    if verp:
        rimg = concat_hor((rimg, graphicVer))

    if horp:
        rimg = concat_ver((rimg, graphicHor))
    
    return rimg

def getDrawProjectionVer(lp, verp):
    verp2 = verp.astype(int)
    w = np.max(verp2) + 5
    graphicVer = np.zeros((lp.shape[0], w), dtype = "uint8")

    for i in range(len(verp2)):
        graphicVer[i, 0:verp2[i]] = 255

    return graphicVer

def getDrawProjectionHor(lp, horp):
    horp2 = horp.astype(int)
    h = int(np.max(horp2) + 5)
    graphicHor = np.zeros((h, lp.shape[1]), dtype = "uint8")

    for i in range(len(horp2)):
        graphicHor[int(graphicHor.shape[0]-horp2[i]):graphicHor.shape[0], i] = 255

    return graphicHor

def getDrawProjectionHorNeg(lp, horp):
    horp2 = horp.astype(int)
    horp2[horp2 > 0] = 0
    horp2 = np.abs(horp2)
    h = int(np.max(horp2) + 5)
    graphicHor = np.zeros((h, lp.shape[1]), dtype = "uint8")

    for i in range(len(horp2)):
        graphicHor[0:int(horp2[i]), i] = 255

    return graphicHor

def projectionVer(plate):
    return np.sum(plate, axis=1)/255


def projectionHor(plate):
    return np.sum(plate, axis=0)/255