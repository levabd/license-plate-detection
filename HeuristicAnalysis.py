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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # get mask
    maskpath = imagepath.replace('.jpg', 'M.jpg')
    mask = cv2.imread(maskpath)
    mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Heuristic analysis and priority selection of number plate candidates
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''''''''''''''''''''''''''
    Basic Heuristic Analysis
    '''''''''''''''''''''''''''
    basicHeuristicAnalysisRanges = []

    for r in lpranges:
        # Analysis # 1
        a1 = r[1] - r[0]
        # Analysis # 2
        band = gradX[r[0]:r[1], r[2]:r[3]]
        verp = tl.projectionVer(band)
        a2 = np.max(verp)
        # Analysis # 3
        a3 = np.sum(verp)
        # Analysis # 4
        a4 = abs((r[3] - r[2]) / a1 - 4.64)
        asum = 0.15 * a1 + 0.25 * a2 + 0.4 * a3 + 0.4 * a4
        basicHeuristicAnalysisRanges.append((r[0], r[1], r[2], r[3], asum))

        # h = r[1]-r[0]
        # w = r[3]-r[2]

        # if minPlateW * 2 < w and w < maxPlateW * 2 and abs(w/h-5) < 3:
        #     band = gradX[r[0]:r[1],r[2]:r[3]]
        #     verp = tl.projectionVer(band)
        #     # sortParam = np.sum(verp)
        #     sortParam = np.max(verp)
        #     basicHeuristicAnalysisRanges.append((r[0], r[1], r[2], r[3], sortParam))

    temp = np.asarray(basicHeuristicAnalysisRanges)
    basicHeuristicAnalysisRanges = temp[temp[:, 4].argsort()]
    basicHeuristicAnalysisRanges = basicHeuristicAnalysisRanges.astype(int)

    '''''''''''''''''''''''''''
    Deep Heuristic Analysis
    '''''''''''''''''''''''''''
    _verprojsBHA = np.zeros((1, img.shape[1]), dtype=np.uint8)

    for r in basicHeuristicAnalysisRanges:
        _band = gray[r[0]:r[1], r[2]:r[3]]
        if _band.shape[0] > 0 and _band.shape[1] > 0:
            bmin = np.min(_band)
            bmax = np.max(_band)
            bmid = int((bmax - bmin) / 2)
            hist, bins = np.histogram(_band.ravel(), 256, [0, 256])
            b2 = np.sum(hist[bmid:bmax]) - np.sum(hist[bmin:bmid])

            # Debug Heuristic analysis
            _img = img[r[0]:r[1], r[2]:r[3]]
            _color = (0, 255, 0) if b2 > 0 else (0, 0, 255)
            _img = cv2.copyMakeBorder(_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                      value=_color)
            _verprojsBHA = tl.concat_ver2((_verprojsBHA, _img))

    showImg = tl.concat_hor((img, _verprojsBHA))
    cv2.imwrite("debug_imgs/HBA/" + str(framenum) + ".jpg", showImg)
    framenum += 1