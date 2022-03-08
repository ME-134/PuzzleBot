import cv2
from vision import *
from thomas_detector import ThomasDetector

matcher = VisionMatcher('../done_exploded_colored2.jpg')
detector = ThomasDetector()
matched = None
warped = None
count = 50
cv2.namedWindow("preview")
cv2.namedWindow("piece")
cv2.namedWindow("match")
vc = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    detector.process(frame)
    detector.pieces = [p for p in detector.pieces if p.is_valid()]
    if (len(detector.pieces) > 0):
        piece = detector.pieces[0]
        warped = piece.natural_img * (piece.img.reshape(piece.img.shape[0], piece.img.shape[1], 1) > 128)
        cv2.imshow("piece", warped)
        cors = matcher.calculate_xyrot(cvt_color(warped))
        matched = cvt_color(matcher.piece_grid[cors[0][0]][cors[0][1]].natural_img)
        cv2.imshow("match", matched)

    rval, frame = vc.read()
    key = cv2.waitKey(0)
    if key == 27: # exit on ESC
        break
    elif (key == 98): # Bad
        count += 1
        if (len(detector.pieces) > 0):
            cv2.imwrite(f'../vision/imgs/bad1_{count}.jpg', detector.pieces[0].natural_img)
            cv2.imwrite(f'../vision/imgs/bad2_{count}.jpg', matched)
    elif (key==103): # Good
        count += 1
        if (len(detector.pieces) > 0):
            cv2.imwrite(f'../vision/imgs/good1_{count}.jpg', detector.pieces[0].natural_img)
            cv2.imwrite(f'../vision/imgs/good2_{count}.jpg', matched)
    else:
        print(key)
cv2.destroyWindow("preview")
vc.release()