#!/usr/bin/env python3
#
#   puzzleoffset.py
#
#   Determine the puzzle piece offset.
#

# Imports
import cv2
import numpy as np

#
#   Definitions
#
# Set the approximate piece side length (in pixels).  This is used to
# sub-divide the long side of connected pieces.
SIDELEN = 125

# Set the number of points per side to match against another side.
SIDEPOINTS = 20

# Color List (to show the sides)
COLORLIST = ((000, 000, 255),           # Red
             (000, 255, 000),           # Green
             (255, 000, 000),           # Blue
             (255, 255, 000),           # Yellow
             (000, 255, 255),           # Cyan
             (255, 000, 255),           # Magenta
             (255, 128, 000),           # Orange
             (000, 255, 128),
             (128, 000, 255),
             (255, 000, 128),
             (128, 255, 000),
             (000, 128, 255))

#
#   Draw/Fill a Contour, Draw Side(s)
#
#   Note sides are effectively open contours of just one side of a
#   puzzle piece.  Both rotate counter-clockwise around the piece.
#   And both are numpy arrays of size Nx1x2.  I.e. an (Nx1) array of
#   (x,y) pixel values.
#
def drawContour(image, contour, color):
    cv2.drawContours(image, [contour], 0, color)

def fillContour(image, contour, color):
    cv2.drawContours(image, [contour], 0, color, thickness=cv2.FILLED)

def drawSide(image, side, color):
    for x,y in side.reshape(-1,2):
        image[y,x] = color

def drawSides(image, sides):
    for i in range(len(sides)):
        drawSide(image, sides[i], COLORLIST[i % len(COLORLIST)])


#
#   Contour Information
#
def centerContour(contour):
    # Compute the center of the contour
    M = cv2.moments(contour)
    x = M["m10"] / M["m00"]
    y = M["m01"] / M["m00"]
    return(np.array([x, y]))

def infoContour(contour):
    # Compute the center of the contour
    c = centerContour(contour)
    print("Contour Center ", c)

    # Compute the contour area.
    A = cv2.contourArea(contour, oriented=False)
    print("Contour Area   ", A)

    # Compute the contour length.
    L = cv2.arcLength(contour, closed=True)
    print("Contour Len    ", L)


#
#   Base Polygon
#
#   Find the polygon underlying a puzzle piece or connected pieces.
#   This basically ignores the holes and tabs and turns each piece
#   into a square.
#
def findBase(contour, imagewidth, imageheight):
    # Create a blank image, to allow the erosion and dilation without
    # interferring with other image elements.
    binary = np.zeros((imageheight, imagewidth), dtype=np.uint8)

    # Draw the original contour shape on the blank image.
    cv2.drawContours(binary, [contour], 0, color=255, thickness=cv2.FILLED)

    # Dilate and erode to remove the holes.
    N = int(SIDELEN/8)
    binary = cv2.dilate(binary, None, iterations=N)
    binary = cv2.erode(binary,  None, iterations=N)

    # Erode and dilate to remove the tabs.
    N = int(SIDELEN/6)
    binary = cv2.erode(binary,  None, iterations=N)
    binary = cv2.dilate(binary, None, iterations=N)
    
    # Re-find the countour of the base shape.  Again, do not
    # approximate, so we get the full list of pixels on the boundary.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    base = contours[0]

    # Convert the base shape into a simple polygon.
    polygon = cv2.approxPolyDP(base, SIDELEN/5, closed=True)
    return(polygon)


#
#   Corner Indicies
#
#   Create a list of puzzle piece corners.  This also works on
#   connected pieces, effectively sub-dividing long sides.
#
def refineCornerIndex(contour, index):
    # Set up the parameters.
    N = len(contour)
    D = int(SIDELEN/6)          # Search a range +/- from the given
    d = int(SIDELEN/8)          # Compute the angle +/- this many pixels
    
    # Search for the best corner fit, checking +/- the given index.
    maxvalue = 0
    for i in range(index-D,index+D+1):
        p  = contour[(i  )%N, 0, :]
        da = contour[(i-d)%N, 0, :] - p
        db = contour[(i+d)%N, 0, :] - p
        value = (da[0]*db[1] - da[1]*db[0])**2
        if value > maxvalue:
            maxvalue = value
            index    = i%N

    # Return the best index.
    return(index)

def findCornerIndices(contour, polygon):
    # Prepare the list of corner indices.
    indices = []

    # Loop of the polygon points, sub-dividing long lines (across
    # multiple pieces) into single pieces.
    N = len(polygon)
    for i in range(N):
        p1 = polygon[ i,      0, :]
        p2 = polygon[(i+1)%N, 0, :]

        # Sub-divide as appropriate.
        n  = int(round(np.linalg.norm(p2-p1) / SIDELEN))
        for j in range(n):
            p = p1*(n-j)/n + p2*j/n

            # Find the lowest distance to all contour points.
            d = np.linalg.norm(contour-p, axis=2)
            index = int(np.argmin(d, axis=0))

            # Refine the corner index for real corners.
            if (j == 0):
                index = refineCornerIndex(contour, index)

            # Use that index.
            indices.append(index)

    # Return the indices.
    return(indices)


#
#   Find Sides
#
#   Process a contour (list of pixels on the boundary) into the sides.
#
def findSides(image, contour):
    # Create the base polygon.
    polygon = findBase(contour, image.shape[1], image.shape[0])

    # Get the indices to the corners.
    indices = findCornerIndices(contour, polygon)

    # Pull out the sides between the indicies.
    sides = []
    N = len(indices)
    for i in range(N):
        index1 = indices[i]
        index2 = indices[(i+1)%N]
        if (index1 <= index2):
            side = contour[index1:index2, 0, :]
        else:
            side = np.vstack((contour[index1:, 0, :],
                              contour[0:index2, 0, :]))
        sides.append(side)


    # Check the number of pieces (just for fun).
    A = cv2.contourArea(polygon, oriented=False)
    n = np.round(A/SIDELEN/SIDELEN)
    print("Guessing contour has %d pieces" % n)

    # Report the indices (just for debugging).
    print("Corner indices ", indices)

    # Show the polygon (just FYI).
    imagecopy = image.copy()
    fillContour(imagecopy, polygon, (200, 200, 200))
    for index in indices:
        cv2.circle(imagecopy, tuple(contour[index,0,:]), 3, (0, 0, 255), -1)
    cv2.imshow("Base polygon and corners", imagecopy)
    cv2.waitKey(0)

    # Return the sides
    return sides



#
#   Select the two contours to demo...
#
def selectContours(image):
    # Isolate the pieces: Convert to HSV and threshold.  The pieces
    # should show color (saturation >5%) or be dark (value <20%).
    # That is, the background (white) is both below 5% saturation and
    # above 20% value.  A simple threshold would probably also work.
    hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #binary = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 50), (255, 12, 255)))
    binary = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 50), (255, 15, 255)))

    # Grab all external contours - skip the internal (inner) holes.
    # Also do not approximate the contour, but report all pixels.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    # Down-select only contours big enough to hold pieces.
    contours = [contour for contour in contours if len(contour) > 2*SIDELEN]

    # Pick the two pieces to process.  Return the larger first.
    if len(contours[0]) > len(contours[1]):
        return (contours[0], contours[1])
    else:
        return (contours[1], contours[0])


#
#   Check the Translation/Orientation/Match between 2 Sides
#
def compareSides(sideA, sideB, center):
    # Grab the points from the two sides, relative to the center.
    M  = SIDEPOINTS
    iA = [int(round(j*(len(sideA)-1)/(M-1))) for j in range(M)]
    iB = [int(round(j*(len(sideB)-1)/(M-1))) for j in range(M-1,-1,-1)]
    pA = sideA[iA] - center
    pB = sideB[iB] - center

    # Pull out a list of the x/y coordinqtes.
    xA = pA[:,0].reshape((-1, 1))
    yA = pA[:,1].reshape((-1, 1))
    xB = pB[:,0].reshape((-1, 1))
    yB = pB[:,1].reshape((-1, 1))
    c0 = np.zeros((M,1))
    c1 = np.ones((M,1))

    # Build up the least squares problem for 4 parameters: dx, dy, cos, sin
    b  = np.hstack(( xA, yA)).reshape((-1,1))
    A1 = np.hstack(( c1, c0)).reshape((-1,1))
    A2 = np.hstack(( c0, c1)).reshape((-1,1))
    A3 = np.hstack((-yB, xB)).reshape((-1,1))
    A4 = np.hstack(( xB, yB)).reshape((-1,1))
    A  = np.hstack((A1, A2, A3, A4))

    param = np.linalg.pinv(A.transpose() @ A) @ (A.transpose() @ b)
    dtheta = np.arctan2(param[2][0], param[3][0])

    # Rebuild the least squares problem for 2 parameters: dx, dy
    b = b - A @ np.array([0, 0, np.sin(dtheta), np.cos(dtheta)]).reshape(-1,1)
    A = A[:, 0:2]

    param = np.linalg.pinv(A.transpose() @ A) @ (A.transpose() @ b)
    dx = param[0][0]
    dy = param[1][0]

    # Check the residual error.
    err = np.linalg.norm(b - A @ param) / np.sqrt(M)

    # Return the data.
    return (dx, dy, dtheta, err)
  


#
#   Main Code
#

# Grab the image.
image = cv2.imread("pieces1.jpg", flags=cv2.IMREAD_COLOR)

# Select counters (larger first).
(contourA, contourB) = selectContours(image)


# Show the starting contours.
imagecopy = image.copy()
drawContour(imagecopy, contourA, (0, 255, 255))
drawContour(imagecopy, contourB, (255, 255, 0))
cv2.imshow("The Original Contours", imagecopy)
cv2.waitKey(0)


# Process into the sides and centers.
sidesA = findSides(image, contourA)
sidesB = findSides(image, contourB)

centerB = centerContour(contourB)


# Draw the sides
drawSides(image, sidesA)
drawSides(image, sidesB)
cv2.circle(image, tuple(centerB.astype(int)), 5, (0, 255, 255), -1)
cv2.imshow("Sides on the Image", image)
cv2.waitKey(0)


# Compute the translation and rotation for all combinations
for iA in range(len(sidesA)):
    for iB in range(len(sidesB)):
        (dx, dy, dtheta, err) = compareSides(sidesA[iA], sidesB[iB], centerB)
        print(("Fixed sA[%d], rotating sB[%d]:  " +
               "dx %4d pixels, dy %4d pixels, " +
               "dtheta %4d deg, mean error %7.2f pixels") %
              (iA, iB, int(dx), int(dy), int(np.rad2deg(dtheta)), err))

        # CAUTION.  THE PIXEL COORDINATES ARE RIGHT/DOWN, so that
        # ROTATION ABOUT THE Z AXIS IS INTO THE PLANE (Z DOWN)!
        imagecopy = image.copy()
        c = np.cos(dtheta)
        s = np.sin(dtheta)
        for x,y in sidesB[iB].reshape(-1,2):
            x = x - centerB[0]
            y = y - centerB[1]
            xs = int(round(centerB[0] + dx + c*x - s*y))
            ys = int(round(centerB[1] + dy + s*x + c*y))
            imagecopy[ys,xs] = (255, 255, 255)
        cv2.imshow("Shifted side", imagecopy)
        cv2.waitKey(0)
