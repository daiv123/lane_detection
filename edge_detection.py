import cv2
import numpy as np
import matplotlib.pyplot as pyplot

def polynomial(x, p) :
    return p[0] * x * x + p[1] * x + p[2]

# calculate gradient in the x direction, non destructive
# input: image
# output: image_gradient scaled from 0-255
def sobel_gradient_x(image) :
    img = image[:]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return scaled_sobel

# perform vertical perspective transformation 
# visualize: draw the window that will be trasnformed
# input: image: image
#        k: ratio between top and bottom border length
# output: transformed image
def perspective_transform(image, k, visualize = False) :
    h, w = np.shape(image)[:2]
    tw = w * k
    toffset = int((w-tw)/2)
    pts =  ((toffset,1), (w-toffset-1, 1), (w-1, h-1), (1,h-1))
    # pts = ((100,100),(2000,100),(3000,600),(50,600))
    pts = np.array(pts, dtype = "float32")
    print(pts)
    if visualize :
        return visualize_lanes(image, [pts])

    return four_point_transform(image, pts)

# NOT IN USE helper function for four point transform
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

#perform a four point perspective transformation
#input: image: image
#       pts: four points oredered tl, tr, br, bl
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
    # rect = order_points(pts)
    rect = pts
    print(rect)
    (tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

# perform canny edge detection
def edge_detection(image) :
    if len(image[0]) == 3 :
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else :
        image_gray = np.copy(image)
    image_smooth = cv2.GaussianBlur(image_gray, (5,5), 0)
    image_canny = cv2.Canny(image_smooth, 50, 100)
    return image_canny

def hard_threshold(image, threshold) :
    img_thresh = np.zeros_like(image)
    img_thresh[(image > gradient_thresh[0]) & (image < gradient_thresh[1])] = 255
    return img_thresh

def adaptive_threshold(image) :
    image_smooth = cv2.GaussianBlur(image, (5,5), 0)
    image_threshold = cv2.adaptiveThreshold(image_smooth,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return cv2.bitwise_not(image_threshold)
def color_mask(image) :
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(image_gray, 220, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    image_yw = cv2.bitwise_and(image_gray, mask_yw)
    return image_yw

def line_detection(image) :
    hough = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength = 400, maxLineGap= 500)
    hough = np.reshape(hough, (len(hough),4))
    return hough

def lane_detection(lines, maxGap = 50) :
    if len(lines) == 1 :
        return lines[0]
    lane = []
    lane.append(lines[0][:2])
    lane.append(lines[0][2:])
    next_lines = []
    finished = False
    while not finished :
        finished = True
        print(len(lines))
        for line in lines[1:] :
            # print(line)
            pt1 = line[:2]
            pt2 = line[2:]
            for i, pt in enumerate(lane):
                if dist(pt, pt1) < maxGap :
                    pt = (pt + pt1)/2
                    lane[i] = pt
                    lane.insert(i+1,pt2)
                    finished = False
                    break

                elif dist(pt, pt2) < maxGap :
                    pt = (pt + pt2)/2
                    lane[i] = pt
                    lane.insert(i, pt1)
                    finished = False
                    break
        lines = next_lines
    lane = np.array(lane)
    return lane.astype(int)

def poly_detection(image) :

    indicies = np.where(image == [255])
    # print(indicies)
    # coordinates = zip(indices[0], indices[1])
    p = np.polyfit(indicies[1], indicies[0], 2)
    print(p)
    return p

def poly_detection_sliding_box(image, starting_pos, box_dim = (1000, 400), visualize = False, frame = None) :
    box_width = box_dim[0]
    box_height = box_dim[1]
    #gives bounding points based on center of bottom of the box
    def box_cords(pos, width, height):
        x1 = pos[0] - (width//2)
        x2 = pos[0] + (width//2)
        y1 = pos[1] - height
        y2 = pos[1]
        return x1, y1, x2, y2

    curPos = np.copy(starting_pos)

    #shut up leave me alone :( i dont know how to do this
    left_ind = []
    left_ind.append([])
    left_ind.append([])

    #while box in frame
    while curPos[1] > box_height :
        #establish box
        x1, y1, x2, y2 = box_cords(curPos, box_width, box_height)
        if x1 < 0 :
            x1 = 0
        if x2 > len(image[0]) :
            x2 = len(image[0])
        print(x1, y1, x2, y2)

        #find points in box
        new_ind = np.where(image[y1:y2, x1:x2] == [255])
        # print(image[y1:y2][x1:x2])
        # print(new_ind)
        left_ind[0].extend(new_ind[1]+x1)
        left_ind[1].extend(new_ind[0]+y1)

        #recalculate and find next pos
        p = np.polyfit(left_ind[1], left_ind[0], 2) #f(y) = x
        curPos = (int(polynomial(curPos[1] - box_height, p)),curPos[1] - box_height)

        if visualize :
            p = np.polyfit(left_ind[0], left_ind[1], 2)
            visualize_poly(frame, p, copy=False, color = (255, int(y1/len(image) * 255), int(y1/len(image) * 255)))
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0))
    

    # indicies = np.where(image == [255])
    # print(indicies)
    # coordinates = zip(indices[0], indices[1])
    p = np.polyfit(left_ind[0], left_ind[1], 2)
    print(p)
    return p

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.copy(frame)
    # Checks if any lines are detected
    if lines is not None:
        for (x1, y1, x2, y2) in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

def visualize_lanes(frame, lanes):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.copy(frame)
    # Checks if any lines are detected
    if lanes is not None:
        for lane in lanes :
            print(lane)
            pts = lane.reshape((-1,1,2))
            cv2.polylines(lines_visualize, [pts], False, (0,0,255), thickness = 5)

    return lines_visualize

def visualize_poly(frame, p, copy = True, color = (0,0,255)):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    if copy :
        lines_visualize = np.copy(frame)
    else: 
        lines_visualize = frame
    # Checks if any lines are detected
    pts = []
    for x in range(len(frame[0])):
        y = polynomial(x, p)
        if y >=0 and y < len(frame):
            pts.append((x,int(y)))
    
    pts = np.array(pts)
    
    pts = pts.reshape((-1,1,2))
    cv2.polylines(lines_visualize, [pts], False, color, thickness = 5)

    return lines_visualize

def dist(pt1, pt2) :
    return np.linalg.norm(pt1-pt2)



def main() :
    img = cv2.imread("images/car2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gradient = sobel_gradient_x(img)
    gradient_thresh = (10,255)
    img_gradient_thresh = np.zeros_like(img_gradient)
    img_gradient_thresh[(img_gradient > gradient_thresh[0]) & (img_gradient < gradient_thresh[1])] = 255
    cv2.imshow("grad_thesh", img_gradient_thresh)



    # img_warped = perspective_transform(img, 0.4, visualize=False)
    # cv2.imshow('warped', img_warped)

    # img_yw = color_mask(img_warped)
    # img_edge = edge_detection(img_yw)

    # print("image processesd")
    # box = (500, 200)
    # start = (len(img[0])//4, len(img))
    # img_poly = np.copy(img)
    # p = poly_detection_sliding_box(img_edge, start,box_dim=box, visualize=True,frame=img_poly)
    # img_poly = visualize_poly(img_poly, p, copy=False)
    # # p = poly_detection_sliding_box(img_edge, (len(img[0])//4*3, len(img)), box_dim=box, visualize=True,frame=img_poly)
    # # img_poly = visualize_poly(img_poly, p, copy=False)


    # cv2.imshow('image',img_poly)
    # cv2.imshow('edges', img_edge)
    # cv2.imshow('mask', img_yw)
    # img = perspective_transform(img, 0.4, visualize=False)

    # img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # cv2.GaussianBlur(img_hls, (5,5), 0)
    # img_s = img_hls[:,:,2]
    # img_l = img_hls[:,:,1]

    # s_thresh = (50,255)
    # s_binary = np.zeros_like(img_s)
    # s_binary[(img_s >= s_thresh[0]) & (img_s <= s_thresh[1])] = 255
    # cv2.imshow("bin", s_binary)

    # l_thresh = (230,255)
    # l_binary = np.zeros_like(img_l)
    # l_binary[(img_l >= l_thresh[0]) & (img_l <= l_thresh[1])] = 255
    # cv2.imshow("lbin", l_binary)


    # img_poly = np.copy(img)
    # p = poly_detection_sliding_box(l_binary, (3000, len(img)),box_dim=(1000,00), visualize=True,frame=img)
    # img_poly = visualize_poly(img_poly,p)
    # cv2.imshow('poly', img_poly)
    # cv2.imshow('image', img)
    # cv2.imshow('image hls', img_hls)
    # cv2.imshow('image s', img_s)
    # cv2.imshow('image l', img_l)


    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()