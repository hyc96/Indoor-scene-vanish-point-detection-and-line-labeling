import cv2
import numpy as np
import numpy.linalg as lin
import itertools
import matplotlib.pyplot as plt
import math

IM_RATIO = 1
INPUT_DIR = "input/"
OUTPUT_DIR = "output/"

# def load_image(fname):
#     im = Image.open(INPUT_DIR + fname)
#     new_w = im.size[0] // IM_RATIO
#     new_h = im.size[1] // IM_RATIO
#     im = im.resize((new_w, new_h), Image.ANTIALIAS)
#     # im.show()
#     return im
def load_image(fname):
    img = cv2.imread(INPUT_DIR + fname)
    return img
   
def normalize(line):
    '''
    returns the normalized c of a line segment
    '''
    # print(line)
    a, b, c = line
    normalize = np.sqrt(a ** 2 + b ** 2)
    return np.array([a/normalize, b/normalize, c/normalize])

def pt_to_line_dist(line, pt):
    a, b, c = line
    x, y = pt
    return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

def show_lines(img, lines, normalized = False):
    '''
    used for testing
    '''
    if normalized:
        for i in lines:
            x1,y1,x2,y2 = i[2]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.imwrite('pruned_lines.jpg',img)

    else:
        for i in lines:
            x1,y1,x2,y2 = i[0]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.imwrite('lines.jpg',img)

def draw_lines(lines, img, color):
    '''
    for showing line memberships
    '''
    for i in lines:
        x1,y1,x2,y2 = i[2]
        cv2.line(img,(x1,y1),(x2,y2),color,2)

def draw_points(votes, img):
    # print(len(votes))
    # plt.figure()
    # plt.imshow(img)
    for i in range(len(votes)):
        # print(votes[len(votes) - 1 - i])
        cv2.circle(img,(int(votes[i][0][0]),int(votes[i][0][1])),2,(255,0,255),-1)
        # cv2.circle(img,(299, 107),5,(255,0,255),-1)
        # plt.plot([int(votes[i][0][0]),int(votes[i][0][1])],'g')
    # plt.show()
    cv2.imwrite('points.jpg',img)

def pts_to_line(x1,y1,x2,y2):
    '''
    endpoints coord to tuple of:
    line equation ax+by+c = 0 (a,b,c)
    center coord
    endpoint coord
    angle
    '''
    pt1 = np.array([x1, y1, 1])
    pt2 = np.array([x2, y2, 1])
    line = np.cross(pt1, pt2)
    # line = line / line[-1] 
    # line = normalize(line)
    # h_lines = np.append(h_lines, line.reshape((3, 1)), axis=1)
    center = (pt1 + pt2) / 2
    # centers = np.append(centers, center.reshape((3, 1)), axis=1)
    if line[1] != 0: angle = np.arctan(-line[0] / line[1])
    else: angle = (np.pi/2)
    
    return (line, center, (x1,y1,x2,y2), angle)
