import cv2
from PIL import Image
import numpy as np
import numpy.linalg as lin
import itertools
import matplotlib.pyplot as plt
import math
from util import *

"""
Vanishing point detection
"""

SIGMA = 0.1 # vanishing point score parameter


def Hough(img, minLineLength = 10, maxLineGap = 10):
    '''
    Hough transform for line detection
    '''
    # blur = cv2.bilateralFilter(img,9,75,75)
    # gray_img = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img,50,200,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180/4,100,minLineLength,maxLineGap) 
    # print(lines)    
    # show_lines(img, lines)
    return lines

def vanish_pts(lines):
    # get all vanishing point candidates
    pairs = range(0,len(lines))
    idx = list(itertools.combinations(pairs,2))
    V = []
    for x in idx:
        i, j = x
        pt = np.cross(lines[i][0],lines[j][0])
        # if lin.norm(pt) < 5:
        # print(lines[i][2])
        # print(lines[j][2])
        # print(pt)
        # print('-----')
        if pt[-1] == 0: # indicate a infinite vanishing point
            a1, b1, c1 = lines[i][0]
            a2, b2, c2 = lines[j][0]
            normalize1 = np.sqrt(a1** 2 + b1 ** 2)
            normalize2 = np.sqrt(a2** 2 + b2 ** 2)
            co = 20000
            if b1 == 0: pt = np.array([-(c1 + c2)/2/a1 ,a1/normalize1 * co,1])
            else: 
                pt = np.array([-b1/normalize1 * co, a1/normalize1 * co - (c1/b1 + c2/b2)/2, 1])
                # print(lines[i][0])
                # print(lines[j][0])
                # print(c1/b1/normalize1)
                # print(pt)
            pt = pt.astype(int)
            V.append(pt)
        else:
            pt = pt/pt[-1]
            pt = pt.astype(int)
            # print(pt)
            V.append(pt)
    
    #vote/rank candidates
    votes = []
    for pt in V:
        score = 0
        for i in range(len(lines)):
            theta = lines[i][-1]
            v1 = np.array([lines[i][1][0] - pt[0], lines[i][1][1] - pt[1]])
            x1,y1,x2,y2 = lines[i][2]
            v2 = np.array([x2 - x1, y2 - y1])
            # print("here")
            if lin.norm(v1) == 0:
                val = 1
            else: val = v1.dot(v2) / lin.norm(v1) / lin.norm(v2)
            val = round(val, 6)
            # if abs(val) == 1 or val == 0: val = int(val)
            alpha = math.acos(val)
            # print(val)
            if alpha > np.pi/2: alpha = np.pi - alpha
            # print(pt, lines[i][2], alpha)
            score += lin.norm(v2) * np.exp(-abs(alpha) / 2 / SIGMA ** 2) 
        votes.append((pt,score))
    
    votes = sorted(votes, key=lambda tup: tup[-1], reverse=True)
    return votes

def process_lines(lines, merge = True):
    '''
    h_lines is the list of tuple (ax+by+c line coordinates, line center, line endpoints, line angle)
    '''
    h_lines = []
    # change line representations
    for i in lines:
        x1,y1,x2,y2 = i[0]
        h_lines.append(pts_to_line(x1,y1,x2,y2))

    # sort lines based on angle
    h_lines = sorted(h_lines, key=lambda tup: tup[-1], reverse=True)

    # merge similar lines
    if merge:
        print("Total detected line segments: ",len(h_lines))
        h_lines = merge_lines(h_lines)
        print("Merged line segments: ", len(h_lines))
    return h_lines

def merge_lines(lines, line_threshold=2, ang_threshold=5, gap_threshold=50):
    '''
    line_threshold: difference in c to merge two lines
    '''
    # high precision clustering to weed out similar lines
    ret = []
    assigned = set()
    for i in range(len(lines)):
        if i in assigned: continue
        line1 = lines[i]
        temp = [line1]
        for j in range(i+1, len(lines)):
            line2 = lines[j]
            if abs(line1[-1] - line2[-1]) < ang_threshold/180*np.pi or \
               abs(line1[-1] - line2[-1]) > np.pi - ang_threshold/180*np.pi:
                x1, y1, x2, y2 = line2[2]
                dist1 = pt_to_line_dist(line1[0],(x1,y1)) 
                dist2 = pt_to_line_dist(line1[0],(x2,y2))
                if dist1 <= line_threshold and dist2 <= line_threshold:
                    if i not in assigned: 
                        temp = [i,j]
                        assigned.add(i)
                        assigned.add(j)
                    else:
                        temp.append(j)
                        assigned.add(j)
        if i not in assigned:
            temp = [i]

        idx = [lines[k][2] for k in temp]
        idx = sorted(idx, key=lambda tup: tup[0], reverse=False)
        curr = idx[0]
        done = False
        for k in range(1,len(idx)):
            x1, y1, x2, y2 = curr
            x3, y3, x4, y4 = idx[k]
            if x2 >= x3 or lin.norm([x3-x2, y3-y2]) <= gap_threshold:
                curr = (x1, y1, x4, y4)
                done = False
            else: 
                ret.append(pts_to_line(*curr))
                curr = x3, y3, x4, y4
                done = True
        if not done: ret.append(pts_to_line(*curr))

    ret = sorted(ret, key=lambda tup: tup[-1], reverse=True)
    return ret

def cluster(votes, max_dist = 50, threshold = 5000):
    '''
    max_dist: maximum distance for clustering
    '''
    def assign(i, assigned, temp):
        if i not in assigned:
            temp = [i,j]
            assigned.add(i)
            assigned.add(j)
        else:
            temp.append(j)
            assigned.add(j)
    # print("Total candidate points: ", len(votes))

    # cluster close candidate points 
    # since votes are sorted, only cluster with one level of hierarchy
    cluster = []
    assigned = set()
    temp = []
    for i in range(len(votes)):
        if i in assigned: continue
        # if votes[i][0][-1] == 0: print(votes[i][0])
        pt1 = votes[i][0]# / votes[i][0][-1]
        for j in range(i+1, len(votes)):
            pt2 = votes[j][0]# / votes[j][0][-1]
            dist = lin.norm(pt1 - pt2)
            if dist <= max_dist:
                assign(i, assigned, temp)

            #horizontal infinite case
            elif abs(pt1[0]) >= threshold and pt1[1] == 0:
                if abs(pt2[0]) >= threshold and pt2[1] == 0: 
                    assign(i, assigned, temp)
                elif pt2[1] == 0: pass
                elif abs(pt2[0] /pt2[1]) >= threshold / 100:
                    assign(i, assigned, temp)
            elif abs(pt1[0] /pt1[1]) >= threshold / 100:
                if abs(pt2[0]) >= threshold and pt2[1] == 0: 
                    assign(i, assigned, temp)
                elif pt2[1] == 0: pass
                elif abs(pt2[0] /pt2[1]) >= threshold / 100:
                    assign(i, assigned, temp)

            #vertical infinite case:
            elif abs(pt1[1]) >= threshold and pt1[0] == 0:
                if abs(pt2[1]) >= threshold and pt2[0] == 0: 
                    assign(i, assigned, temp)
                elif pt2[0] == 0: pass
                elif abs(pt2[1] /pt2[0]) >= threshold / 100:
                    assign(i, assigned, temp)
            elif abs(pt1[1] /pt1[0]) >= threshold / 100:
                if abs(pt2[1]) >= threshold and pt2[0] == 0: 
                    assign(i, assigned, temp)
                elif pt2[0] == 0: pass
                elif abs(pt2[1] /pt2[0]) >= threshold / 100:
                    assign(i, assigned, temp)

        if i not in assigned:
            temp = [i]
        cluster.append([votes[i] for i in temp])
    # print("Clustered points: ", len(cluster))

    # generate average points for each cluster, combine scores
    ret = []
    scores = []
    # print(cluster[-1])
    for i in cluster:
        avg_pt = np.array([0,0,0])
        total_score = 0
        for j in i:
            pt, score = j
            # if same infinite vanish point intersected at opposite directions:
            if abs(pt[0]) >= threshold and pt[1] == 0:
                if pt[0] < 0: pt[0] = -pt[0]
            elif pt[1] == 0: pass
            elif abs(pt[0] /pt[1]) >= threshold / 100:
                if pt[0] < 0: pt[0] = -pt[0]
            if abs(pt[1]) >= threshold and pt[0] == 0:
                if pt[1] < 0: pt[1] = -pt[1]
            elif pt[0] == 0: pass
            elif abs(pt[1] /pt[0]) >= threshold / 100:
                if pt[1] < 0: pt[1] = -pt[1]

            if abs(pt[0]) > threshold and abs(pt[1]) > threshold:
                if pt[0] > 0 and pt[1] < 0: pass
                if pt[0] < 0 and pt[1] < 0 or pt[0] < 0 and pt[1] > 0: 
                    pt[0] = -pt[0]
                    pt[1] = -pt[1]
            avg_pt = avg_pt + pt * score
            total_score += score
            scores.append(score)

        ret.append((avg_pt/total_score,max(scores)))
    ret = sorted(ret, key=lambda tup: tup[-1], reverse=True)
    return ret

def remove_lines(v, lines, threshold1 = 7000, threshold2 = 5):
    '''
    threshold1: threshold for identifying infinite vanishing points
    threshold2: max difference in degrees to remove a line from a cluster
    '''
    ret = []
    member = []
    #horizontal infinite points
    if abs(v[0]) >= threshold1 and v[1] == 0:
        for i in lines:
            if abs(i[-1]) > threshold2/180*np.pi:
                ret.append(i)
            else: member.append(i)
    elif abs(v[0] / v[1]) >= threshold1 / 100:
        for i in lines:
            if abs(i[-1]) > threshold2/180*np.pi:
                ret.append(i)
            else: member.append(i)
    #vertical infinite points
    elif abs(v[1]) >= threshold1 and v[1] == 0:
        for i in lines:
            if abs(abs(i[-1]) - np.pi/2) > threshold2/180*np.pi:
                ret.append(i)
            else: member.append(i)
    elif abs(v[1] / v[0]) >= threshold1 / 100:
        for i in lines:
            if abs(abs(i[-1]) - np.pi/2) > threshold2/180*np.pi:
                ret.append(i)
            else: member.append(i)

    else:
        for i in range(len(lines)):
            # print("here")
            v1 = np.array([lines[i][1][0] - v[0], lines[i][1][1] - v[1]])
            x1,y1,x2,y2 = lines[i][2]
            v2 = np.array([x2 - x1, y2 - y1])
            if lin.norm(v1) == 0:
                val = 1
            else: val = v1.dot(v2) / lin.norm(v1) / lin.norm(v2)
            val = round(val, 6)
            # if abs(val) == 1 or val == 0: val = int(val)
            alpha = math.acos(val)
            if alpha > np.pi/2 : alpha = np.pi - alpha
            # except: alpha = np.arccos(np.clip(val,a_min = -0.99999, a_max = 0.99999))
            # print(alpha, abs(np.pi - alpha))
            if alpha > threshold2/180*np.pi and abs(np.pi - alpha) > threshold2/180*np.pi:
                # print("here2")
                ret.append(lines[i])
            else: member.append(lines[i])
    ret = sorted(ret, key=lambda tup: tup[-1], reverse=True)
    return ret, member

def orthogonal(v1, member1, v2, member2, v3, member3, threshold = 7000, threshold1 = 5, threshold3 = 300):
    '''
    determine if candiate triplet satisfies orthogonality
    '''
    V = [v1,v2,v3]
    inf = [(abs(v1) > threshold).any(), (abs(v2) > threshold).any(), (abs(v3) > threshold).any()]
    num_inf = int((abs(v1) > threshold).any()) + int((abs(v2) > threshold).any()) + int((abs(v3) > threshold).any())
    if num_inf == 3: 
        return False
        
    if num_inf == 2:
        idx = np.where(inf)[0]
        a1, a2 = V[idx[0]], V[idx[1]]
        val = a1.dot(a2) / lin.norm(a1) / lin.norm(a2)
        val = round(val, 6)
        # if abs(val) == 1 or val == 0: val = int(val)
        alpha = math.acos(val)
        if abs(alpha - np.pi/2) < threshold1 / 180 * np.pi:
            # print(alpha)
            return True

    if num_inf == 1:
        idx = np.where(inf)[0]
        a3 = V[idx[0]]
        l_idx = [0,1,2]
        l_idx.remove(idx.item())
        a1, a2 = V[l_idx[0]], V[l_idx[1]]
        diff = a1 - a2
        if lin.norm([diff[0],diff[1]]) < threshold3: return False #too close
        v_line = np.cross(a1, a2)
        angle = np.arctan(-v_line[0] / v_line[1])
        # print(angle)
        # print(v_line)
        if abs(a3[0]) >= threshold and a3[-1] == 0 or abs(a3[0] /a3[1]) >= threshold / 100:
            if abs(angle) < threshold1/180*np.pi:
                return True
        
        elif abs(a3[1]) >= threshold and a3[-1] == 0 or abs(a3[1] /a3[0]) >= threshold / 100:
            if abs(angle - np.pi/2) < threshold1/180*np.pi:
                return True

    return False

def detect(img, fname, draw_member = True):
    '''
    main function of v_detect
    '''
    lines = Hough(img)
    lines = process_lines(lines, merge = True)

    #search for 3 orthogonal vanish points
    votes = vanish_pts(lines)
    cluster1 = cluster(votes)
    flag = False
    for i in cluster1:
        if flag: break
        v1 = i[0]
        lines1, member1 = remove_lines(v1,lines)
        votes1 = vanish_pts(lines1)
        cluster2 = cluster(votes1)
        for j in cluster2:
            if flag: break
            v2 = j[0]
            lines2, member2 = remove_lines(v2,lines1)
            votes2 = vanish_pts(lines2)
            cluster3 = cluster(votes2)
            for k in cluster3:
                v3 = k[0]
                outlier, member3 = remove_lines(v3,lines2)
                if orthogonal(v1,member1,v2,member2,v3,member3): 
                    flag = True
                    break
    if draw_member:
        draw_lines(member1, img, (255,0,0))
        draw_lines(member2, img, (0,255,0))
        draw_lines(member3, img, (0,0,255))
        draw_lines(outlier, img, (0,128,255))
        # cv2.circle(img,(297, 103),3,(255,0,255),-1)
        cv2.imwrite(OUTPUT_DIR + "membership_" + fname,img)

    print("Vanishing points found:") 
    print(v1,v2,v3) 
    return [v1, v2, v3]