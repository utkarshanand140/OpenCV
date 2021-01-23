import numpy as np
import cv2

cap = cv2.VideoCapture('c:/Users/SHASTA/Desktop/challenge_video.mp4')
##fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', 0x7634706d, 20.0, (640,480))




def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    draw_lines(line_img, lines[:,:])
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    
    return cv2.addWeighted(initial_img, α, img, β, λ)
try:
    while (True):
        success,img = cap.read()
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100],dtype = 'uint8')
        upper_yellow = np.array([30, 225, 225],dtype = 'uint8')
        mask_yellow = cv2.inRange(imgHSV, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(imgGray, 200, 255)
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        mask_yw_image = cv2.bitwise_and(imgGray, mask_yw)

        kernel = 5
        imgGauss = cv2.GaussianBlur(mask_yw_image,(7,7),5)   

        low_threshold = 50
        high_threshold = 150
        imgCanny = cv2.Canny(imgGauss,low_threshold,high_threshold)

        

        
        roi = imgCanny[500:,285:900]    ##CHANGE
        
        temp = img[400:,285:900]                ##CHANGE
        rho = 2
        theta = np.pi/180
        
        threshold = 20
        min_line_len = 50
        max_line_gap = 400

        line = hough_lines(roi, rho, theta, threshold, min_line_len, max_line_gap)
        result = weighted_img(line, img[500:,285:900], α=0.8, β=1., λ=0.) ## CHANGE

        if success==True:
            out.write(temp)
        cv2.imshow("Video",result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except cv2.error as e:
    s = str(e)


'''cap.release()
out.release()
cv2.destroyAllWindows()
'''
