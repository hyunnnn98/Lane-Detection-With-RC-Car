################################################################################
######## START - 차선범위를 넘어간 이미지 예외처리 기능 ########################
################################################################################

# IMPORT NECESSARY LIBRARIES
import numpy as np

class LaneFrame:
    def __init__(self):
        # 밝기 값이 적용된 thresh img
        self.thresh = None
        self.minverse = None
        # 차선 라인 평균값
        self.draw_info = None
        # 차선 곡률반경
        self.curveRad = None
        self.curveDir = None
        
    def checkBackedImg(self):
        if (self.thresh is not None) :
            return True
        
        return False
    
    def loadFrameData(self):
        return self.thresh, self.minverse, self.draw_info, self.curveRad, self.curveDir
        
    def saveFrameData(self, thresh, minverse, draw_info, curveRad, curveDir):
        # 밝기 값이 적용된 thresh img
        self.thresh = thresh
        self.minverse = minverse
        # 차선 라인 평균값
        self.draw_info = draw_info
        # 차선 곡률반경
        self.curveRad = curveRad
        self.curveDir = curveDir

# 차선 인식 예외처리
def exception_handler(left_fitx, right_fitx, curveRad):
    
    right_fit_x_avg = int(np.mean(right_fitx))
    left_fit_x_avg = int(np.mean(left_fitx))
    # print((right_fit_x_avg - left_fit_x_avg) , left_fit_x_avg, right_fit_x_avg, curveRad)
    
    # 1. 예외 알고리즘 1 ) 오른쪽 mean - 왼쪽 mean == 250 정도.. ?
    # 2. 예외 알고리즘 2 ) 왼쪽은 100 ~ 300, 오른쪽은 1000 ~ 1200
    # 3. 곡률반경(=curveRad) > 3000
    overed_lane_detected = right_fit_x_avg - left_fit_x_avg > 450
    overed_lane_curveRad = curveRad > 2000
    left_lane_detected = left_fit_x_avg < 60 or left_fit_x_avg > 140
    right_lane_detected = right_fit_x_avg < 500 or right_fit_x_avg > 605
    
    # dev mode
    # print(overed_lane_detected, overed_lane_curveRad,
    #       left_lane_detected, right_fit_x_avg, right_lane_detected)
    
    # 에러가 감지 된 경우
    if left_lane_detected or right_lane_detected or overed_lane_detected or overed_lane_curveRad:
        return True
    
    return False
