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
        # return "백업 데이터 전송 테스트"
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
