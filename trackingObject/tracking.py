import argparse
import cv2 
import tkinter as toolkit
from PIL import Image, ImageTk

class Tracker():
    def __init__(self,margin = 30,method = cv2.TM_CCORR_NORMED):
        self.method = method
        self.margin = margin
        self.leftPoint = None
        self.RightPoint = None
        self.leftPointWithRegion = None
        self.RightPointWithRegion = None
        self.templateWithMargin = None
        self.template = None

    def getRegionOfTemplate(self,frame):
        top = max(0,self.leftPoint[1] -self.margin)
        bottom = min(frame.shape[0],self.RightPoint[1]+self.margin)
        left = max(0,self.leftPoint[0]-self.margin)
        right = min(frame.shape[1],self.RightPoint[0]+self.margin)
        self.templateWithMargin = frame[top:bottom,left:right]
        self.leftPointWithRegion = [left,top]
        self.RightPointWithRegion = [right,bottom]


    def track(self,frame):
        frameCopy = frame.copy()
        self.getRegionOfTemplate(frame)
        result = cv2.matchTemplate(self.templateWithMargin,self.template,self.method)
        minValue,maxValue,minLoc,maxLoc = cv2.minMaxLoc(result)
        self.leftPoint = [maxLoc[0]+self.leftPoint[0]-self.margin,maxLoc[1]+self.leftPoint[1]-self.margin]
        self.RightPoint = [self.leftPoint[0]+self.template.shape[1],self.leftPoint[1]+self.template.shape[0]]
        self.template = frameCopy[self.leftPoint[1]:self.RightPoint[1],self.leftPoint[0]:self.RightPoint[0]]

    def drawRegion(self,frame):
        cv2.rectangle(frame,self.leftPoint,self.RightPoint,(0,0,255),1)
        cv2.rectangle(frame,self.leftPointWithRegion,self.RightPointWithRegion,(0,255,0),2)
        return frame


    def setTemplate(self,points,template):
        self.leftPoint = points[0]
        self.RightPoint = points[1]
        self.template = template


class selector():
    def __init__(self, frame):
        self.frame = frame
        self.frameCopy = frame.copy()
        self.points = list()
        self.selected = False
    
    def mouseCallBack(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x,y])
        elif event == cv2.EVENT_LBUTTONUP:
            self.points.append([x,y])
            cv2.rectangle(self.frameCopy,self.points[0],self.points[1],(0,0,255),1)
            cv2.imshow("Region selector",self.frameCopy)
            self.confirmingWindow()
            self.selected = True
    
    def confirmingWindow(self):
        self.root = toolkit.Tk()
        self.root.title("ROI selector")

        cv2image = cv2.cvtColor(self.frameCopy,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgForTk = ImageTk.PhotoImage(image=img)

        panel = toolkit.Label(self.root,image=imgForTk)
        panel.pack(padx=10,pady=10)
        continueButton = toolkit.Button(self.root,text="Continue",command=self.continueButtonPressed)
        continueButton.pack(side=toolkit.LEFT,padx=10,pady=10)

        retryButton = toolkit.Button(self.root,text="Retry",command=self.retryButtonPressed)
        retryButton.pack(side=toolkit.LEFT,padx=10,pady=10)
        
        self.root.mainloop()

    def continueButtonPressed(self):
        self.root.destroy()

    def retryButtonPressed(self):
        self.root.destroy()
        self.frameCopy = self.frame.copy()
        self.points = list()
        self.selected = False
        self.showFrameFromSelect()

    def showFrameFromSelect(self):
        cv2.namedWindow("Region selector")
        cv2.setMouseCallback("Region selector",self.mouseCallBack)
        while True:
            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("Region selector",self.frameCopy)
            if key == ord("e") or self.selected == True:
                break
        cv2.destroyWindow("Region selector")

    def getROI(self):
        return self.points,self.frame[self.points[0][1]:self.points[1][1],self.points[0][0]:self.points[1][0]]






def main(source = 0 ):
  
    capture = cv2.VideoCapture(source)
    value , frame = capture.read()
    if not value:
        print("Error while opening video from source")
    selectorROI = selector(frame)
    selectorROI.showFrameFromSelect()
    points,template = selectorROI.getROI()
    objectTracker = Tracker()
    objectTracker.setTemplate(points,template)
    while True:
        retValue ,frame = capture.read()
        if not retValue:
            break
        objectTracker.track(frame)
        frame = objectTracker.drawRegion(frame)
        cv2.imshow('video',frame)
        if cv2.waitKey(25) &0xFF == ord ('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking Object from video")
    parser.add_argument("--source",default=0,help="path to videofile")
    args = parser.parse_args()
    main(args.source)
