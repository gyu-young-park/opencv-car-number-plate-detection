import cv2
import numpy as np

class Transformer:
    APPROXIMATION = "approx"
    HULL = "hull"

    def __init__(self):
        self.original = None
        self.original_gray = None
        self.currentImage = None
        self.currentFilename = None

        # Normalizing

        # Smooth param
        self.blurrSize = 3

        # Canny Edging
        self.autoDetect = True  # if true, automatically calculate threshold
        self.autoSigma = 0.33  # sigma used in auto detect
        self.cannyMin = 30
        self.cannyMax = 200
        self.apertureSize = 3
        self.L2 = False

        # Morphing param
        self.morphFunction = cv2.morphologyEx
        self.morphType = cv2.MORPH_OPEN
        self.morphKernelSize = 7
        self.SEKernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morphKernelSize, self.morphKernelSize))

        # line param
        self.rho = 1
        self.lineThresh = 130
        self.theta = 1*np.pi/180
        self.minLineLength = 100
        self.maxLineGap = 30
        self.lineThickness = 2
        # contour param
        self.extractMethod = Transformer.APPROXIMATION
        self.epsilonConstant = 0.018
        self.minWidth = 6
        self.minRatio = 0.5
        self.minHeight = self.minWidth * self.minRatio
        self.areaThresh = 30
        self.approxThresh = 50  # Higher the more easier
        #new
        self.contourFind = cv2.RETR_EXTERNAL
        #new
        self.contourExpress = cv2.CHAIN_APPROX_NONE

        # Corner detector
        self.blockSize = 2
        self.kernelSize = 3
        self.freeParam = 0.04

        # window param
        self.winname = "window1"
        self.winnameCar = "window2"

    def show(self, image, winname, image2, winname2):
        cv2.imshow(winname, image)
        cv2.imshow(winname2, image2)

    def smooth(self, image):
        return cv2.GaussianBlur(image, (self.blurrSize, self.blurrSize), 0)

    def toBinary(self, image):
        _, image = cv2.threshold(   
            image, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        return image

    def detectEdge(self, image):
        if not self.autoDetect:
            return cv2.Canny(image,100,200)
        elif self.autoDetect:
            v = np.mean(image)
            lower = int(max(0, (1.0 - self.autoSigma) * v))
            upper = int(min(255, (1.0 + self.autoSigma) * v))
            return cv2.Canny(image,lower,upper)

    def morph(self, image):
        morphed = None
        kernel = np.ones((3,3),np.uint8)
        #tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        #blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        #return blackhat


    def lines(self, image, todraw):
        lines = cv2.HoughLinesP(image, rho=self.rho, theta=self.theta, threshold=self.lineThresh,
                                minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
        if lines is not None:
            for line in lines:
                line = line[0]
                ###여기까
                if(line[2] - line[0] > 100 and line[3] - line[1] > 100 ):
                    print(line)
                    
                    cv2.line(image, (line[0], line[1]), (line[2],
                                                          line[3]), (255, 255, 255), self.lineThickness)
        return image

    def detectSURF(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        #surf = cv2.xfeatures2d_SURF()
        kp = sift.detect(image, None)
        return cv2.drawKeypoints(image, kp, image)

    def getBlank(self, size):
        return np.zeros(size)

    def featureExtract(self, image, todraw):
        #new
        box1=[]
        f_count=0
        select=0
        plate_width=0
        _,contours, _ = cv2.findContours(
            image, self.contourFind, self.contourExpress)
        #new
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        #for i in range(len(contours)):
        #       cnt=contours[i]          
        #       area = cv2.contourArea(cnt)
        #       x,y,w,h = cv2.boundingRect(cnt)
        #       rect_area=w*h  #area size
        #       aspect_ratio = float(w)/h # ratio = width/height
                  
        #       if  (aspect_ratio>=0.2)and(aspect_ratio<=1.0)and(rect_area>=500)and(rect_area<=1000): 
        #            cv2.rectangle(todraw,(x,y),(x+w,y+h),(0,255,0),1)
        #            box1.append(cv2.boundingRect(cnt))
        for contour in contours:
            area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour)
            rect_area=w*h  #area size
            aspect_ratio = float(w)/h # ratio = width/height
            if (aspect_ratio>=0.2)and(aspect_ratio<=1.0)and(rect_area>=200)and(rect_area<=5500): 
                epsilon = self.epsilonConstant*cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) <= self.approxThresh:
                    if self.extractMethod == Transformer.APPROXIMATION:
                        rect = cv2.minAreaRect(contour)
                        if rect[2] > 20 or rect[2] < -100:
                            continue
                        if rect[1][0] < self.minWidth or rect[1][1] < self.minHeight:
                            continue
                        # Gets rotated rectangle
                        box1.append(cv2.boundingRect(contour))
                        box = cv2.boxPoints(rect)
                        box_d = np.int0(box)
                        cv2.drawContours(
                            todraw, [box_d], 0, (0, 255, 0), self.lineThickness)
                    elif self.extractMethod == Transformer.HULL:
                        box1.append(cv2.boundingRect(contour))
                        area = cv2.contourArea(contour)
                        if area > self.areaThresh:
                            hull = cv2.convexHull(contour)
                            cv2.polylines(todraw, [hull], True,(0, 255, 0), self.lineThickness)
                
        for i in range(len(box1)): ##Buble Sort on python
               for j in range(len(box1)-(i+1)):
                    if box1[j][0]>box1[j+1][0]:
                         temp=box1[j]
                         box1[j]=box1[j+1]
                         box1[j+1]=temp
                         
        for m in range(len(box1)):
               count=0
               for n in range(m+1,(len(box1)-1)):
                    delta_x=abs(box1[n+1][0]-box1[m][0])
                    if delta_x > box1[m][2]*9:
                         break
                    delta_y =abs(box1[n+1][1]-box1[m][1])
                    if delta_x ==0:
                         delta_x=1
                    if delta_y ==0:
                         delta_y=1           
                    gradient =float(delta_y) /float(delta_x)
                    if gradient<0.25:
                        count=count+1
               #measure number plate size         
               if count > f_count:
                    select = m
                    f_count = count;
                    plate_width=delta_x
        xList = []
        yList = []
        boxs = []
        box1 = sorted(box1)
        #x,y,w,h 
        #혹시 모르니 if에 추가하자 and box1[j][0] <=box1[select][0] + box1[select][2]*9 
        for j in range(len(box1)):
            #coordinateList.append(box1[m+j])
            #if( box1[select][3] - 20 <= box1[j][3] and box1[j][3]<=box1[select][3] + 20 and box1[select][1] - 40 <= box1[j][1] and box1[j][1]<=box1[select][1] + 40):   
            if(box1[select][0] - box1[select][2]<= box1[j][0] and box1[j][0] <= box1[select][0] + box1[select][2]*9 and box1[select][1] - box1[select][3] <= box1[j][1] and box1[j][1]<=box1[select][1] + box1[select][3]):
                xList.append(box1[j][0])
                yList.append(box1[j][1])
                boxs.append(box1[j])
        try:    
            boxs = sorted(boxs)
            minwidth = min(xList)
            maxWidth = max(xList)
            
            maxy = 0
            for i in range(len(boxs)):
                if boxs[i][0] == maxWidth:
                    maxy = boxs[i]
            maxheight = max(yList)
            minheight = min(yList)
            #number_plate=todraw[box1[select][1]-10:box1[select][3]+box1[select][1]+20,box1[select][0]-10:140+box1[select][0]]
            arr = []
            #arr.append([minwidth,minheight -20])
            #arr.append([minwidth,maxheight + box1[select][3]])
            #arr.append([maxWidth + box1[select][2],maxheight + box1[select][3]])
            #arr.append([maxWidth + box1[select][2],minheight -20])
            arr.append([minwidth-box1[select][2]-20,box1[select][1] - 20])
            arr.append([minwidth-box1[select][2]-20,box1[select][1] + box1[select][3] + 20])
            arr.append([maxWidth + box1[select][2]+20,maxy[1] + maxy[3] + 20])
            arr.append([maxWidth + box1[select][2]+20,maxy[1] - 20])
            
            number_plate=todraw[minheight -20 :maxheight + box1[select][3],minwidth-box1[select][2]:maxWidth + box1[select][2]]
            x, y = number_plate.shape
            print(x)
            print(y)
            if float(y)/x < 1.2:
                text = "Nothing"
                img = np.zeros((200,400,3),np.uint8)
                cv2.putText(img, text,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
                print("번호판을 찾지 못하였습니다.")
                return img
            else:
                resize_plate=cv2.resize(number_plate,None,fx=1.8,fy=1.8,interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
                ret,th_plate = cv2.threshold(resize_plate,150,255,cv2.THRESH_BINARY)
                output = self.wrap(arr,self.original)
                #self.show(self.original,self.winname,output,self.winnameCar)  # , rectangles
                return output
        except:
            text = "error"
            img = np.zeros((200,400,3),np.uint8)
            cv2.putText(img, text,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
            print("특징점 추출에 실패하였습니다. 이미지의 명도가 너무 높거나 너무 낮은 경우, 또는 자동차 번호판의 인식이 불분명한 경우에 해당됩니다.")
            return img

    def corner(self, image, todraw):
        dst = cv2.cornerHarris(image, self.blockSize,
                               self.kernelSize, self.freeParam)
        todraw[dst > 0.01 * dst.max()] = [0, 0, 255]
        return todraw

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def normalize(self, image):
        return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    def wrap(self,arr,image):
        rows,columns = 300,400
        print(arr)
        point1 = np.float32([arr[0],arr[3],arr[1],arr[2]])
        point2 = np.float32([[0,0],[columns,0],[0,rows],[columns,rows]])
        
        P =  cv2.getPerspectiveTransform(point1,point2)
        return cv2.warpPerspective(image,P,(columns,rows))
    
    def makeImageBright(self,image):
        return cv2.equalizeHist(image)
    
    def bitAnd(self, thresh, morph):
        return cv2.bitwise_and(thresh,thresh,mask = morph)


# New
"""
Manually find lefttop righttop leftbottom rightbottom
"""
n=1
cont = True
if __name__ == '__main__':

    while(cont):
        try:
            try:
                filename = input("Enter the FileName: ")
                t = Transformer()
                t.currentImage = cv2.imread(filename)
                t.currentImage = t.resize(t.currentImage, width=1200,height=900)
                t.original = t.currentImage
                t.currentImage = cv2.cvtColor(t.currentImage, cv2.COLOR_BGR2GRAY)
                #히스토그램 평활
                t.currentImage = t.makeImageBright(t.currentImage)
                t.original_gray = t.currentImage
                t.currentImage = t.normalize(t.currentImage)
                t.currentImage = t.smooth(t.currentImage)
                t.currentImage = t.toBinary(t.currentImage)
                t.currentImage = t.bitAnd(t.currentImage,t.morph(t.currentImage))
                #t.currentImage = t.morph(t.currentImage)
                t.currentImage = t.detectEdge(t.currentImage)
                t.currentImage = t.lines(t.currentImage, t.currentImage)

                #boxed, rectangles = t.featureExtract(t.currentImage, t.original)
                imgs=t.featureExtract(t.currentImage, t.original_gray)
            except IOError:
                print ("I/O error ")
            except:
                print("예상치 못한 에러가 발생하였습니다.")
                text = "error"
                imgs = np.zeros((200,400,3),np.uint8)
                cv2.putText(imgs, text,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
                print("실행 오류입니다.")
            finally:
                t.show(t.original,t.winname,imgs,t.winnameCar)
                while(True):
                    key = cv2.waitKey(1)
                    if key == 13: #r키
                        cv2.destroyAllWindows()
                        break
                    elif key==27:
                        cont = False
                        cv2.destroyAllWindows()
                        break
                if cont == False:
                    break;
        except:
            print("파일 이름을 정학히 입력해주세요")
    cv2.destroyAllWindows()