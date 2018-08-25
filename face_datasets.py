import cv2 #import package opencv with cv2's name for image processing  OPENCV = Open Source Computer Vision

vid_cam = cv2.VideoCapture(0)  #create videocam frame and using port = 0 beacuse port 0  for default computer camera
vid_cam.set(cv2.cv2.CAP_PROP_BRIGHTNESS,100) #fix brightness for frame
vid_cam.set(cv2.cv2.CAP_PROP_GAMMA,50) #fix gamma for back color and white color
vid_cam.set(cv2.cv2.CAP_PROP_FRAME_WIDTH,600) #set eidth for frame
vid_cam.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT,600) #set height for frame

#Working with a boosted cascade of weak classifiers includes two major stages: the training and the detection stage. The detection stage using either HAAR or ('haarcascade_frontalface_default.xml')
#LBP based models ,  CascadeClassifier has face feacher for detect face  and Includes weights of shapes to specify facial shapes
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id =input("enter face id : ") # set id for each face

sampleNum = 0 #counter for Counting face

while (True): #video is frame of pictures while for showing all

    rest,image_frame = vid_cam.read() # read frame of cam and save in image_frame Variable
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)  #change image_frame color from RGB to GRAY color
    faces = face_detector.detectMultiScale(gray, 1.3, 5) #cascade – Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load()
    #const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size()

    for (x,y,w,h) in faces: #The face has four attributes for detection
            sampleNum = sampleNum + 1
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2) #creating rectangle around face image_frame = image (x,y)=weights , height , (255...) = color
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #after detect face  create frame then saving image with jpg format
    cv2.imshow('face detection', image_frame) #show videocam frame after creating in line 3
    cv2.waitKey(1) #stop frame with key = 1
    if (sampleNum > 100): #after saving 100 sample break while
        break
    if cv2.waitKey(100) & 0xFF == ord('q'): #stop VideoCapture with Q  keyboard
        break
vid_cam.release() #release camera on port 0
cv2.destroyAllWindows() #kill / destroing  window's frame
