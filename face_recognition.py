import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create() #LBPH=Local Binary Patterns Histograms
# with this code line create object of binary face Patterns and _create() is Local static method of LBPHFaceRecognizer
#and face is namespace on opencv for face detection and recognizer
recognizer.read('trainer/trainer.yml') #read train file
cascadePath = "haarcascade_frontalface_default.xml" #Initialize cascadePath with CascadeClassifier path
faceCascade = cv2.CascadeClassifier(cascadePath) #create Variable of CascadeClassifier
vid_cam = cv2.VideoCapture(0)
vid_cam.set(cv2.cv2.CAP_PROP_BRIGHTNESS,100)
vid_cam.set(cv2.cv2.CAP_PROP_GAMMA,80)
vid_cam.set(cv2.cv2.CAP_PROP_FRAME_WIDTH,600)
vid_cam.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT,600)

while (True):
    ret, im = vid_cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-8,y-8), (x+w+10,y+h+10), (100,255,100), 6)
        Id,conf = recognizer.predict(gray[y:y+h,x:x+w]) # get lsit of gray image of faces and id's
        if (conf <= 80):
            if (Id == 1):
                Id = "REZA" # user id =1 /name = REZA
            elif (Id == 2):
                Id = "amir" # user id =2 /name = amir
        else:
            Id = "Unknown"
        print ("found face id : " + str(Id))
        cv2.rectangle(im, (x-22,y-65), (x+w+22, y-22), (255,255,255), -1)
        cv2.putText(im, str(Id), (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3) #set name text top of the rectangle
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
vid_cam.release()
cv2.destroyAllWindows()
