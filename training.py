import  os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm #package for create processingbar
from time import sleep

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def getImagesAndLabels(path): #methd for create Traing face list
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #get image form dataset folder
    faceSamples=[] #array of face samples
    ids = [] # array of face's id
    print ("Traing face dataset  : \n")
    sleep(0.01)
    for imagePath in tqdm(imagePaths):
        PIL_img = Image.open(imagePath).convert('L') #change image to PIL image #PIL_img is spical format in opencv for image processing on cv2.LBPHFaceRecognizer
        img_numpy = np.array(PIL_img,'uint8') # array of PIL_img with "uint8" coding
        id = int(os.path.split(imagePath)[-1].split(".")[1]) #get id of each image
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w]) #append face  to array list
            ids.append(id) #append id to array list
    return faceSamples,ids
faces,ids = getImagesAndLabels('dataset') # call method to get img and id
recognizer.train(faces, np.array(ids)) #train  LBPHFaceRecognizer_NETWORK with faces and array id
recognizer.write('trainer/trainer.yml') #create train file with yml format for recognizer
