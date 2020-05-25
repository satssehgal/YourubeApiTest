import json
import pandas as pd
import cv2
import sys
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from matplotlib.pyplot import imshow
from pytube import YouTube
from youtube_search import YoutubeSearch
import warnings
warnings.filterwarnings("ignore")
from docsim import YTCompare

pd.set_option('display.max_columns',None)

search_term='learn python programming'
max_results=30

results = YoutubeSearch(search_term, max_results=max_results).to_json()
res = json.loads(results)['videos']

links=[res[i]['link'] for i in range(max_results)]

ytdatalist=[]
for i in range(max_results):
	yt = YouTube('http://youtube.com/{}'.format(links[i]))
	ytdata={
		'title': yt.title,
		'thurl':yt.thumbnail_url,
		'length':yt.length,
		'views':yt.views,
        'author':yt.author
	}
	ytdatalist.append(ytdata)

df=pd.DataFrame(ytdatalist)
df['faces']=0
df['face_eyes']=0
df['smile']=0

def addFaces(thurl):
    response = requests.get(thurl)
    img_PIL = Image.open(BytesIO(response.content))
    gray_PIL = Image.open(BytesIO(response.content)).convert('L')
    img = np.array(img_PIL) 
    gray = np.array(gray_PIL) 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyesz=[]
    smilesz=[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyesz.append(eyes)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
            smilesz.append(smiles)
            
    return len(faces), len(smilesz), len(eyesz)
 
for f in range(max_results):
    imagePath=df.iloc[f]['thurl']
    df['faces'].iloc[f]=int(addFaces(imagePath)[0])
    df['smile'].iloc[f]= 1 if int(addFaces(imagePath)[1]) >0 else 0
    df['face_eyes'].iloc[f]= int(addFaces(imagePath)[2]/2)

df['SimScores']=0
df['SimScores']=[YTCompare(search_term, ytdatalist[i]['title']) for i in range(max_results)]   
df.sort_values(by='views', ascending=False)


