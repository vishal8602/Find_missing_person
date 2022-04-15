import numpy as np
import cv2
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report



face_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_eye.xml')


def get_cropped_image_if_2_eyes(img):
    #img = cv2.imread(image_path)
    gray=None
    try :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except: pass
    if gray is not None  :
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #if len(eyes) >= 2:
            return roi_color
    return None

# #--------------------------------------wavelet transform--------------------------

import pywt
import cv2    

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


# #--------------------------------------------------------------------------------------------------------------------------
# #--------------------------------------------------------------------------------------------------------------------------


import streamlit as st



#modules

import pyrebase
from pyrebase.pyrebase import storage 
import firebase_admin
from firebase_admin import storage, credentials
import pickle
from datetime import datetime
#configuration 

firebaseConfig = {
  'apiKey': "AIzaSyDhgd3FaW3e3l3DJJ79eParkGJB-0UHjuc",
  'authDomain': "findmissing-e2f89.firebaseapp.com",
  'projectId': "findmissing-e2f89",
  'storageBucket': "findmissing-e2f89.appspot.com",
  'databaseURL' :"https://findmissing-e2f89-default-rtdb.europe-west1.firebasedatabase.app/",
  'messagingSenderId': "221231936886",
  'appId': "1:221231936886:web:f6164bf1e571bf14d8b657",
  'measurementId': "G-4DEHRL9Z56",
  'serviceAccount': "findmissing-e2f89-firebase-adminsdk-wm9zz-cdb4e0c510.json"
  
}

#firebase authentication

firebase=pyrebase.initialize_app(firebaseConfig)

auth=firebase.auth()

#database

db=firebase.database()
storage=firebase.storage()





def find_person(img):
             
              import numpy as np
              image=get_cropped_image_if_2_eyes(img)
              
              if(image is None):
                
                st.text("Not Able to crop that image Please Upload clear Image")
                col1,col2=st.columns(2)
                with col1:
                 if(st.button("If You Don't Have Any Other Image Continue with same image But chances of getting correct result will decreases")):
                      image=img
                      X=[]
                      y=[]

                      scalled_raw_img = cv2.resize(image, (32, 32))
                      st.image(scalled_raw_img,"scalled_raw_img image")
                      img_har = w2d(image,'db1',5)
                      print(type(scalled_raw_img))

                      st.image(img_har,"Wavelet transform of image")
                      scalled_img_har = cv2.resize(img_har, (32, 32))
                      st.image(scalled_img_har,"scalled_img_har image")
                      print(type(scalled_img_har))

                      combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
                      # combined_img=np.array(combined_img)
                      #st.image(combined_img,"Combined image for feature extraction")
                      print(type(combined_img))
                      X.append(combined_img)

                      X = np.array(X).reshape(len(X),4096).astype(float)

                      import urllib.request, urllib.parse, urllib.error
                      url = storage.child("name.txt").get_url(None)
                      text_file =urllib.request.urlopen(url).read()
                      name=pickle.loads(text_file)

                      url = storage.child("pipeSVM.pkl").get_url(None)
                      text_file =urllib.request.urlopen(url).read()
                      pipeSVM=pickle.loads(text_file)

        #               url = storage.child("pipeLOGistic.pkl").get_url(None)
        #               text_file =urllib.request.urlopen(url).read()
        #               pipeLOGistic=pickle.loads(text_file)





                      ansSVM=pipeSVM.predict(X)
                      st.text_area("SVM Match Found name of the Person is ",name[ansSVM[0]])
        #             ansLOGistic=pipeLOGistic.predict(X)
                with col2:    
                    import pyautogui
                    if st.button("New Image"):
                        pyautogui.hotkey("ctrl","F5")
              else:
                      st.image(image,"cropped image")
                      X=[]
                      y=[]

                      scalled_raw_img = cv2.resize(image, (32, 32))
                      st.image(scalled_raw_img,"scalled_raw_img image")
                      img_har = w2d(image,'db1',5)
                      print(type(scalled_raw_img))

                      st.image(img_har,"Wavelet transform of image")
                      scalled_img_har = cv2.resize(img_har, (32, 32))
                      st.image(scalled_img_har,"scalled_img_har image")
                      print(type(scalled_img_har))

                      combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
                      # combined_img=np.array(combined_img)
                      #st.image(combined_img,"Combined image for feature extraction")
                      print(type(combined_img))
                      X.append(combined_img)

                      X = np.array(X).reshape(len(X),4096).astype(float)

                      import urllib.request, urllib.parse, urllib.error
                      url = storage.child("name.txt").get_url(None)
                      text_file =urllib.request.urlopen(url).read()
                      name=pickle.loads(text_file)

                      url = storage.child("pipeSVM.pkl").get_url(None)
                      text_file =urllib.request.urlopen(url).read()
                      pipeSVM=pickle.loads(text_file)

        #               url = storage.child("pipeLOGistic.pkl").get_url(None)
        #               text_file =urllib.request.urlopen(url).read()
        #               pipeLOGistic=pickle.loads(text_file)





                      ansSVM=pipeSVM.predict(X)
                      st.text_area("SVM Match Found name of the Person is ",name[ansSVM[0]])
                #             ansLOGistic=pipeLOGistic.predict(X)
             


           
           
               
                   

def report_person(image_list,name_report):
            

            import urllib.request, urllib.parse, urllib.error
            
            url = storage.child("x.txt").get_url(None)
            text_file =urllib.request.urlopen(url).read()
            X=pickle.loads(text_file)


          
            url = storage.child("y.txt").get_url(None)
            text_file =urllib.request.urlopen(url).read()
            Y=pickle.loads(text_file)


           
            url = storage.child("name.txt").get_url(None)
            text_file =urllib.request.urlopen(url).read()
            name=pickle.loads(text_file)
            
         
            
            url = storage.child("pipeSVM.pkl").get_url(None)
            text_file =urllib.request.urlopen(url).read()
            pipeSVM=pickle.loads(text_file)
            
            
            
#             url = storage.child("pipeLOGistic.pkl").get_url(None)
#             text_file =urllib.request.urlopen(url).read()
#             pipeLOGistic=pickle.loads(text_file)
            
           
            st.header("Already register User Name with ID")
            new_name  = pd.DataFrame.from_records([name])
            st.dataframe(new_name)
            
            
            name[len(name)]=name_report
            new_x=[]
            count=0
            
            
            st.text_area("TOTAL IMAGES TILL NOW ",len(X))
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
            
            
            SCORE1=pipeSVM.score(X_test, y_test)
            SCORE1=float(SCORE1)
            st.text_area("SCORE OF SVM BEFORE ",SCORE1)
            
            
#             SCORE2=pipeLOGistic.score(X_test, y_test)
#             SCORE2=float(SCORE2)
#             st.text_area("SCORE OF LOGISTIC REGRESSION ",SCORE2)
            
            for i in image_list:
#               st.image(i,"before cropping")
              image=get_cropped_image_if_2_eyes(i)
              count=count+1
              #st.image(image,"cropped image")
              
          
              scalled_raw_img = cv2.resize(image, (32, 32))
              #st.image(scalled_raw_img,"scalled_raw_img image")

              img_har = w2d(image,'db1',5)
              #st.image(img_har,"Wavelet transform of image")

              scalled_img_har = cv2.resize(img_har, (32, 32))
              #st.image(scalled_img_har,"scalled_img_har image")
              

              combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
              new_x.append(combined_img)
              Y.append(name[name_report])   
            st.text_area("TOTAL NO OF IMAGES UPLOADED BY YOU",count)
           
            new_x = np.array(new_x).reshape(len(new_x),4096).astype(float)
            X=np.vstack((X,new_x))
            X=np.array(X).reshape(len(X),4096).astype(float)
            # Y=np.array(Y)
#             print("len of x and y",len(X),len(Y)) 
#             print("type o x and y",type(X),type(Y))
            # from sklearn.preprocessing import MultiLabelBinarizer  
            # Y=MultiLabelBinarizer().fit_transform(Y)     
             
   
           
#             st.text_area(Y)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
            pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
            pipe.fit(X_train, y_train)
            SCORE=pipe.score(X_test, y_test)
            SCORE=float(SCORE)
            st.text_area("NEW SCORE of SVM NEWLY CREATED MODEL is",SCORE)
            
            storage.delete('pipeSVM.pkl',"https://firebasestorage.googleapis.com/v0/b/findmissing-e2f89.appspot.com/o/pipeSVM.pkl?alt=media&token=64756737-fdca-4e3d-8517-d205a1476bab")
            filename=pickle.dumps(pipeSVM)
            chunk_size='262144'
            storage.child('pipeSVM.pkl').put(filename,chunk_size)
            
            
#             pipe=LogisticRegression(solver='liblinear',multi_class='auto')
#             pipe.fit(X_train, y_train)
#             SCORE=pipe.score(X_test, y_test)
#             SCORE=float(SCORE)
#             st.text_area("NEW SCORE of LOGISTIC REGRESSION NEWLY CREATED MODEL is",SCORE)
            
            
#             storage.delete('pipeLOGistic.pkl',"https://firebasestorage.googleapis.com/v0/b/findmissing-e2f89.appspot.com/o/pipeSVM.pkl?alt=media&token=64756737-fdca-4e3d-8517-d205a1476bab")
#             filename=pickle.dumps(pipe)
#             chunk_size='262144'
#             storage.child('pipeSVM.pkl').put(filename,chunk_size)
            
            storage.delete('name.txt',"https://firebasestorage.googleapis.com/v0/b/findmissing-e2f89.appspot.com/o/pipeSVM.pkl?alt=media&token=64756737-fdca-4e3d-8517-d205a1476bab")
            filename=pickle.dumps(name)
            chunk_size='262144'
            storage.child('name.txt').put(filename,chunk_size)
            
            storage.delete('x.txt',"https://firebasestorage.googleapis.com/v0/b/findmissing-e2f89.appspot.com/o/pipeSVM.pkl?alt=media&token=64756737-fdca-4e3d-8517-d205a1476bab")
            filename=pickle.dumps(X)
            chunk_size='262144'
            storage.child('x.txt').put(filename,chunk_size)
            
            storage.delete('y.txt',"https://firebasestorage.googleapis.com/v0/b/findmissing-e2f89.appspot.com/o/pipeSVM.pkl?alt=media&token=64756737-fdca-4e3d-8517-d205a1476bab")
            filename=pickle.dumps(Y)
            chunk_size='262144'
            storage.child('y.txt').put(filename,chunk_size)
            

if(st.button("Upload Model")):
    storage.child('pipeSVM.pkl').put('pipeSVM.pkl')


st.title('Find Missing Person APP')
option=st.sidebar.selectbox("Menu",("Home","Login","Sign Up","Find_Missing_Person","Report_Missing_Person","Register Users"))


if(option=="Home"):
     st.header("Welcome !! We are here to help you in finding missing person")


elif(option=="Login"):

    st.header("Login Page")
    email=st.text_input("Username")
    password=st.text_input("Password" ,type="password")
    # create_usertable()
    if(st.button("Login") ):
      try:
        user=auth.sign_in_with_email_and_password(email, password)
        st.success("Logged in succesfully")
        st.balloons()
      except:
         st.warning("Username Password Not valid")

elif(option=="Sign Up"):
    st.header("Sign Up Page")
    email=st.text_input("Email",value='Default')
    password=st.text_input("Password",type="password")
    if(st.button("SignUp")):
        
        try:
          user=auth.create_user_with_email_and_password(email,password)
          st.success("Sign Up succesfully Please select login to login")
          st.balloons()   
        except:
          st.warning("User Already Register")

elif(option=="Find_Missing_Person"):
      st.header("Find Missing Person")
      st.text_input("Name of person")
      st.text_input("Age")
      input_image=st.file_uploader("Upload Image Of Missing Person")
      if(input_image):
        from PIL import Image
        image=Image.open(input_image)
        image=np.array(image)
        st.header("Image Upaloded Succesfully")
        st.image(image,"uploaded image")
        find_person(image)



elif(option=="Report_Missing_Person"):
      st.header("Report_Missing_Person")
      name=st.text_input("Name of person")
      age=st.text_input("Age")
      st.text("Please Upload at Least more than 5 image of Reporting Person size Must >=1MB")
      input_image=st.file_uploader("Upload Image Person",accept_multiple_files=True)
      new_list=[]
      if(input_image):
        from PIL import Image
        for i in input_image:
              image=Image.open(i)
              image=np.array(image)
              new_list.append(image)
        report_person(new_list,name)

elif(option=="Register Users"):
     import urllib.request, urllib.parse, urllib.error
     url = storage.child("name.txt").get_url(None)
     text_file =urllib.request.urlopen(url).read()
     name=pickle.loads(text_file)
     st.table([name])
     
# ------------testing

# if(st.button("For deleting the old model")):
#     storage.delete("Vishal","https://firebasestorage.googleapis.com/v0/b/findmissing-e2f89.appspot.com/o/pipeSVM.pkl?alt=media&token=64756737-fdca-4e3d-8517-d205a1476bab")
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------


# !streamlit  run app.py & npx localtunnel --port 8501
# !pip install streamlit pyrebase
#!pip install firebase-admin
# import streamlit as st
# st.title("Learning")
# st.header("Load image")
# input_image=st.file_uploader("Select image")
# from PIL import Image
# if(input_image):
  
#   image=Image.open(input_image)
#   st.image(image)
#   print(type(image))
#   image=np.array(image)
#   image=get_cropped_image_if_2_eyes(image)
#   print(type(image))
#   plt.imshow(image)
#   st.image(image,"cropped image")
