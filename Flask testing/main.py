from flask import Flask, redirect, url_for, render_template,request,jsonify
from flask import send_file, current_app as app
from gtts import gTTS 
import pycountry
from flask_bootstrap import Bootstrap  
import os,cv2,pytesseract
from PIL import Image
from werkzeug.utils import secure_filename
import speech_recognition as sr
import traceback
import numpy as np
import math
from flask_bootstrap import Bootstrap 
# NLP Packages
import enchant
from textblob import TextBlob,Word 
import random 
import time
from textblob import TextBlob
import pandas as pd
import datetime
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract'

app = Flask(__name__)

Bootstrap(app)
@app.route('/')
def home():
    return render_template("home.html")


@app.route('/Text2Speech')
def Text2Speech():
    return render_template("Text2Speech.html")


@app.route('/tts',methods=['GET','POST'])
def tts():
    mytext=request.form['text']
    mylang=request.form['lang']
    lang2 = TextBlob(mytext)  
        #lane = "en"
    print(lang2.detect_language()) 
    iso_code = lang2.detect_language()  
        # # iso_code = "ta"
    language = pycountry.languages.get(alpha_2=iso_code)
    lan = language.name
    text_output = lang2.translate(to = mylang)
    print(text_output)
    extra_line = f'"{text_output}"'
    obj2 = gTTS(text = extra_line, lang = 'en', slow=True)
    while True:
        try:
            obj2.save("text2speech.mp3")
            os.startfile("text2speech.mp3")
            break
        except Exception as e:
            print("")
    return render_template('tts.html',n="success")

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/Image2Speech')
def Image2Speech():
    return render_template('Image2Speech.html')


@app.route('/api/ocr', methods=['POST','GET'])
def upload_file():
    if request.method == "GET":
        return "This is the api BLah blah"
    elif request.method == "POST":
        file = request.files['image']

        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f)
        # print(file.filename)

        image = cv2.imread(UPLOAD_FOLDER+"/"+file.filename)
        os.remove(UPLOAD_FOLDER+"/"+file.filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        preprocess = request.form["preprocess"]
        if  preprocess == "thresh":
            gray = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # make a check to see if median blurring should be done to remove
        # noise

        elif preprocess == "blur":
            gray = cv2.medianBlur(gray, 3)
        print(preprocess)
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        # print("C:/Users/mzm/PycharmProjects/My_website/ocr_using_video/"+filename,Image.open("C:\\Users\mzm\PycharmProjects\My_website\ocr_using_video\\"+filename))
        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        print("Text in Image :\n",text)
        language = 'en'
        myobj = gTTS(text=text, lang=language, slow=True) 
        while True:
            try:     
                myobj.save("test.mp3") 
                os.startfile("test.mp3") 
                break
            except Exception as e:
                print('null')
        return jsonify({"text" : text})


UPLOAD_FOLDER = "./"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/speech", methods=["GET", "POST"])
def speech():
    extra_line = ''
    if request.method == "POST":
        # Check if the post request has the file part.
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If user does not select file, browser also
        # submit an empty part without filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            # Speech Recognition stuff.
            r = sr.Recognizer()
            audio_file = sr.AudioFile(file)
            with audio_file as source:
                r.adjust_for_ambient_noise(source)
                audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            #text = recognizer.recognize_google(audio_data, key=GOOGLE_SPEECH_API_KEY, language="en-IN",show_all=True)
            
            extra_line = f'Your text: "{text}"'

            # Saving the file.
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            extra_line += f"<br>File saved to {filepath}"
            with open('out.txt', 'w') as f:
   
               print('Your Text:', text, file=f)  # Python 3.x

    
    return f"""
    <!doctype html>
    <title>Speech to Text Conversion for Audio files</title>
    
    <link rel="stylesheet" href='/static/style.css' />
    <div class="center">
    <h1 align="center">Speech to Text Conversion for Audio Files</h1>
    </div>
    </br>
    <div class="image">
    <img src="/static/img/new2.gif" align="left">
    </div>
    <h2>Upload new File</h2>
    <div class="form-group">
    <form method=post enctype=multipart/form-data>
     <p> <input type=file name=file>
      <p/>
      <p><input type="submit" value=Upload></p>
      
      
    </form>
    </div>
    <div class="align">
    {extra_line}
    </div>
    """




@app.route("/microphone", methods=["GET", "POST"])
def microphone():
    extra_line=''
    if request.method == "POST":
        store = sr.Recognizer()
        with sr.Microphone() as s:
     
            print("Speak...")
            store.adjust_for_ambient_noise(s)
            audio_input = store.record(s, duration=10)
            print("Recording time:",time.strftime("%I:%M:%S"))
    
            try:
                text_output = store.recognize_google(audio_input)
                # print("Text converted from audio:\n")
                # print(text_output)
                # print("Finished!!")
                extra_line = f'Finished...!!!'
                extra_line = f'Your text: "{text_output}"'

                # print("Execution time:",time.strftime("%I:%M:%S"))
            except:
                print("Couldn't process the audio input.")
    return f"""
    <!doctype html>
    
    
    <link rel="stylesheet" href='/static/style.css' />
    <div class="center">
    <h1 align="center">Speech to Text Conversion using Microphone</h1>
    </div>
    </br>
    <div class="image">
    <img src="/static/img/mic.gif" align="left">
    </div>
    <h2>Input Voice via Microphone</h2>
    <div class="form-group">
    <form method=post enctype=multipart/form-data>
      <p><input type="submit" value=Speak></p>
      
      
    </form>
    </div>
    <h2 alig="center">
    {extra_line}
    </h2>
    """
@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        #NLP Stuff
        blob = TextBlob(rawtext)
        received_text2 = blob
        blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
        number_of_tokens = len(list(blob.words))
        # Extracting Main Points
        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
                len_of_words = len(nouns)
                rand_words = random.sample(nouns,len(nouns))
                final_word = list()
                for item in rand_words:
                    word = Word(item).pluralize()
                    final_word.append(word)
                    summary = final_word
                    end = time.time()
                    final_time = end-start
                    
                    
    return render_template('index.html',received_text = received_text2,number_of_tokens=number_of_tokens,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,summary=summary,final_time=final_time)



@app.route("/gesture", methods=["GET", "POST"])
def gesture():

    # capture = cv2.VideoCapture(0)

    # while capture.isOpened():

    # # Capture frames from the camera
    #     ret, frame = capture.read()

    # # Get hand data from the rectangle sub window
    #     cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    #     crop_image = frame[100:300, 100:300]

    # # Apply Gaussian blur
    #     blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # # Change color-space from BGR -> HSV
    #     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # # Create a binary image with where white will be skin colors and rest is black
    #     mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # # Kernel for morphological transformation
    #     kernel = np.ones((5, 5))

    # # Apply morphological transformations to filter out the background noise
    #     dilation = cv2.dilate(mask2, kernel, iterations=1)
    #     erosion = cv2.erode(dilation, kernel, iterations=1)

    # # Apply Gaussian Blur and Threshold
    #     filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    #     ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # # Show threshold image
    #     cv2.imshow("Thresholded", thresh)

    # # Find contours
    #     #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     image, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     try:
    #     # Find contour with maximum area
    #         contour = max(contours, key=lambda x: cv2.contourArea(x))

    #     # Create bounding rectangle around the contour
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

    #     # Find convex hull
    #         hull = cv2.convexHull(contour)

    #     # Draw contour
    #         drawing = np.zeros(crop_image.shape, np.uint8)
    #         cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
    #         cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

    #     # Find convexity defects
    #         hull = cv2.convexHull(contour, returnPoints=False)
    #         defects = cv2.convexityDefects(contour, hull)

    #     # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
    #     # tips) for all defects
    #         count_defects = 0

    #         for i in range(defects.shape[0]):
    #             s, e, f, d = defects[i, 0]
    #             start = tuple(contour[s][0])
    #             end = tuple(contour[e][0])
    #             far = tuple(contour[f][0])

    #             a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    #             b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    #             c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    #             angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

    #         # if angle > 90 draw a circle at the far point
    #             if angle <= 90:
    #                 count_defects += 1
    #                 cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

    #             cv2.line(crop_image, start, end, [0, 255, 0], 2)

    #     # Print number of fingers
    #         if count_defects == 0:
    #             cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
    #         elif count_defects == 1:
    #             cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    #         elif count_defects == 2:
    #             cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    #         elif count_defects == 3:
    #             cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    #         elif count_defects == 4:
    #             cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    #         else:
    #             pass
    #     except:
    #         pass

    # # Show required images
    #     cv2.imshow("Gesture", frame)
    #     all_image = np.hstack((drawing, crop_image))
    #     cv2.imshow('Contours', all_image)

    # # Close the camera if 'q' is pressed
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # return render_template('home.html')
    # capture.release()
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
     
    while(1):
        
        try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
            ret, frame = cap.read()
            frame=cv2.flip(frame,1)
            kernel = np.ones((3,3),np.uint8)
        
            #define region of interest
            roi=frame[100:300, 100:300]
        
        
            cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
            # define range of skin color in HSV
            lower_skin = np.array([0,20,70], dtype=np.uint8)
            upper_skin = np.array([20,255,255], dtype=np.uint8)
        
            #extract skin colur imagw  
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
            #extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask,kernel,iterations = 4)
        
            #blur the image
            mask = cv2.GaussianBlur(mask,(5,5),100) 
     
            #find contours
            #contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            hierarchy, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(contours)
            print(hierarchy)
            #find contour of max area(hand)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
            #approx the contour a little
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
            #make convex hull around hand
            hull = cv2.convexHull(cnt)
        
            #define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
      
            #find the percentage of area not covered by hand in convex hull
            arearatio=((areahull-areacnt)/areacnt)*100
    
            #find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
        
            # l = no. of defects
            l=0
        
            #code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt= (100,180)
            
            
                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
                #distance between point and convex hull
                d=(2*ar)/a
            
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d>30:
                    l += 1
                    cv2.circle(roi, far, 3, [255,0,0], -1)
            
                #draw lines around hand
                cv2.line(roi,start, end, [0,255,0], 2)
            
            
            l+=1
        
            #print corresponding gestures which are in their ranges
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l==1:
                if areacnt<2000:
                    cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    if arearatio<12:
                        cv2.putText(frame,'0 - stop',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    elif arearatio<17.5:
                        cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
                    else:
                        cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
            elif l==2:
                cv2.putText(frame,'2 - peace',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            elif l==3:
         
                if arearatio<27:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
            elif l==4:
                cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            elif l==5:
                cv2.putText(frame,'5 - Hi',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            elif l==6:
                cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            else :
                cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        #show the windows
            cv2.imshow('mask',mask)
            cv2.imshow('frame',frame)
        except Exception:
            traceback.print_exc()
            pass
       # break
        
    
        if cv2.waitKey(1) == ord('q'):
            break
    return render_template('home.html')
    cv2.destroyAllWindows()
    cap.release()    


@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    return render_template("reviews.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    df= pd.read_csv("clothing.csv", encoding="latin-1")
    df['review'] = df['review'].fillna('')
    
    def clean_and_tokenize(review):
        text = review.lower()
    
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokens = tokenizer.tokenize(text)
    
        stemmer = nltk.stem.WordNetLemmatizer()
        text = " ".join(stemmer.lemmatize(token) for token in tokens)
        text = re.sub("[^a-z']"," ", text)
        return text
    df["Clean_Review"] = df["review"].apply(clean_and_tokenize)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['recommend']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5320)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    lr_pred=lr.predict(X_test)

        
    if request.method == 'POST':
       # message = request.form['message']
        text=request.form.get('text')
        data = [text]
        vect = vectorizer.transform(data).toarray()
        my_prediction = lr.predict(vect)
    return render_template('prediction_reviews.html',prediction = my_prediction)

@app.route('/similar_words',methods=["GET","POST"])
def similar_words():
    extra_line=''
    if request.method =="POST":
        mytext=request.form['text']
        mylang=request.form['lang']
        d = enchant.Dict(mylang)
        lan = 'en-UK'
        word = mytext
        d.check(word)
        print(d.suggest(word))
        text_output = d.suggest(word)
        extra_line = f'Similar Words are : "{text_output}"'
        print(extra_line)
        obj = gTTS(text = extra_line, lang = lan, slow=False)
        obj.save("words.mp3")
        os.startfile("words.mp3")
    return render_template('Similar_Word_Suggestion.html',new = extra_line)  


@app.route('/language_detection',methods=["GET","POST"])
def language_detection():
    extra_line=''
    if request.method =="POST":
        mytext=request.form['text'] 
            # Language Detection 
        lang1 = TextBlob(mytext)  
        lane = "en"
        print(lang1.detect_language()) 
        iso_code = lang1.detect_language()  
        # iso_code = "ta"
        language = pycountry.languages.get(alpha_2=iso_code)
        lan = language.name
        extra_line = f'The language detected is :"{lan}"'
        obj1 = gTTS(text = extra_line, lang = lane, slow=True)
        while True:
            try:
                obj1.save("language.mp3")
                os.startfile("language.mp3")
                break
            except Exception as e:
                print("")
    return render_template('language_detection.html',new = extra_line)


@app.route('/language_translator',methods=["GET","POST"])
def language_translator():
    extra_line=''
    if request.method =="POST":
        mytext1=request.form['text']
        mylang=request.form['lang'] 
            # Language Detection 
        lang2 = TextBlob(mytext1)  
        #lane = "en"
        print(lang2.detect_language()) 
        iso_code = lang2.detect_language()  
        # # iso_code = "ta"
        language = pycountry.languages.get(alpha_2=iso_code)
        lan = language.name
        text_output = lang2.translate(to = mylang)
        print(text_output)
        extra_line = f'The language is translated as  :"{text_output}"'
        obj2 = gTTS(text = extra_line, lang = 'en', slow=False)
        obj2.save("translated.mp3")
        os.startfile("translated.mp3")
    return render_template('language_translator.html',new = extra_line)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/visualizattions')
def visualizattions():
    return render_template('visualizattions.html')



@app.route('/draw_air',methods=["GET","POST"])
def draw_air():
    x,y,k = 200,200,-1
    cap = cv2.VideoCapture(0)
    def take_inp(event, x1, y1, flag, param):
        global x, y, k
        if event == cv2.EVENT_LBUTTONDOWN:
            x = x1
            y = y1
            k = 1

    cv2.namedWindow("enter_point")
    cv2.setMouseCallback("enter_point", take_inp)

##### taking input point ######################
    while True:
     
        _, inp_img = cap.read()
        inp_img = cv2.flip(inp_img, 1)
        gray_inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    
        cv2.imshow("enter_point", inp_img)
    
        if k == 1 or cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            break

##############################################
    stp = 0
########## opical flow starts here ###########


    old_pts = np.array([[x, y]], dtype=np.float32).reshape(-1,1,2)
    mask = np.zeros_like(inp_img)

    while True:
        _, new_inp_img = cap.read()
        new_inp_img = cv2.flip(new_inp_img, 1)
        new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)     
        new_pts,status,err = cv2.calcOpticalFlowPyrLK(gray_inp_img, 
                         new_gray, 
                         old_pts, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))

        for i, j in zip(old_pts, new_pts):
            x,y = j.ravel()
            a,b = i.ravel()
            if cv2.waitKey(2) & 0xff == ord('s'):
                stp = 1
            
            elif cv2.waitKey(2) & 0xff == ord('w'):
                stp = 0
        
            elif cv2.waitKey(2) == ord('n'):
                mask = np.zeros_like(new_inp_img)
            
            if stp == 0:
                mask = cv2.line(mask, (a,b), (x,y), (0,0,255), 6)

            cv2.circle(new_inp_img, (x,y), 6, (0,255,0), -1)
    
        new_inp_img = cv2.addWeighted(mask, 0.3, new_inp_img, 0.7, 0)
        cv2.putText(mask, "'s' to gap 'w' - start 'n' - clear", (10,50), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
        cv2.imshow("ouput", new_inp_img)
        cv2.imshow("result", mask)

    
        gray_inp_img = new_gray.copy()
        old_pts = new_pts.reshape(-1,1,2)
    
        #if cv2.waitKey(1) & 0xff == 27:
         #   break
        if cv2.waitKey(1) == ord('q'):
            break
    return render_template('home.html')
    
#### thank you for this vi

    cv2.destroyAllWindows()
    cap.release()


@app.route('/security',methods=["GET","POST"])
def security():
    cap = cv2.VideoCapture(0)

    _, inp_img = cap.read()
    inp_img = cv2.flip(inp_img, 1)
    inp_img = cv2.blur(inp_img, (4,4))
    gray_inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

##############################################

########## tracking starts here ##########
    old_pts = np.array([[350, 180], [350, 350]], dtype=np.float32).reshape(-1,1,2)

    backup = old_pts.copy()
    backup_img = gray_inp_img.copy()

#### text o/p window
    outp = np.zeros((480,640,3))

#### variable ####
    ytest_pos = 40
###############

    while True:
        _, new_inp_img = cap.read()
        new_inp_img = cv2.flip(new_inp_img, 1)
        new_inp_img = cv2.blur(new_inp_img, (4,4))
        new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)     
        new_pts,status,err = cv2.calcOpticalFlowPyrLK(gray_inp_img, 
                         new_gray, 
                         old_pts, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))
    #### boundries
        if new_pts.ravel()[0]  >= 600:
            new_pts.ravel()[0] = 600
        if new_pts.ravel()[1] >= 350:
            new_pts.ravel()[1] = 350
        if new_pts.ravel()[0]  <= 20:
            new_pts.ravel()[0] = 20
        if new_pts.ravel()[1] <= 150:
            new_pts.ravel()[1] = 150
        if new_pts.ravel()[2]  >= 600:
            new_pts.ravel()[2] = 600
        if new_pts.ravel()[3] >= 350:
            new_pts.ravel()[3] = 350
        if new_pts.ravel()[2]  <= 20:
            new_pts.ravel()[2] = 20
        if new_pts.ravel()[3] <= 150:
            new_pts.ravel()[3] = 150
    ###############

    ##### drawing line
        x,y = new_pts[0,:,:].ravel()
        a,b = new_pts[1,:,:].ravel()
        cv2.line(new_inp_img, (x,y), (a,b), (0,0,255), 15)
    

        cv2.imshow("ouput", new_inp_img)
    

    
        if new_pts.ravel()[0]  > 400 or new_pts.ravel()[2]  > 400:        
            if new_pts.ravel()[0] > 550 or new_pts.ravel()[2]  > 550:
                new_pts = backup.copy()
                new_inp_img = backup_img.copy()
                ytest_pos += 40
                cv2.putText(outp, "gone at {}".format(datetime.datetime.now().strftime("%H:%M")), (10,ytest_pos), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))



        
        elif new_pts.ravel()[0]  < 200 or new_pts.ravel()[2]  < 200:        
            if new_pts.ravel()[0] < 50 or new_pts.ravel()[2]  < 50:
                new_pts = backup.copy()
                new_inp_img = backup_img.copy()
                ytest_pos += 40
                cv2.putText(outp, "came at {}".format(datetime.datetime.now().strftime("%H:%M")), (10,ytest_pos), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255))
  
        
    
        cv2.imshow('final', outp)
        gray_inp_img = new_gray.copy()
        old_pts = new_pts.reshape(-1,1,2)

        #if cv2.waitKey(1) & 0xff == 27:
            #break
        if cv2.waitKey(1) == ord('q'):
            break
    
#### thank you for this video
    return render_template('home.html')
    cv2.destroyAllWindows()
    cap.release()



@app.route('/speed',methods=["GET","POST"])
def speed():
    cap = cv2.VideoCapture(0)
    startx = -1

    def captu():
        print("captu called")
        global startx
        startx = -1    
    ####### capture startx
        _, prevc = cap.read()
        prevc= cv2.flip(prevc, 1)

        while True:
            _, newc = cap.read()
            newc = cv2.flip(newc, 1)
            diffc = cv2.absdiff(prevc, newc)
            diffc = cv2.cvtColor(diffc, cv2.COLOR_BGR2GRAY)
            diffc = cv2.blur(diffc, (4,4))
            _, diffc = cv2.threshold(diffc, 10,255, cv2.THRESH_BINARY)
            _,contorc,_ = cv2.findContours(diffc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            for contc in contorc :
                if cv2.contourArea(contc) > 30000:
                    (x1c,y1c),rad = cv2.minEnclosingCircle(contc)
                    if x1c < 545 and x1c > 105:
                        startx = x1c
                
            prevc = newc.copy()

            cv2.imshow("diffc", diffc)


            if cv2.waitKey(1) == 27 or startx != -1:
                break

        return startx

################################################
    def right():
        print("from right")
        endxr = -1
        global startx
        start_time = time.time()
        _, prevr = cap.read()
        prevr= cv2.flip(prevr, 1)
        while True:
            _, newr = cap.read()
            newr = cv2.flip(newr, 1)
            diffr = cv2.absdiff(prevr, newr)
            diffr = cv2.cvtColor(diffr, cv2.COLOR_BGR2GRAY)
            diffr = cv2.blur(diffr, (4,4))
            _, diffr = cv2.threshold(diffr, 10,255, cv2.THRESH_BINARY)
            _,contorr,_ = cv2.findContours(diffr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contr in contorr :
                if cv2.contourArea(contr) > 30000:
                    (x,y,w,h) = cv2.boundingRect(contr) 
                    (x1r,y1r),rad = cv2.minEnclosingCircle(contr)
                    cv2.rectangle(prevr, (x,y), (x+w,y+h), (0,255,0), 2)
                    if x1r > 550:                    
                        endxr = x1r
                        end_time = time.time()
                        print("it took {}".format(end_time-start_time))
                        cv2.putText(prevr, "speed:{} px/sec".format(math.trunc(np.sqrt((550-startx)**2)/(end_time-start_time))),(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                        break
                    
            cv2.imshow("right", prevr)
            prevr = newr.copy()

            if endxr > 550 or cv2.waitKey(1) == 27:
                break
        return None

    def left():
        global startx
        endx = 700
        print("from left")
        start_time = time.time()
        _, prev = cap.read()
        prev= cv2.flip(prev, 1)
        while True:     
            _, new = cap.read()
            new = cv2.flip(new, 1)
            diff = cv2.absdiff(prev, new)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff = cv2.blur(diff, (4,4))
            _, diff = cv2.threshold(diff, 10,255, cv2.THRESH_BINARY)
            _,contor,_ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cont in contor :
                if cv2.contourArea(cont) > 30000:
                
                    (x,y,w,h) = cv2.boundingRect(cont) 
                    (x1,y1),rad = cv2.minEnclosingCircle(cont)
                    cv2.rectangle(prev, (x,y), (x+w,y+h), (0,255,0), 2)
                    if x1 < 100:                      
                        endx = x1
                        end_time = time.time()
                        print("it took {}".format(end_time-start_time))
                        cv2.putText(prev, "speed:{} px/sec".format(math.trunc(np.sqrt((startx - x1)**2 )/(end_time-start_time))),(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

                        break
                    
            cv2.imshow("left", prev)
            prev = new.copy()

            if endx < 100 or cv2.waitKey(1) == 27:
                break
        return None

############ CAPTURE ENDX    ###########################

    while True : 
        if captu() < 150:
            right()    
        else:
            left()

        #if cv2.waitKey(1) == 27:
         #   break
        if cv2.waitKey(1) == ord('q'):
            break

    return render_template('home.html')
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    app.run(debug=True)