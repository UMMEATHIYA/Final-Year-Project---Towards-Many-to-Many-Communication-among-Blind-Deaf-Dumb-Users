import os,cv2,pytesseract
from flask import Flask, render_template, request,jsonify
from PIL import Image
# to speech conversion 
from gtts import gTTS 
import os 

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract'

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


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
        myobj = gTTS(text=text, lang=language, slow=False) 
        myobj.save("test.mp3") 
        os.startfile("test.mp3")
        #os.system("mpg321 welcome.wav") 
        return jsonify({"text" : text})
        
        
app.run("0.0.0.0",5000,threaded=True,debug=True)


