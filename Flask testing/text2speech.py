from flask import Flask, render_template,request
# Import the required module for text 
# to speech conversion 
from gtts import gTTS 
import os 
app=Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/tts',methods=['GET','POST'])
def tts():
    mytext=request.form['text']
    mylang=request.form['lang']
    myobj = gTTS(text=mytext, lang=mylang, slow=False) 
    myobj.save("tts1.mp3")
    os.startfile("tts1.mp3")
    return render_template('tts.html',n="success")
if __name__=='__main__':
    app.run()
