# Import the necessary libraries
from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import model_from_json
from tensorflow.nn import softmax

from numpy import asarray, expand_dims, argmax
from json import dumps, load
from PIL import Image
from io import BytesIO
from base64 import b64decode

# Declare the categories an their respective recylabilities (0=non-recylable, 1=recylcable, and 2,3,4=special instructions)
classes = ['an aerosol can', 'cardboard', 'clothing', 'coffee grounds', 'disposable plastic cutlery', 'eggshells', 'food waste', 'a glass container', 'a glass food jar', 'a magazine', 'a metal can', 'a newspaper', 'office paper', 'a paper cup', 'a plastic bag', 'a plastic bottle', 'a plastic cup lid', 'a plastic detergent bottle', 'a plastic food container', 'a plastic straw', 'a shoe', 'styrofoam', 'a tea bag']
re = [4, 1, 1, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1, 0, 3, 4, 1, 4, 4, 0, 1, 0, 0]

# Start the app
app = Flask(__name__)

# Load the model
with open("model.json", 'r') as json_file:
    architecture = load(json_file)
    model = model_from_json(dumps(architecture))
    model.load_weights("weights.h5")

# Declare URL's
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Centers')
def centers():
    return render_template('Centers.html')
@app.route('/Scan', methods=["GET", "POST"])
def scan():
  user_agent = request.headers.get('User-Agent')
  print(user_agent)
  if request.method == "POST":
    b64 = request.form["data"].split(",")[1]
    decode = b64decode(b64)
    file = BytesIO(decode)
    img = Image.open(file)
    img = img.convert("RGB").resize((256,256))
    data = expand_dims(asarray(img), axis=0)
    p = softmax(model.predict(data))
    i = argmax(p)
    s = "This looks like " + classes[i] + " with " +  "{:.2f}".format(p[0][i].numpy()*100) + "% certainty. "
    if (re[i] == 0):
        s = s + "Unfortunately, this object is not recyclable - place it in the trash bin instead."
    elif (re[i] == 1):
        s = s + "This object is recyclable! Feel free to place it in the recycle bin."
    elif (re[i] == 2):
        s = s + "While this object is not recyclable, it is compostable! If you have a compost bin, place it there - otherwise you can put it in the trash bin."
    elif (re[i] == 4):
        s = s + "This object can be recycled only if it is empty and clean."
    elif (re[i] == 3):
        s = s + "This object may be recyclable if it is empty and clean, but it does depend on the type of plastic bag."
    return jsonify(result=s)
  return render_template("Scan.html")
  

      
if __name__ == '__main__':
  app.run()
