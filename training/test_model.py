from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
import numpy as np
from PIL import Image

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
classes = ['aerosol_cans', 'cardboard', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells', 'food_waste', 'glass_container', 'glass_food_jars', 'magazines', 'metal_cans', 'newspaper', 'office_paper', 'paper_cups', 'plastic_bags', 'plastic_bottles', 'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_straws', 'shoes', 'styrofoam', 'tea_bags']
model = load_model("recycle_model_accurate.h5")

def predict(file):
    img = Image.open(file)
    img = img.convert("RGB").resize((256,256))
    img.load()
    img.show()
    data = np.expand_dims(np.asarray(img), axis=0)
    p = softmax(model.predict(data))
    print(p)
    print(classes[np.argmax(p)])
    print(np.argmax(p))
