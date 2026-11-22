# LetsGoGreen.ai

LetsGoGreen.ai is a web app that tells you whether something can be recycled. Take a photo of an item with your device's camera, and an image classifier identifies it and tells you whether it goes in the recycling bin, compost, or trash, along with any special instructions (for example, "recyclable only if empty and clean").

The site also has a **Recycling Centers** page to help you find where to drop items off.

## How it works

- **Frontend** (`templates/`): the Scan page uses the browser camera API to take a photo and posts it as a base64 image to the Flask server.
- **Backend** (`server.py`): resizes the image to 256×256, runs it through the trained Keras model, and returns the predicted class, the model's confidence, and recycling guidance for that class.
- **Model** (`model.json` + `weights.h5`): a MobileNetV2 base pretrained on ImageNet, with a new classification head trained on 23 household waste categories:

  aerosol cans, cardboard, clothing, coffee grounds, disposable plastic cutlery, eggshells, food waste, glass containers, glass food jars, magazines, metal cans, newspaper, office paper, paper cups, plastic bags, plastic bottles, plastic cup lids, plastic detergent bottles, plastic food containers, plastic straws, shoes, styrofoam, tea bags

## Running the app

Requires Python 3.10–3.11 and TensorFlow 2.12 (the model was saved with Keras 2.12).

```bash
pip install flask "tensorflow==2.12.*" pillow numpy
python server.py
```

Then open http://127.0.0.1:5000. Camera access requires `localhost` or HTTPS.

## Training

The `training/` folder contains the dataset and scripts used to build the model.

- `images/{train,val,test}/<class>/`: training data, one folder per class. Each class has both studio-style (`default`) and real-world photos. The images appear to come from the Kaggle *Recyclable and Household Waste Classification* dataset, with some similar categories merged (for example, aluminum soda cans and food cans became `metal_cans`).
- `imagenet_base_model/`: MobileNetV2 ImageNet weights without the top layer.
- `train_model.py`: builds the transfer-learning model (frozen MobileNetV2 with random flip/rotation augmentation, global average pooling, dropout, and a dense output layer), trains it for 5 epochs, and plots accuracy and loss.
- `organize_data.py`: the one-off script used to merge the soda can and food can classes.
- `test_model.py`: a helper for running the model on a single image file.

Run the training scripts from inside `training/`, because their paths are relative to that folder:

```bash
cd training
python train_model.py
```

## Contact

letsgogreen.ai@gmail.com
