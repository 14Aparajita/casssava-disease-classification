import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Layer
from keras.models import load_model
from keras.layers import Layer
import tensorflow as tf

# Define the ECALayer class as it was in your training script
class ECALayer(Layer):
    def __init__(self, k_size=3, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.k_size = k_size
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.conv = tf.keras.layers.Conv1D(filters=1, kernel_size=k_size, padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        y = self.avg_pool(inputs)
        y = tf.expand_dims(y, axis=1)  # Expand dimensions for Conv1D
        y = self.conv(y)
        y = tf.squeeze(y, axis=1)  # Squeeze back to 3D tensor
        y = self.sigmoid(y)
        return inputs * tf.expand_dims(tf.expand_dims(y, axis=1), axis=1)

    def get_config(self):
        config = super(ECALayer, self).get_config()
        config.update({
            'k_size': self.k_size
        })
        return config

# Load the model with the custom layer
model_path = "cassava_final_model.h5"
model = load_model(model_path, custom_objects={'ECALayer': ECALayer})

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

# Define the full names of the diseases
disease_names = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mite (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

# def get_className(classNo):
#     class_names = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
#     return class_names[classNo]

def get_className(classNo):
    """ Map class number to full disease name. """
    return disease_names.get(classNo, "Unknown Disease")

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))  # Resize to the size expected by your model
    image = np.array(image)
    image = image / 255.0  # Normalize the image to the range [0, 1]
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        class_no = getResult(file_path)
        result = get_className(class_no)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
