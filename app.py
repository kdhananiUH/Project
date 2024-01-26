from flask import Flask, render_template, request
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('effnet.h5')

@app.route('/')
def home():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the form
    img_data = request.form['image']
    img_data = img_data.split(",")[1]  # remove the 'data:image/jpeg;base64,' part
    img = Image.open(BytesIO(base64.b64decode(img_data)))

    # Preprocess the image as needed
    new_image = cv2.resize(img, (150, 150))
    #new_image = new_image / 255.0  # Normalize the image
 
    # Make predictions
    new_image = np.expand_dims(new_image, axis=0)

    # Make a prediction
    predictions = model.predict(new_image)
    predicted_class_index = np.argmax(predictions)
    # print(predictions)
    # Map the predicted class index to the corresponding label
    labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
    predicted_tumor_type = labels[predicted_class_index]

    # Process the prediction and return the result
    # ...

    return render_template('templates/result.html', result=predicted_tumor_type)

if __name__ == '__main__':
    app.run(debug=True)
