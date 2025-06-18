import os
import webbrowser
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


app = Flask(__name__)

def load_model_and_predict(image_path):
    cur_path=os.getcwd()
    # Load the saved model
    Trained_Model=os.path.join(cur_path,"TrainedModel","rice_disease_model90.keras")
    model = tf.keras.models.load_model(Trained_Model)

    # Preprocess the input image
    def preprocess_image(image_path):
        img = Image.open(image_path)
        img= img.convert("RGB")
        img = img.resize((120, 120))
        img_array = np.array(img)
        img_array = img_array.reshape(-1, 120, 120, 3)
        return img_array

    # Make predictions
    def predict_disease(image_path):
        preprocessed_image = preprocess_image(image_path)
        prediction_probs = model.predict(preprocessed_image)
        class_index = np.argmax(prediction_probs)
        return class_index

    # Map class index to disease label
    def get_disease_label(class_index):
        disease_labels = ["Brown Spot", "Healthy", "Hispa", "Leaf Blast","Bacterial Leaf Blight"]

        return disease_labels[class_index]

    # Perform prediction
    predicted_class_index = predict_disease(image_path)
    predicted_disease_label = get_disease_label(predicted_class_index)

    return predicted_disease_label


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        cur_path= os.getcwd()
        SaveImage='WebTestedImages'
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join(SaveImage, secure_filename(f.filename))
        # Save the file to the specified path
        f.save(file_path)
        # Call the model prediction function
        result = load_model_and_predict(file_path)

        return result
    return None


if __name__ == '__main__':
    # Open the application in a web browser
    webbrowser.open('http://localhost:5000')

    # Start the Flask application
    app.run(debug=False, host='0.0.0.0')
