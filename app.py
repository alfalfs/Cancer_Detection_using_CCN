from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)
model = load_model('brain_tumor_model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file = BytesIO(file.read())
        
        img = image.load_img(file, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        return render_template('result.html', prediction=str(prediction[0][0]))
    return render_template('index.html')
    #     return 'Cancerous probability: ' + str(prediction[0][0])
    # return '''
    # <!doctype html>
    # <title>Upload an image</title>
    # <h1>Upload an image</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''

if __name__ == '__main__':
    app.run(debug=True)
