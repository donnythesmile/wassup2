import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from flask import Flask, render_template, request
from deepface import DeepFace

app = Flask(__name__)

# uploads 폴더가 없으면 생성
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img1 = request.files['img1']
        img2 = request.files['img2']

        img1_path = 'uploads/' + img1.filename
        img2_path = 'uploads/' + img2.filename

        img1.save(img1_path)
        img2.save(img2_path)

        result = DeepFace.verify(img1_path, img2_path, model_name='Facenet', detector_backend='mtcnn')

        if result['verified']:
            message = "두 이미지는 동일인입니다."
        else:
            message = "두 이미지는 다른 사람입니다."

        similarity = result['distance']

        return render_template('result.html', message=message, similarity=similarity)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)