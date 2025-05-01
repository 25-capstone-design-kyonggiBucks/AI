from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from deepface import DeepFace
import numpy as np
import cv2
import io
import config
import forImage
import emotion as emotion_enum

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = str(config.UPLOAD_FOLDER)
app.config['DEBUG'] = config.DEBUG


## 이미지 리턴하는 방식 약 3초
@app.route('/api/v1/image/align-crop',methods=['post'])
def arrange_UserImage() :
    try:
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({"[ERROR]" : "이미지가 존재하지 않습니다."}),400
        
        # 메모리에서 바로 numpy array로 읽기
        img = forImage.convert_file_to_cv2_image(image_file)

        ## 이미지 bgr 변환 
        face_bgr = forImage.extract_faceImage(img)

        # 메모리에 JPEG로 저장
        _, buffer = cv2.imencode('.jpg', face_bgr)
        img_io = io.BytesIO(buffer)

        # 바로 리턴
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"[error]": str(e)}), 500
    

## 이미지 정렬 후 저장하는 방식
@app.route('/api/v2/image/align-crop',methods=['post'])
def arrange_and_upload_userImage() :

    save_path = request.form.get('save_path')
    save_path = save_path.lstrip('/')

    emotion = request.form.get('emotion')

    image_file = request.files.get('image')
    img = forImage.convert_file_to_cv2_image(image_file)

    if not save_path or not emotion or not image_file:
            return jsonify({"[ERROR]": "누락된 필드가 존재합니다."}), 400
    
    try:
            emotion = emotion_enum.Emotion(emotion)
    except ValueError:
         return jsonify({"[error]": f"Invalid emotion value: {emotion}"}), 400
    
    # 저장할 전체 경로 생성
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], save_path)
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 없으면 생성

    original_filename = secure_filename(image_file.filename)
    new_filename = forImage.create_fileName(original_filename,emotion)

    print(new_filename)

    ## 저장 경로
    output_path = os.path.join(output_dir, new_filename)
    image_url = os.path.join(save_path,new_filename)

    try:
        ## 이미지 정렬 기능 함수 호출
        forImage.extract_and_save_face2(img,output_path)
    except ValueError:
         return jsonify({
              "sucess" :False,
              "message" : "파일 저장 실패"
         }),422
    
    return jsonify({
    "sucess" : True,
    "message": "파일 저장 성공",
    "image_url": "/" + image_url,
    "image_path": output_path
}), 201

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

        
        