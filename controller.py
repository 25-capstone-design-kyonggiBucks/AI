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
def arrange_UserImage():
    try:
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({"success": False, "message": "이미지가 존재하지 않습니다."}),400
        
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
        return jsonify({"success": False, "message": str(e)}), 500
    

## 이미지 정렬 후 저장하는 방식
@app.route('/api/v2/image/align-crop',methods=['post'])
def arrange_and_upload_userImage():
    try:
        emotion = request.form.get('emotion')
        image_file = request.files.get('image')
        
        if not emotion or not image_file:
            return jsonify({"success": False, "message": "누락된 필드가 존재합니다."}), 400
        
        try:
            emotion = emotion_enum.Emotion(emotion)
        except ValueError:
            return jsonify({"success": False, "message": f"Invalid emotion value: {emotion}"}), 400
        
        # 경로 중복 문제 해결
        # 스프링 서버의 설정에 맞게 uploads/images로 고정
        images_path = 'uploads/images'
        
        # 저장할 전체 경로 생성
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], images_path)
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리 없으면 생성
        print(f"이미지 저장 디렉토리: {output_dir}")

        # 파일명 생성
        original_filename = secure_filename(image_file.filename)
        new_filename = forImage.create_fileName(original_filename, emotion)

        # 저장 경로
        output_path = os.path.join(output_dir, new_filename)
        output_path = os.path.normpath(output_path)  # 경로 정규화
        
        # 스프링에서 인식할 수 있는 URL 경로 (앞에 / 추가)
        image_url = f"/{images_path}/{new_filename}"
        
        img = forImage.convert_file_to_cv2_image(image_file)
        
        # 이미지 정렬 기능 함수 호출
        forImage.extract_and_save_face2(img, output_path)
        
        print(f"파일 저장 완료: {output_path}")
        print(f"이미지 URL: {image_url}")
        
        # 스프링 서버의 FaceAlignResponse 클래스와 필드명을 일치시킴
        response_data = {
            "success": True,
            "message": "파일 저장 성공",
            "imageUrl": image_url,  # camelCase로 변경 (스프링 필드명과 일치)
            "imagePath": output_path
        }
        return jsonify(response_data), 201
        
    except Exception as e:
        print(f"예외 발생: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"파일 처리 중 오류 발생: {str(e)}"
        }), 500

if __name__ == "__main__":
    print(f"서버 시작: 설정된 업로드 폴더 = {app.config['UPLOAD_FOLDER']}")
    app.run(host='0.0.0.0', port=5001, debug=True)