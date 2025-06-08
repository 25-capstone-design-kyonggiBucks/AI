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
import customForVideo
import uuid
import json
import shutil

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

@app.route('/api/custom_video',methods=['post'])
def create_video_with_defaultVoice():
    try:
        # 요청 데이터 확인
        data = request.get_json()
        print(f"받은 데이터 타입: {type(data)}")
        print(f"받은 데이터 내용: {data}")
        
        # 데이터의 모든 키 출력
        if isinstance(data, dict):
            print(f"데이터 키: {list(data.keys())}")
        
        # 스프링 백엔드에서 보내는 데이터 구조에 맞게 처리
        title = data.get("title")
        
        # expressionUrlMap 객체 확인
        expression_map = data.get("expressionUrlMap")
        print(f"expressionUrlMap: {expression_map}")
        
        # 표정 이미지 URL - 두 가지 방식으로 시도
        if expression_map and isinstance(expression_map, dict):
            # expressionUrlMap 객체에서 추출
            happy_image_url = expression_map.get("HAPPY")
            sad_image_url = expression_map.get("SAD")
            angry_image_url = expression_map.get("ANGRY")
            surprised_image_url = expression_map.get("SURPRISED")
        else:
            # 루트 레벨에서 직접 가져옴
            happy_image_url = data.get("happyImageURL")
            sad_image_url = data.get("sadImageURL")
            angry_image_url = data.get("angryImageURL")
            surprised_image_url = data.get("surprisedImageURL")
        
        # 오디오 경로 (커스텀 음성용)
        audio_path = data.get("audioPath")
        
        print(f"파일명 = {title}, 오디오 경로 = {audio_path}")
        print(f"표정 이미지 URL: HAPPY={happy_image_url}, SAD={sad_image_url}, ANGRY={angry_image_url}, SURPRISED={surprised_image_url}")

        # 이미지 URL이 없는 경우 처리
        if not any([happy_image_url, sad_image_url, angry_image_url, surprised_image_url]):
            return jsonify({
                "success": False,
                "message": "표정 이미지 URL이 제공되지 않았습니다."
            }), 400
        
        # 이미지 경로 검증을 위한 데이터 재구성
        validation_data = {
            "happyImageURL": happy_image_url,
            "sadImageURL": sad_image_url,
            "angryImageURL": angry_image_url,
            "surprisedImageURL": surprised_image_url
        }
        
        # 이미지 경로 검증
        errors = validate_image_paths(validation_data)
        if errors:
            return jsonify({
                "success": False,
                "message": "파일 검증 실패",
                "errors": errors
            }), 400
        
        # 입력 비디오 경로 설정 - 타이틀에 따라 다른 영상 선택
        if "금도끼" in title or "은도끼" in title:
            input_video_path = os.path.normpath("C:/Users/winte/Desktop/uploads/videos/default/axe.mp4")
            print(f"금도끼 은도끼 영상 선택: {input_video_path}")
        elif "아낌없이" in title or "나무" in title:
            input_video_path = os.path.normpath("C:/Users/winte/Desktop/uploads/videos/default/tree.mp4")
            print(f"아낌없이 주는 나무 영상 선택: {input_video_path}")
        else:
            # 기본 영상 (금도끼 은도끼)
            input_video_path = os.path.normpath("C:/Users/winte/Desktop/uploads/videos/default/axe.mp4")
            print(f"기본 영상 선택: {input_video_path}")
        
        # 출력 디렉토리 설정 - 경로 일관성을 위해 normpath 사용
        output_dir = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], "videos/custom"))
        os.makedirs(output_dir, exist_ok=True)
        
        # 출력 파일명 생성 - UUID 전체 사용
        unique_id = str(uuid.uuid4())  # UUID 전체 사용
        output_filename = f"{unique_id}_{title}.mp4"
        output_path = os.path.normpath(os.path.join(output_dir, output_filename))
        
        print(f"업로드 폴더: {app.config['UPLOAD_FOLDER']}")
        print(f"출력 디렉토리: {output_dir}")
        print(f"출력 경로: {output_path}")
        print(f"생성된 파일명: {output_filename}")
        
        # 애노테이션 JSON 사용 - 제공된 파일 활용
        json_path = os.path.normpath(os.path.join(os.getcwd(), "annotation.json"))
        if not os.path.exists(json_path):
            return jsonify({
                "success": False,
                "message": "애노테이션 파일이 존재하지 않습니다."
            }), 400
        else:
            print(f"기존 annotation.json 파일을 사용합니다: {json_path}")
        
        # 표정 이미지 경로 설정 - 사용자별 디렉토리 생성
        expressions_dir = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], "expressions", unique_id))
        os.makedirs(expressions_dir, exist_ok=True)
        
        # 이미지 URL에서 파일 경로 추출 및 복사
        # 스프링 백엔드의 FacialExpression enum과 일치시킴
        # 애노테이션 파일에서 사용하는 표현식 이름: base, sad, mad, smile
        image_mapping = {
            "base": happy_image_url,  # 기본적으로 happy 이미지를 base로 사용
            "smile": happy_image_url,  # happy -> smile로 매핑
            "sad": sad_image_url,
            "mad": angry_image_url,   # angry -> mad로 매핑
            "surprised": surprised_image_url  # surprised는 그대로 (애노테이션에서 사용하지 않음)
        }
        
        for expr, url in image_mapping.items():
            if url:
                # URL에서 파일 경로 추출
                relative_path = url.lstrip('/')
                
                # uploads가 중복되지 않도록 처리
                if relative_path.startswith('uploads/'):
                    relative_path = relative_path.replace('uploads/', '', 1)
                
                src_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], relative_path))
                print(f"복사: {expr} -> {src_path}")
                
                # 표정 디렉토리에 복사
                dst_path = os.path.normpath(os.path.join(expressions_dir, f"{expr}.jpg"))
                shutil.copy(src_path, dst_path)
        
        # fallback 이미지 설정 - base 이미지를 fallback으로 사용
        fallback_img = os.path.normpath(os.path.join(expressions_dir, "base.jpg"))
        if not os.path.exists(fallback_img):
            # base 이미지가 없으면 smile(happy) 이미지를 사용
            fallback_img = os.path.normpath(os.path.join(expressions_dir, "smile.jpg"))
            if not os.path.exists(fallback_img):
                return jsonify({
                    "success": False,
                    "message": "기본 표정 이미지가 없습니다."
                }), 400
        
        # 비디오 처리 함수 호출
        result_path = customForVideo.process_video(
            video_path=input_video_path,
            json_path=json_path,
            expressions_dir=expressions_dir,
            fallback_img=fallback_img,
            output_path=output_path
        )
        
        # 상대 URL 경로 생성 - 스프링 백엔드에서 기대하는 형식으로 변환
        # URL은 항상 슬래시(/)를 사용하도록 함
        video_url = f"/uploads/videos/custom/{output_filename}".replace("\\", "/")
        
        # 스프링 백엔드의 CreateCustomVideoResponse 클래스와 필드명 일치
        return jsonify({
            "videoName": output_filename,
            "videoURL": video_url
        }), 200
        
    except Exception as e:
        print(f"예외 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"비디오 처리 중 오류 발생: {str(e)}"
        }), 500

def validate_image_paths(data):
    errors = []
    
    # 스프링 백엔드의 FacialExpression enum과 일치시킴
    required_keys = ['happyImageURL', 'sadImageURL', 'angryImageURL', 'surprisedImageURL']
    
    # 필수 표정 이미지 검증 - 최소 하나의 표정 이미지는 있어야 함
    has_any_expression = any(data.get(key) for key in required_keys)
    if not has_any_expression:
        errors.append("최소 하나 이상의 표정 이미지가 필요합니다.")
        return errors
    
    # 제공된 표정 이미지 검증
    for key in required_keys:
        url_path = data.get(key)
        if not url_path:
            # 필수가 아니므로 누락되어도 오류로 처리하지 않음
            continue
            
        # URL 경로에서 상대 경로 추출
        relative_path = url_path.lstrip('/')
        
        # uploads가 중복되지 않도록 처리
        if relative_path.startswith('uploads/'):
            relative_path = relative_path.replace('uploads/', '', 1)
        
        # 절대 경로 생성
        absolute_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], relative_path))
        
        print(f"검증 중: {key} -> {absolute_path}")
        
        # 파일 존재 여부 확인
        if not os.path.isfile(absolute_path):
            errors.append(f"{key} 파일이 존재하지 않습니다: {absolute_path}")
    
    return errors

if __name__ == "__main__":
    print(f"서버 시작: 설정된 업로드 폴더 = {app.config['UPLOAD_FOLDER']}")
    app.run(host='0.0.0.0', port=5001, debug=True)