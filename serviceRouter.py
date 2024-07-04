from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
import json
from pydub import AudioSegment
import os
import logging
import shutil
from model import makeWavFile, make_initial_wav_file, extract_unique_pitches  # makeWavFile 함수를 이 파일에서 가져옴

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()



@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    UPLOAD_DIRECTORY = "./upload/"
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    # 임시 파일 경로 생성
    temp_file_path = os.path.join(UPLOAD_DIRECTORY, f"temp_{file.filename}")
    
    try:
        # 업로드된 파일을 임시 파일로 저장합니다.
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Temporary file saved: {temp_file_path}")

        # 임시 파일을 WAV 형식으로 변환합니다.
        wav_file_path = os.path.splitext(file_path)[0] + ".wav"
        sound = AudioSegment.from_file(temp_file_path)
        sound.export(wav_file_path, format="wav")
        
        logger.info(f"WAV file created: {wav_file_path}")

        # 임시 파일 삭제
        os.remove(temp_file_path)
        logger.info(f"Temporary file removed: {temp_file_path}")
        
        return {"filename": file.filename, "wav_filename": os.path.basename(wav_file_path)}
    
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        # 오류 발생 시 임시 파일 삭제 시도
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise

@router.get("/audio")
def generate_audio():
    try:
        makeWavFile()  # WAV 파일 생성
        file_path = "output_new3.wav"  # 생성된 파일 경로
        return FileResponse(path=file_path, filename="output_new3.wav", media_type='audio/wav')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    

    
@router.get("/pitch")
def get_metadata():
    make_initial_wav_file()
    midi_file_path = 'drums_2bar_oh_lokl_sample_0.mid'
    unique_pitches = extract_unique_pitches(midi_file_path)  # 이 함수의 구현 코드가 필요합니다.
    return JSONResponse({"unique_pitches": unique_pitches})

@router.get("/wav")
def download_file():
    wav_file_path = 'drums_2bar_oh_lokl_sample_0.wav'
    if os.path.exists(wav_file_path):
        with open(wav_file_path, 'rb') as file:
            wav_data = file.read()
        headers = {'Content-Disposition': 'attachment; filename="drums_2bar_oh_lokl_sample_0.wav"'}
        return Response(content=wav_data, headers=headers, media_type="audio/wav")
    else:
        return HTTPException(status_code=404, detail="File not found")
