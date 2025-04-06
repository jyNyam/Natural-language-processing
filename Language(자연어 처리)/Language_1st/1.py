import pandas as pd
import os
import librosa

# 설정
metadata_path = 'path/to/metadata.xlsx'
audio_dir = 'path/to/audio_files/'

# 1. 메타데이터 불러오기
metadata = pd.read_excel(metadata_path)

# 컬럼 구조를 확인 (실제 칼럼명을 기준으로)
# ['번호', '말뭉치 파일명', '음성 파일명', '주제(질문항목 번호)', '발화자 분류', 
# '발화자 ID', '나이(age)', '성별(sex)', '현 거주지(current_residence)', 
# '출생지(birthplace)', '학력(education)']

# 2. 오디오 경로 추가
def get_audio_path(file_name):
    # file_name에 확장자가 없을 경우 .wav 추가
    if not file_name.endswith('.wav'):
        file_name += '.wav'
    return os.path.join(audio_dir, file_name)

metadata['audio_path'] = metadata['음성 파일명'].apply(get_audio_path)

# 3. 오디오 로딩 및 정보 확인
for idx, row in metadata.iterrows():
    audio_file = row['audio_path']
    speaker_id = row['발화자 ID']
    age = row['나이(age)']
    sex = row['성별(sex)']
    topic = row['주제(질문항목 번호)']
    
    if os.path.exists(audio_file):
        audio, sr = librosa.load(audio_file, sr=None)
        print(f"[{idx}] 발화자 ID: {speaker_id}, 나이: {age}, 성별: {sex}, 주제: {topic}")
        print(f" -> {row['음성 파일명']} 로드 성공 (길이: {len(audio)}, 샘플링: {sr})")
    else:
        print(f"[경고] {audio_file} 파일 없음!")

