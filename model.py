import glob
import os
import numpy as np
import tensorflow.compat.v1 as tf
import magenta.music as mm
import librosa
import soundfile as sf
from pydub import AudioSegment
import pyloudnorm as pyln
import note_seq
import pretty_midi
import pydub
import random
from collections import defaultdict
from models import drums_models
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# oneDNN 최적화 비활성화
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 라이브러리 불러오기 및 환경 설정

# TensorFlow 1.x 비활성화
tf.disable_v2_behavior()


def makeWavFile():

    adjust_loudness_and_highlight()
    
    # MIDI 파일 불러오기
    midi_data = pretty_midi.PrettyMIDI('/home/s20191048/dl/final/music_vae/dlma/drums_2bar_oh_lokl_sample_0.mid')

    # 빈 AudioSegment 생성 (총 10000ms로 설정)
    audio_segment = pydub.AudioSegment.silent(duration=10000)
    
    # MIDI 이벤트 순회
    for instrument in midi_data.instruments:
    # 시작 시간별로 노트를 그룹화
        notes_by_start = group_notes_by_start_time(instrument.notes)
        for start_time, notes in sorted(notes_by_start.items()):
        # 각 시작 시간별로 병합할 AudioSegment (125ms로 설정)
            merged = pydub.AudioSegment.silent(duration=250)

            for note in notes:
                try:                    
                        # wave_file = pydub.AudioSegment.from_wav(f"/home/s20191048/dl/final/music_vae/samples1/note_{note.pitch}.wav")
                    wave_file = pydub.AudioSegment.from_wav(f"/home/s20191048/dl/final/music_vae/dlma/upload_highlighted/note_{note.pitch}.wav")
                    note_duration_ms = 250
                    trimmed_wave = wave_file[:note_duration_ms]

                    fade_in_duration = note_duration_ms // 4  # 노트 지속시간의 1/4
                    fade_out_duration = note_duration_ms // 2  # 노트 지속시간의 1/2

                    faded_wave = trimmed_wave.fade_in(duration=fade_in_duration).fade_out(duration=fade_out_duration)
                    # 음량을  줄이도록 페이드 아웃 적용

                    merged = merged.overlay(faded_wave)
                except Exception as e:
                    print(f"Error processing note {note.pitch} at {start_time}s: {e}")

                # 병합된 AudioSegment 오버레이
            position_ms = int(start_time * 1000)
            audio_segment = audio_segment.overlay(merged, position=position_ms)

    # 결과 AudioSegment 저장        
    audio_segment.export("output_new3.wav", format="wav")
    
def download_wav(note_sequence, filename):
    download_sequence(note_sequence, filename, synth=note_seq.fluidsynth)

def download_sequence(sequence,
                      output_wav_path,
                      synth=note_seq.fluidsynth,
                      sample_rate=44100,
                      colab_ephemeral=True,
                      **synth_args):
    """Synthesizes a note sequence and saves it as a .wav file using pydub.

    Args:
        sequence: A music_pb2.NoteSequence to synthesize and save.
        synth: A synthesis function that takes a sequence and sample rate as input.
        sample_rate: The sample rate at which to synthesize.
        output_wav_path: The path where the output .wav file will be saved.
        **synth_args: Additional keyword arguments to pass to the synth function.
    """
    array_of_floats = synth(sequence, sample_rate=sample_rate, **synth_args)
    
     # Convert numpy float array to int16
    int16_array = np.int16(array_of_floats * 32767)

    # numpy array to AudioSegment
    audio_segment = AudioSegment(
        int16_array.tobytes(), 
        frame_rate=sample_rate,
        sample_width=2,  # 2 bytes for 16-bit audio
        channels=1
    )

    # Export the audio segment to a wav file
    audio_segment.export(output_wav_path, format="wav")
    print(f"Saved synthesized audio to {output_wav_path}")


def make_initial_wav_file():
    drums_sample_model = "drums_2bar_oh_lokl" #@param ["drums_2bar_oh_lokl", "drums_2bar_oh_hikl", "drums_2bar_nade_reduced", "drums_2bar_nade_full"]
    temperature = 0.5 #@param {type:"slider", min:0.1, max:1.5, step:0.1}
    drums_sample = drums_models[drums_sample_model].sample(n=1, length=256, temperature=temperature)
    
    for i, ns in enumerate(drums_sample):
        download(ns, '%s_sample_%d.mid' % (drums_sample_model, i))
        download_wav(ns, '%s_sample_%d.wav' % (drums_sample_model, i))

def extract_unique_pitches(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    pitches = set()

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitches.add(note.pitch)

    return sorted(pitches)

def adjust_loudness_and_highlight():

    input_directory = '/home/s20191048/dl/final/music_vae/dlma/upload'
    output_directory = '/home/s20191048/dl/final/music_vae/dlma/upload_adjust_loudness'

    adjust_loudness_for_wav_files(input_directory, output_directory)
    
    input_directory = '/home/s20191048/dl/final/music_vae/dlma/upload_adjust_loudness'
    output_directory = '/home/s20191048/dl/final/music_vae/dlma/upload_highlighted'

    adjust_highlight_for_wav_files(input_directory, output_directory)
# Function to detect highlights using a simple amplitude-based method
def detect_highlight(y, sr, window_size=1):
    hop_length = int(sr * window_size / 2)
    max_rms = 0
    highlight_start = 0
    
    for i in range(0, len(y), hop_length):
        window = y[i:i + hop_length]
        rms = np.sqrt(np.mean(window**2))
        
        if rms > max_rms:
            max_rms = rms
            highlight_start = i
            
    highlight_end = highlight_start + hop_length
    return highlight_start, highlight_end

def find_runs(x):
    """Find runs of consecutive items in an array."""
    if len(x) == 0:
        return np.array([])
    
    # Find the indices where the array changes
    changes = np.diff(x.astype(int))
    change_points = np.where(changes != 0)[0] + 1
    
    # Include the start and end of the array
    change_points = np.concatenate(([0], change_points, [len(x)]))
    
    # Pair up start and end points
    runs = np.column_stack((change_points[:-1], change_points[1:]))
    
    # Only return runs where x is True
    return runs[x[runs[:, 0]]]

def detect_highlight_improved(y, sr, min_duration=0.5, max_duration=3.0):
    # 스펙트로그램 계산
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 평균 에너지 계산 및 임계값 설정
    mean_energy = np.mean(S_db)
    threshold = mean_energy + 0.5 * np.std(S_db)  # 임계값 조정 가능
    
    # 프레임별 에너지 계산
    frame_energy = np.mean(S_db, axis=0)
    
    # 중요 구간 찾기
    important_frames = frame_energy > threshold
    
    # 모든 프레임이 임계값 이하인 경우 처리
    if not np.any(important_frames):
        print("Warning: No frames above threshold found. Using entire audio.")
        return 0, len(y)
    
    # 연속된 중요 구간 찾기
    ranges = find_runs(important_frames)
    
    if len(ranges) == 0:
        print("Warning: No continuous important segments found. Using entire audio.")
        return 0, len(y)
    
    # 가장 긴 중요 구간 선택
    longest_range = max(ranges, key=lambda x: x[1] - x[0])
    start_frame, end_frame = longest_range
    
    # 시간으로 변환
    start_time = librosa.frames_to_time(start_frame, sr=sr)
    end_time = librosa.frames_to_time(end_frame, sr=sr)
    
    # 최소 및 최대 지속 시간 적용
    duration = end_time - start_time
    if duration < min_duration:
        end_time = min(start_time + min_duration, len(y) / sr)
    elif duration > max_duration:
        end_time = min(start_time + max_duration, len(y) / sr)
    
    return int(start_time * sr), min(int(end_time * sr), len(y))

def download(note_sequence, filename):
    mm.sequence_proto_to_midi_file(note_sequence, filename)
    print(f'MIDI file saved as {filename}')
    

def adjust_highlight_for_wav_files(input_directory, output_directory):
    """
    입력 디렉토리의 모든 WAV 파일에서 하이라이트를 감지하고 조정하여 출력 디렉토리에 저장하는 함수.
    
    :param input_directory: 입력 디렉토리 경로
    :param output_directory: 출력 디렉토리 경로
    """
    if not os.path.exists(output_directory):  # 출력 디렉토리가 존재하지 않으면
        os.makedirs(output_directory)  # 출력 디렉토리 생성

    for filename in os.listdir(input_directory):  # 입력 디렉토리의 모든 파일에 대해
        if filename.endswith('.wav'):  # 파일이 .wav 확장자를 가지면
            file_path = os.path.join(input_directory, filename)  # 파일의 전체 경로 생성
            
            # librosa를 사용하여 오디오 로드
            y, sr = librosa.load(file_path, sr=None)  # mono=True로 단일 채널 로드

            highlight_start, highlight_end = detect_highlight_improved(y, sr)
            
            # 0.25초 길이의 하이라이트 추출
            highlight_duration = int(0.25 * sr)
            highlight_end = min(highlight_start + highlight_duration, len(y))
            highlight_y = y[highlight_start:highlight_end]
            
            # 하이라이트 길이가 0.25초보다 짧으면 패딩 추가
            if len(highlight_y) < highlight_duration:
                padding = np.zeros(highlight_duration - len(highlight_y))
                highlight_y = np.concatenate((highlight_y, padding))
            
            # 하이라이트 오디오를 새로운 파일로 내보내기
            output_path = os.path.join(output_directory, filename)  # 출력 파일 경로 생성
            sf.write(output_path, highlight_y, sr)  # 사운드파일 라이브러리를 사용하여 파일 저장

    print(f"처리 완료. 결과물은 {output_directory}에 저장되었습니다.")        
def calculate_loudness(sound):
    meter = pyln.Meter(sound.frame_rate)
    samples = np.array(sound.get_array_of_samples())
    # float64로 변환 후 정규화
    samples = samples.astype(np.float64) / np.iinfo(samples.dtype).max
    
    if sound.channels == 1:
        # 모노를 스테레오로 변환
        samples = np.column_stack((samples, samples))
    else:
        samples = samples.reshape(-1, sound.channels)
    
    loudness = meter.integrated_loudness(samples)
    return loudness

def match_target_loudness(sound, target_loudness):
    meter = pyln.Meter(sound.frame_rate)
    samples = np.array(sound.get_array_of_samples())
    # float64로 변환 후 정규화
    samples = samples.astype(np.float64) / np.iinfo(samples.dtype).max
    
    if sound.channels == 1:
        # 모노를 스테레오로 변환
        samples = np.column_stack((samples, samples))
    else:
        samples = samples.reshape(-1, sound.channels)
    
    loudness = meter.integrated_loudness(samples)
    loudness_difference = target_loudness - loudness
    return sound.apply_gain(loudness_difference)

def adjust_loudness_for_wav_files(input_directory, output_directory, target_loudness=-5.0):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_directory, filename)
            try:
                sound = AudioSegment.from_file(file_path, format="wav")
                
                print(f"처리 중: {filename}")
                print(f"채널 수: {sound.channels}, 프레임 레이트: {sound.frame_rate}, 길이: {len(sound)}ms")
                
                original_loudness = calculate_loudness(sound)
                print(f"원본 음량: {original_loudness:.2f} LUFS")
                
                # 처음 0.125초 부분 제거
                trimmed_sound = sound[125:]
                
                normalized_sound = match_target_loudness(trimmed_sound, target_loudness)
                
                output_path = os.path.join(output_directory, filename)
                normalized_sound.export(output_path, format="wav")
                
                adjusted_loudness = calculate_loudness(normalized_sound)
                print(f"조정된 음량: {adjusted_loudness:.2f} LUFS")
                print("---")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {filename}")
                print(f"오류 메시지: {str(e)}")
                print("---")
                
# 노트를 시작 시간별로 그룹화하는 함수
def group_notes_by_start_time(notes):
    notes_by_start = defaultdict(list)
    for note in notes:
        notes_by_start[note.start].append(note)
    return notes_by_start


