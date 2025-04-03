"""
실행 코드
python chat.py --output_file results/result1.json --persona_type persona_20s_friend --chat_id chat101 --user_id user1234
"""

import json
from pathlib import Path
from agents.client_agent import ClientAgent
from agents.counselor_agent import CounselorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.sub_llm import SubLLMAgent
from config import get_config, set_openai_api_key
from cbt.cbt_mappings import emotion_strategies, cognitive_distortion_strategies
from DB import get_chat_log, save_chat_log, save_user_info, get_user_info

# --- STT 관련 모듈 추가 ---
import whisper
import sounddevice as sd
import numpy as np
import wave
import keyboard  # pip install keyboard
import time

# 마이크로 Enter 누르면 녹음 시작, 다시 Enter 누르면 녹음 종료
def record_audio_keypress(filename="input.wav", fs=16000):
    print("⏺️ Enter 키를 누르면 녹음 시작, 다시 누르면 종료 (2초 이상 누르면 챗봇 종료)")

    # Enter 눌러서 시작
    while True:
        if keyboard.read_key() == "enter":
            print("🎙️ 녹음 중... 다시 Enter 누르면 종료")
            break

    recording = []
    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16')
    stream.start()

    # 종료 감지
    while True:
        if keyboard.is_pressed("enter"):
            start_time = time.time()
            while keyboard.is_pressed("enter"):
                time.sleep(0.1)
                if time.time() - start_time >= 2.0:
                    print("🛑 Enter 키 2초 이상 누름 → 챗봇 종료")
                    stream.stop()
                    stream.close()
                    return "exit"

            print("⏹️ 녹음 종료")
            break

        data, _ = stream.read(1024)
        recording.append(data)

    stream.stop()
    stream.close()

    audio_data = np.concatenate(recording)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

    print("✅ 저장 완료:", filename)
    return "ok"

# Whisper로 음성을 텍스트로 변환
def whisper_transcribe(filename="input.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename, language='ko')
    return result["text"]

# OpenAI API 키 설정
set_openai_api_key()

# MongoDB 연결
from pymongo import MongoClient
client = MongoClient("mongodb+srv://j2982477:EZ6t7LEsGEYmCiJK@mindAI.zgcb4ae.mongodb.net/?retryWrites=true&w=majority&appName=mindAI")
db = client['mindAI']


# TherapySimulation 클래스
class TherapySimulation:
    def __init__(self, persona_type: str, chat_id: str, user_id: str, max_turns: int = 20):
        self.persona_type = persona_type
        self.chat_id = chat_id
        self.user_id = user_id
        self.max_turns = max_turns
        self.history = []

        user_info = get_user_info(self.user_id)
        if user_info:
            self.name = user_info["name"]
            self.age = user_info["age"]
            self.gender = user_info["gender"]
        else:
            print(f"{self.user_id}는 새로운 사용자입니다. 사용자 정보를 입력해주세요.")
            self.name = input("이름을 입력해주세요: ")
            self.age = int(input("나이를 입력해주세요: "))
            self.gender = input("성별을 입력해주세요: ")
            save_user_info(self.user_id, self.name, self.age, self.gender)

        chat_log = get_chat_log(self.chat_id)
        if chat_log:
            self.history = chat_log
        else:
            self.history.append({
                "role": "client",
                "message": f"{self.name}님, 안녕하세요. 어떤 문제가 있으신가요?"
            })

        self.subllm_agent = SubLLMAgent()
        self.evaluator_agent = EvaluatorAgent(criteria_list=["general_1", "general_2", "general_3", "cbt_1", "cbt_2", "cbt_3"])
        self.counselor_agent = CounselorAgent(
            client_info=f"{self.name}, {self.age}세, {self.gender}",
            total_strategy="",
            persona_type=persona_type,
            emotion="",
            distortion=""
        )
        self._init_history()

    def _init_history(self):
        if not self.history:
            self.history.append({
                "role": "client",
                "message": f"{self.name}님, 안녕하세요. 어떤 문제가 있으신가요?"
            })

    def run(self):
        mode = input("모드를 선택하세요 (text/voice): ").strip().lower()
        if mode not in ["text", "voice"]:
            mode = "text"

        for turn in range(self.max_turns):
            print(f"--- Turn {turn + 1} ({mode.upper()} MODE) ---")

            # 상담자 응답 생성
            counselor_msg = self.counselor_agent.generate_response(self.history)
            self.history.append({"role": "counselor", "message": counselor_msg})
            print("Counselor:", counselor_msg)

            # 사용자 입력 처리
            if mode == "voice":
                result = record_audio_keypress("input.wav", language="ko", fp16=False)  # 수정된 함수는 "ok" 또는 "exit" 반환
                if result == "exit":
                    print("🛑 Enter 키를 2초 이상 눌러 챗봇을 종료합니다.")
                    break
                client_msg = whisper_transcribe("input.wav").strip()
                print(f"{self.name}: {client_msg}")
            else:
                client_msg = input(f"{self.name}: ").strip()

            # 종료 명령어 포함 여부 확인
            if any(keyword in client_msg.lower() for keyword in ["종료", "끝내자", "그만할래"]):
                print("🛑 종료 명령어 감지됨. 대화를 종료합니다.")
                self.history.append({"role": "client", "message": client_msg})
                break

            self.history.append({"role": "client", "message": client_msg})

            # 감정/인지왜곡 분석
            analysis_result = self.subllm_agent.analyze(client_msg)
            emotion = analysis_result.get("감정", "")
            distortion = analysis_result.get("인지왜곡", "")
            total_strategy = analysis_result.get("총합_CBT전략", "")

            print(f"Emotion detected: {emotion}")
            print(f"Cognitive Distortion detected: {distortion}")
            print(f"CBT Strategy: {total_strategy}")
            print()

            # 상담자 agent 업데이트
            self.counselor_agent = CounselorAgent(
                client_info=f"{self.name}, {self.age}세, {self.gender}성",
                total_strategy=total_strategy,
                persona_type=self.persona_type,
                emotion=emotion,
                distortion=distortion
            )

            # 로그 저장
            save_chat_log(self.user_id, self.chat_id, client_msg, counselor_msg)

        # 평가
        evaluation_result = self.evaluator_agent.evaluate_all(self.history)
        return {
            "persona": self.persona_type,
            "cbt_strategy": total_strategy,
            "cognitive_distortion": distortion,
            "emotion": emotion,
            "history": self.history,
            "evaluation": evaluation_result
        }


# 실행
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--persona_type", required=True)
    parser.add_argument("--chat_id", required=True)
    parser.add_argument("--user_id", required=True)
    args = parser.parse_args()

    sim = TherapySimulation(
        persona_type=args.persona_type,
        chat_id=args.chat_id,
        user_id=args.user_id
    )
    result = sim.run()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
