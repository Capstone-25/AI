#해당 chat은 우리가 직접 상담자와 대화하면서 여러가지 응답을 테스트해보는 채팅 실행 코드입니다. 여기서는 내담자&평가자가 개입X
"""
python3 chat2.py --persona_type persona_20s_friend --chat_id chat123
"""
import json
from pathlib import Path
from pymongo import MongoClient
from agents.counselor_agent import CounselorAgent
from config import set_openai_api_key
from datetime import datetime

# API 키 설정
set_openai_api_key()

# MongoDB 연결 설정
client = MongoClient("mongodb+srv://j2982477:EZ6t7LEsGEYmCiJK@mindAI.zgcb4ae.mongodb.net/?retryWrites=true&w=majority&appName=mindAI")
db = client['mindAI']
chat_collection = db['chat_logs']  # 'chat_logs' 컬렉션 사용

class TherapySimulation:
    def __init__(self, persona_type: str, chat_id: str, max_turns: int = 20):
        self.persona_type = persona_type
        self.chat_id = chat_id
        self.max_turns = max_turns
        self.history = []

        # 내담자 정보 임의 설정 (사용자가 직접 입력하는 부분)
        self.client_info = {
            "name": "박정원",
            "age": 25,
            "gender": "남성",
            "emotion_state": "없음",
            "cognitive_distortion": "없음",
            "attitude": "중립",
        }

        # 상담자 에이전트
        self.counselor_agent = CounselorAgent(
            client_info=self.client_info,
            reason="부정적인 사고 때문에 러닝을 지속하기 어렵습니다.",
            cbt_technique="",  # 기법은 아직 사용 안함
            cbt_strategy="",
            persona_type=persona_type,
            emotion="",
            distortion=""
        )
        
        # 대화 기록 초기화
        self._init_history()

    def _init_history(self):
        self.history = []

    def _add_to_db(self):
        timestamp = datetime.now().isoformat()
        chat_log_entry = {
            "chat_id": self.chat_id,
            "timestamp": timestamp,
            "messages": self.history
        }
        chat_collection.update_one(
            {"chat_id": self.chat_id},
            {"$push": {"messages": {"role": "counselor", "message": self.history[-2]['message'], "timestamp": timestamp}}},
            upsert=True
        )
        chat_collection.update_one(
            {"chat_id": self.chat_id},
            {"$push": {"messages": {"role": "client", "message": self.history[-1]['message'], "timestamp": timestamp}}},
            upsert=True
        )
        print(f"Chat log for chat_id {self.chat_id} has been updated successfully!")

    def run(self):
        print("상담을 시작합니다. 종료하려면 'exit' 또는 '[END]'를 입력하세요.")
        for turn in range(self.max_turns):
            print(f"--- Turn {turn + 1} ---")

            # 1. 사용자(내담자) 메시지 입력 받기
            user_message = input("내담자: ")
            if user_message.lower() in ["exit", "[end]"]:
                print("상담을 종료합니다.")
                break

            self.history.append({"role": "client", "message": user_message})

            # 2. 상담자 응답 생성
            counselor_msg = self.counselor_agent.generate_response(self.history)
            self.history.append({"role": "counselor", "message": counselor_msg})
            print("상담자:", counselor_msg)

            # 3. DB에 채팅 기록 저장
            self._add_to_db()

        # 결과 반환
        return {
            "persona": self.persona_type,
            "history": self.history,
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_type", required=True)
    parser.add_argument("--chat_id", required=True)
    args = parser.parse_args()

    sim = TherapySimulation(args.persona_type, args.chat_id)
    result = sim.run()

    # 결과 파일로 저장
    output_file = f"results/{args.chat_id}_result.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)