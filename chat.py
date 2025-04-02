"""
실행 코드
python3 chat.py \
  --input_file data/example1.json \
  --output_file results/result1.json \
  --persona_type persona_20s_friend
"""

import json
from pathlib import Path
from agents.client_agent import ClientAgent
from agents.counselor_agent import CounselorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.sub_llm import SubLLMAgent
from config import get_config, set_openai_api_key
from cbt.cbt_mappings import emotion_strategies, cognitive_distortion_strategies
from DB import save_chat_log  # DB.py에서 함수 import

# API 키 설정
set_openai_api_key()
from pymongo import MongoClient

# 연결 문자열 사용
client = MongoClient("mongodb+srv://j2982477:EZ6t7LEsGEYmCiJK"
"@mindAI.zgcb4ae.mongodb.net/?retryWrites=true&w=majority&appName=mindAI")

# 'mindAI' 데이터베이스에 연결
db = client['mindAI']
class TherapySimulation:
    def __init__(self, example: dict, persona_type: str, max_turns: int = 20):
        self.example = example
        self.persona_type = persona_type
        self.max_turns = max_turns
        self.history = []
        self.metadata = get_config()

        self.client_agent = ClientAgent(example["AI_client"])

        # SubLLM 분석 결과 반영
        self.subllm_agent = SubLLMAgent()
        self.analysis_result = self.subllm_agent.analyze(example["AI_client"]["init_history"])

        # 감정과 인지 왜곡을 분석한 결과
        self.emotion_state = self.analysis_result.get("감정", "")
        self.cognitive_distortion = self.analysis_result.get("인지왜곡", "")
        
        # 각 감정과 인지 왜곡에 대한 독립적인 CBT 전략
        self.emotion_cbt_strategy = emotion_strategies.get(self.emotion_state, "No emotion strategy detected")
        self.distortion_cbt_strategy = cognitive_distortion_strategies.get(self.cognitive_distortion, "No cognitive distortion strategy detected")
        
        # LLM 결과
        self.llm_raw_response = self.analysis_result.get("원본문", "")

        # 평가를 위한 EvaluatorAgent 추가
        self.criteria_list = ["general_1", "general_2", "general_3", "cbt_1", "cbt_2", "cbt_3"]
        self.evaluator_agent = EvaluatorAgent(criteria_list=self.criteria_list)

        # 상담자 에이전트에 전략 전달
        self.counselor_agent = CounselorAgent(
            client_info=example["AI_counselor"]["Response"]["client_information"],
            reason=example["AI_counselor"]["Response"]["reason_counseling"],
            cbt_technique="",  # 별도 필드 없다면 공란 처리
            cbt_strategy=self.emotion_cbt_strategy + " " + self.distortion_cbt_strategy,  # 감정과 인지 왜곡 전략을 결합하여 전달
            persona_type=persona_type,
            emotion=self.emotion_state,
            distortion=self.cognitive_distortion
        )

        self._init_history()

    def _init_history(self):
        init_counselor = self.example["AI_counselor"]["CBT"]["init_history_counselor"]
        init_client = self.example["AI_counselor"]["CBT"]["init_history_client"]
        self.history = [
            {"role": "counselor", "message": init_counselor},
            {"role": "client", "message": init_client},
        ]

    def run(self):
        for turn in range(self.max_turns):
            print(f"--- Turn {turn + 1} ---")

            # 1. 상담자 응답 생성
            counselor_msg = self.counselor_agent.generate_response(self.history)
            self.history.append({"role": "counselor", "message": counselor_msg})
            print("Counselor:", counselor_msg)

            # 2. 클라이언트 응답 생성
            client_msg = self.client_agent.generate(
                self.example["AI_client"]["intake_form"],
                self.example["AI_client"]["attitude_instruction"],
                "\n".join(f"{m['role'].capitalize()}: {m['message']}" for m in self.history)
            )
            self.history.append({"role": "client", "message": client_msg})
            print("Client:", client_msg)

            # 3. SubLLM 분석 (감정 및 인지 왜곡 탐지)
            analysis_result = self.subllm_agent.analyze(client_msg)
            emotion = analysis_result.get("감정", "")
            distortion = analysis_result.get("인지왜곡", "")
            emotion_cbt_strategy = emotion_strategies.get(emotion, "")
            distortion_cbt_strategy = cognitive_distortion_strategies.get(distortion, "")
            cbt_strategy = emotion_cbt_strategy + " " + distortion_cbt_strategy

            print(f"Emotion detected: {emotion}")
            print(f"Cognitive Distortion detected: {distortion}")
            print(f"CBT Strategy: {cbt_strategy}")
            print()

            # 4. 최신 분석 결과로 상담자 에이전트 재정의
            self.counselor_agent = CounselorAgent(
                client_info=self.example["AI_counselor"]["Response"]["client_information"],
                reason=self.example["AI_counselor"]["Response"]["reason_counseling"],
                cbt_technique="",  # 추후 기법 추가 가능
                cbt_strategy=cbt_strategy,
                persona_type=self.persona_type,
                emotion=emotion,
                distortion=distortion
            )
            # 5. 채팅 로그 저장
            save_chat_log("user123", "chat123", client_msg, counselor_msg)  # 채팅 로그를 MongoDB에 저장
            # 6. 종료 조건 체크
            if "[/END]" in client_msg:
                self.history[-1]["message"] = client_msg.replace("[/END]", "")
                break

        # 7. 평가 수행
        evaluation_result = self.evaluator_agent.evaluate_all(self.history)

        # 7. 결과 반환
        return {
            "persona": self.persona_type,
            "cbt_strategy": cbt_strategy,
            "cbt_technique": "",  # 아직 CBT 기법은 미사용
            "cognitive_distortion": distortion,
            "emotion": emotion,
            "history": self.history,
            "evaluation": evaluation_result
        }




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--persona_type", required=True)
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        example = json.load(f)

    sim = TherapySimulation(example, args.persona_type)
    result = sim.run()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
