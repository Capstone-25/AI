from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import os

class CounselorAgent:
    def __init__(self, client_info,  total_strategy, persona_type, emotion, distortion=None, model_name="gpt-4o-mini", temperature=0.7):
        self.client_info = client_info
        self.total_strategy = total_strategy  # 'cbt_strategy' 대신 'total_strategy'
        self.emotion = emotion  
        self.distortion = distortion
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Load persona prompt based on persona_type
        persona_path = f"prompts/{persona_type}.txt"
        with open(persona_path, "r", encoding="utf-8") as f:
            self.persona_prompt = f.read()

        self.prompt_template = self.load_prompt_template()

    def load_prompt_template(self):
        with open("prompts/counselor_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    

        
    # 감정과 인지 왜곡에 맞는 전략 결합
        return f"{emotion_strategy} {distortion_strategy}"
    
    def generate_response(self, history):
        # history를 문자열로 변환
        formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['message']}" for msg in history])

        filled_prompt = self.prompt_template.format(
            client_info=self.client_info,
            history=formatted_history,  # 변환된 문자열 사용
            total_strategy=self.total_strategy,
            persona_prompt=self.persona_prompt,
            emotion=self.emotion,
            distortion=self.distortion
        )

        response = self.llm.invoke(filled_prompt)

        if isinstance(response, AIMessage):
            return response.content 
        return str(response)
