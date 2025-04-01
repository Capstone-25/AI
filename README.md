# CapStone_Chat
2025-1학기 세종대학교 캡스톤 "심리상담 서비스" AI파트
CapStone/  
├── chat.py                     # 💬 전체 상담 세션 실행 메인 루프   
├── config.py                   # ⚙️ 경로, 설정값 등 공통 설정   
│  
├── agents/                     # 🤖 에이전트(모델 역할자) 모듈    
│   ├── client_agent.py         # 내담자 역할 에이전트    
│   ├── counselor_agent.py      # 상담자 역할 에이전트 (서비스용), main LLM   
│   ├── evaluator_agent.py      # 평가자 역할 에이전트 (내부 평가용)  
|   ├── cbt_agent.py            # 에이전트 AI(Assist RAG, 보조 LLM) 감정 소분류, 인지오류, CBT전략등  사용자의 정보를 파악해서 프롬포트 맞춤형 생성 
│  
├── prompts/                    # 📝 모든 에이전트용 프롬프트 텍스트 파일    
│   ├── agent_counselor.txt  
│   └── ...  
│  
├── data/                       # 🗂️ 대화 샘플, 감정 분석 결과 등 저장 위치  
│   ├── example1.json      # 내담자 정보및 시작대화
│   └── ...       
│   
├── results/                    # 📊 평가 결과 저장  
│   ├── emotion_results.json    # 감정 평가 결과 
│   ├── evaluation_scores.json  # 평가자 채점 결과    
│   └── ...  
│
├── logs/     
│   ├── chat_log.json           # 1:1 대화 기록 저장  
│
└── README.md                   # 📘 프로젝트 설명서  