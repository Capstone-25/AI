# CapStone_Chat
2025-1학기 세종대학교 캡스톤 "심리상담 서비스" AI파트


CapStone/  
├── chat.py                     # 💬 전체 상담 세션 실행 메인 루프   
├── config.py                   # ⚙️ 경로, 설정값 등 공통 설정   
├── prompt_builder.py           # 🧾 시스템 프롬프트 동적 생성기  
├── emotion_utils.py            # 😌 감정 분석 및 감정 추출 유틸   
├── persona_prompts.py          # 🧠 페르소나 프롬프트 구성 및 조합  
├── vector_store.py             # 💾 벡터 저장소 관련 함수 (ex. Retrieval)
│  
├── agents/                     # 🤖 에이전트(모델 역할자) 모듈    
│   ├── client_agent.py         # 내담자 역할 에이전트    
│   ├── counselor_agent.py      # 상담자 역할 에이전트 (서비스용)   
│   ├── evaluator_agent.py      # 평가자 역할 에이전트 (내부 평가용)  
│   └── __init__.py  
│  
├── prompts/                    # 📝 모든 에이전트용 프롬프트 텍스트 파일    
│   ├── agent_client.txt  
│   ├── agent_counselor.txt  
│   ├── eval_guided_discovery.txt  
│   ├── eval_focus.txt  
│   ├── eval_strategy.txt  
│   ├── eval_understanding.txt  
│   ├── eval_collaboration.txt  
│   ├── eval_empathy.txt  
│   └── ...  
│  
├── data/                       # 🗂️ 대화 샘플, 감정 분석 결과 등 저장 위치  
│   ├── user_profiles.json      # 내담자 정보 샘플 
│   ├── evaluation_data.json    # 자동 생성 대화 + 평가용 데이터   
│   └── ...       
│   
├── results/                    # 📊 평가 결과 저장  
│   ├── session_logs/           # 1:1 대화 기록 저장  
│   ├── emotion_results.json    # 감정 평가 결과 
│   ├── evaluation_scores.json  # 평가자 채점 결과    
│   └── ...  
│  
├── tests/                      # 🧪 유닛 테스트 및 성능 검증  
│   └── test_agents.py   
│  
└── README.md                   # 📘 프로젝트 설명서  