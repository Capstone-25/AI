from pymongo import MongoClient
from datetime import datetime

# MongoDB 연결 설정
client = MongoClient("mongodb+srv://j2982477:EZ6t7LEsGEYmCiJK@mindai.zgcb4ae.mongodb.net/?retryWrites=true&w=majority&appName=mindAI"
"Current Mongosh Log ID: 67ed3148e753153a041aecb5")
db = client['mindAI']  # 'mindAI' 데이터베이스 사용
chat_collection = db['chat_logs']  # 'chat_logs' 컬렉션 사용
user_collection = db['users']  # 사용자 정보 저장을 위한 컬렉션

# 채팅 로그 저장 함수
def save_chat_log(user_id, chat_id, user_message, bot_response):
    """
    사용자의 메시지와 챗봇의 응답을 채팅 로그에 저장
    """ 
    # 현재 시간
    timestamp = datetime.now().isoformat()

    # 채팅 로그 데이터
    chat_log_entry = {
        "user_id": user_id,
        "chat_id": chat_id,
        "timestamp": timestamp,
        "messages": [
            {
                "role": "user",
                "message": user_message,
                "timestamp": timestamp
            },
            {
                "role": "bot",
                "message": bot_response,
                "timestamp": timestamp
            }
        ]
    }

    # MongoDB에 채팅 로그 저장 (업데이트 혹은 새로 추가)
    chat_collection.update_one(
        {"chat_id": chat_id},  # chat_id가 존재하는지 확인
        {"$push": {"messages": {"role": "user", "message": user_message, "timestamp": timestamp}}},
        upsert=True  # chat_id가 없다면 새로 생성
    )
    
    chat_collection.update_one(
        {"chat_id": chat_id},
        {"$push": {"messages": {"role": "bot", "message": bot_response, "timestamp": timestamp}}},
        upsert=True  # chat_id가 없다면 새로 생성
    )

    print(f"Chat log for chat_id {chat_id} has been saved successfully!")

# 채팅 로그 불러오기 함수
def get_chat_log(chat_id):
    """
    특정 chat_id에 대한 채팅 로그를 불러옴
    """
    chat_log = chat_collection.find_one({"chat_id": chat_id})
    if chat_log:
        return chat_log['messages']  # 메시지 목록 반환
    else:
        return None
"""
사용자 정보는 실제 서비스 시에는 백엔드 서버로부터 전달받을 예정, 현재는 임시로저장하는거
"""

# 사용자 정보 저장 함수
def save_user_info(user_id, name, age, gender):
    """
    사용자 정보를 DB에 저장
    """
    user_info = {
        "user_id": user_id,
        "name": name,
        "age": age,
        "gender": gender
    }
    user_collection.update_one(
        {"user_id": user_id},  # user_id가 존재하는지 확인
        {"$set": user_info},    # 사용자 정보 업데이트
        upsert=True  # user_id가 없다면 새로 추가
    )
    print(f"User info for {user_id} has been saved successfully!")

# 사용자 정보 불러오기 함수
def get_user_info(user_id):
    """
    특정 user_id에 대한 사용자 정보를 불러옴
    """
    user_info = user_collection.find_one({"user_id": user_id})
    if user_info:
        return user_info  # 사용자 정보 반환
    else:
        return None