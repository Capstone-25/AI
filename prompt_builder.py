# 사용안하는 코드

from emotion_utils import emotion_keywords, emotion_guidelines, extract_recent_emotions, get_recent_summaries
from persona_prompts import personality_prompts

def build_system_prompt(metadata, persona_type, recent_n=10):
    top_emotions = extract_recent_emotions(metadata, emotion_keywords, recent_n)
    recent_summaries = get_recent_summaries(metadata)

    emotion_str = " / ".join(top_emotions)
    context_str = "\n".join(f"- {s}" for s in recent_summaries)
    guideline_str = "\n".join(
        f"- {e}: {emotion_guidelines[e]}" for e in top_emotions if e in emotion_guidelines
    )

    return f"""
[사용자 요약 정보]
- 최근 감정 키워드: {emotion_str}
- 최근 대화 요약:
{context_str}

[감정별 대응 가이드라인]
{guideline_str}

이 정보를 바탕으로 대화를 이어가 주세요.

{personality_prompts[persona_type]}
"""