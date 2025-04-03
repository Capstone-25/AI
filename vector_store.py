# 사용안하는 코드

import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from config import USER_ID, VECTOR_DIR

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

faiss_path = VECTOR_DIR / f"{USER_ID}.faiss"
meta_path = VECTOR_DIR / f"{USER_ID}.json"

# 저장소 로딩
if faiss_path.exists() and meta_path.exists():
    index = faiss.read_index(str(faiss_path))
    with open(meta_path, "r") as f:
        metadata = json.load(f)
else:
    index = faiss.IndexFlatL2(384)
    metadata = []

def save_summary(text, emotion_list):
    vec = embed_model.encode([text])
    index.add(np.array(vec).astype("float32"))
    metadata.append({
        "summary": text,
        "emotion": emotion_list,
        "timestamp": datetime.now().isoformat()
    })
    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

def get_metadata():
    return metadata