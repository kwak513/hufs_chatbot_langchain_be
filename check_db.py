import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# PDF가 DB에 잘 저장됐는지 확인하기 위한 코드
# DB 접속 -> 컬렉션 찾기 -> 컬렉션 내의 문서 내용 확인

# 환경 변수
CONNECTION_STRING = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
COLLECTION_NAME = "homepage_pdfplumner_1st"

# PostgreSQL에서 벡터 DB에 들어간 문서 확인
try:
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    # 1. 컬렉션 이름으로 collection_uuid 조회
    cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (COLLECTION_NAME,))
    result = cur.fetchone()

    if not result:
        print(f"'{COLLECTION_NAME}' 컬렉션을 찾을 수 없습니다.")
    else:
        collection_uuid = result[0]
        # 2. 해당 collection_uuid를 가진 문서 조회
        cur.execute("SELECT document, cmetadata FROM langchain_pg_embedding WHERE collection_id = %s", (collection_uuid,))
        rows = cur.fetchall()
        print(f"'{COLLECTION_NAME}' 컬렉션에 저장된 문서(청크) 수: {len(rows)}")

        for i, row in enumerate(rows[:5], 1): # 처음 5개만 미리보기
            doc_content = row[0] if row[0] else "내용 없음"
            meta = row[1] if row[1] else {}

            print(f"\n--- 문서(청크) {i} ---")
            # 메타데이터를 보기 좋게 출력
            print(f"메타데이터: {json.dumps(meta, indent=2, ensure_ascii=False)}")
            print(f"내용 일부: {doc_content[:200]}...")

    cur.close()
    conn.close()
except Exception as e:
    print("DB 확인 중 오류 발생:", e)
