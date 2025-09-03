import os
import psycopg2
from dotenv import load_dotenv

# .env 파일이 Colab에 업로드되어 있어야 합니다.
load_dotenv()

try:
    print("***** 데이터베이스 연결을 시도합니다... *****")
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        port=os.getenv('POSTGRES_PORT'),
        connect_timeout=10 # 10초 이상 응답 없으면 실패 처리
    )
    print("✅ 데이터베이스 연결에 성공했습니다!")
    conn.close()
except Exception as e:
    print("❌ 데이터베이스 연결에 실패했습니다.")
    print("오류 내용:", e)