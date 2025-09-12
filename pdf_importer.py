import os
import pdfplumber
import pandas as pd # pandas 라이브러리 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# 데이터베이스 연결 정보
CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
COLLECTION_NAME = "homepage_pdfplumner_1st"

def create_vector_store():
    data_dir = "data"
    if not os.path.isdir(data_dir):
        print(f"오류: '{data_dir}' 디렉토리를 찾을 수 없습니다.")
        return None

    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    pdf_files = pdf_files[:1] # <-- 처음 1개만 선택 for 테스트
    print(f"***** 총 {len(pdf_files)}개의 파일로 첫 번째 배치를 처리합니다. *****")
    
    if not pdf_files:
        print(f"오류: '{data_dir}' 디렉토리에서 PDF 파일을 찾을 수 없습니다.")
        return None

    documents = []

    print("***** PDF 로드 및 텍스트 추출 시작... *****")
    progress_bar = tqdm(pdf_files, desc="PDF 처리 준비 중")

    for pdf_path in progress_bar:
        progress_bar.set_description(f"처리 중: {os.path.basename(pdf_path)}")

        # pdfplumber를 사용하여 PDF 파일 열기
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_metadata = {"source": pdf_path, "page": i + 1}
                    page_text = page.extract_text()
                    if page_text:
                        documents.append(Document(page_content=page_text, metadata=page_metadata))

                    # 페이지에서 표 추출
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            table_string = ""
                            # pandas를 사용하여 표 데이터를 Markdown 형식으로 변환
                            try:
                                # 첫 번째 행을 헤더로 가정, DataFrame 생성
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_string += df.to_markdown(index=False) + "\n"
                            except Exception as table_e:
                                # pandas 변환 실패 시 기존 탭 구분 방식으로 대체
                                print(f"경고: 페이지 {i+1}의 표 {table_idx+1} 처리 중 오류 발생 (pandas): {table_e}. 탭 구분 방식으로 대체합니다.")
                                for row in table:
                                    cleaned_row = [cell if cell is not None else "" for cell in row]
                                    table_string += "\t".join(cleaned_row) + "\n"
                            
                            if table_string:
                                table_metadata = page_metadata.copy()
                                table_metadata["table_index"] = table_idx + 1
                                documents.append(Document(page_content=table_string, metadata=table_metadata))
                print(f"*****'{pdf_path}' 텍스트 및 표 추출 완료.")

        except Exception as e:
            print(f"'{pdf_path}' 파일 처리 중 오류 발생: {e}")

    if not documents:
        print("오류: 모든 PDF 파일에서 텍스트를 추출하지 못했습니다.")
        return None

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    print("*****텍스트 분할 완료.")

    # 3. 임베딩 모델 로드
    embedding_model_name = 'nlpai-lab/KURE-v1'
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        # model_kwargs={'device': 'cuda'}
        model_kwargs={'device': 'cpu'}
    )
    print("*****임베딩 모델 로드 완료.")

    # PGVector를 사용해 벡터 저장소 생성
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

    print("*****PostgreSQL에 벡터 저장소 생성 완료.")
    return db

if __name__ == "__main__":
    create_vector_store()
