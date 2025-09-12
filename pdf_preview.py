import os
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"

def preview_all_pdfs(data_dir: str = DATA_DIR, num_pages_per_pdf: int = 2):
    """'data' 디렉토리의 모든 PDF에서 추출된 텍스트와 표 일부를 확인"""
    if not os.path.isdir(data_dir):
        print(f"오류: '{data_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"오류: '{data_dir}' 디렉토리에서 PDF 파일을 찾을 수 없습니다.")
        return

    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"\n=========================================")
                print(f"***** PDF 미리보기: {pdf_path} *****")
                print(f"=========================================")
                for i, page in enumerate(pdf.pages[:num_pages_per_pdf]):
                    page_text = page.extract_text()
                    print(f"\n--- 페이지 {i+1} ---")
                    if page_text:
                        print(page_text[:500] + "..." if len(page_text) > 500 else page_text)
                    else:
                        print("텍스트 없음")
                    
                    tables = page.extract_tables()
                    if tables:
                        print(f"--- 페이지 {i+1}의 첫 번째 표 미리보기 ---")
                        try:
                            df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                            print(df.head())
                        except Exception as e:
                            print(f"Pandas DataFrame 변환 실패: {e}")
                            print("원본 테이블 데이터 (최대 5줄):")
                            for row in tables[0][:5]:
                                print(row)
        except Exception as e:
            print(f"'{pdf_path}' 파일 미리보기 중 오류 발생: {e}")

if __name__ == "__main__":
    preview_all_pdfs()
