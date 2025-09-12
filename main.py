import os
import time
import json
import psycopg2
from typing import Dict, List
from dotenv import load_dotenv

# FastAPI 및 slowapi 관련
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# Pydantic 모델
from pydantic import BaseModel

# LangChain 관련
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.messages import SystemMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from pdf_importer import create_vector_store, CONNECTION_STRING, COLLECTION_NAME



load_dotenv()
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(
    model_name='nlpai-lab/KURE-v1',
    model_kwargs={'device': 'cpu'}
)

try:
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings
    )
    print("PostgreSQL에서 벡터 저장소가 로드됨")
except Exception as e:
    print(f"PostgreSQL 연결 오류: {e}")
    print("새로운 백터 저장소 생성")
    vector_store = create_vector_store()
    if vector_store is None:
        exit("오류: 백터 저장소 생성 오류")

llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-flash-8b",
    model="gemini-2.5-flash-lite",
    model_kwargs={
        "system_instruction": SystemMessage(
            content=
            """

당신은 한국외국어대학교(서울)의 **'학사 생활 AI 어드바이저'**입니다. 당신의 지식은 주어진 [학사 규정]과 [주변 상권 정보] 문서로 한정됩니다. 당신의 임무는 이 지식 내에서 학생들의 질문에 명확하고 친절한 전문가의 어조로 답변하는 것입니다.

[답변 원칙]

1. 정확성: 반드시 주어진 참고 문서의 내용에만 근거하여 답변합니다.

2. 친절함: 항상 친절하고 이해하기 쉬운 완전한 문장으로 답변합니다.

3. 맥락 이해: 이전 대화 내용을 기억하여 자연스러운 대화를 이어갑니다.

4. 지식 내재화: 당신은 문서를 단순히 전달하는 로봇이 아닙니다. 주어진 참고 문서는 당신의 '지식'입니다. 답변 시, '제공된 정보', '참고 문서', '주어진 텍스트', '표', '문단' 등 당신이 정보를 어떻게 얻었는지 암시하는 그 어떤 단어도 절대 사용하지 마세요. 검색된 모든 정보를 완전히 자신의 지식인 것처럼 종합하고 자연스럽게 재구성하여, 마치 원래부터 알고 있었던 것처럼 사용자에게 직접 설명해야 합니다.

5. 한국어 사용: 모든 답변은 반드시 완벽한 한국어로만 생성해야 합니다.

[답변 규칙]

1. 자기소개: 만약 사용자가 당신의 정체성에 대해 묻는다면(예: "너는 누구야?", "이름이 뭐야?"), "안녕하세요! 저는 한국외국어대학교 학생들의 캠퍼스 생활을 돕기 위해 만들어진 '학사 생활 AI 어드바이저'입니다. 학사 정보나 학교 생활에 대해 궁금한 점이 있다면 무엇이든 물어보세요." 라고 정확히 소개해야 합니다. 절대로 'Google의 언어 모델'이나 마스코트 '부(Boo)'라고 자신을 소개해서는 안 됩니다.

2. 범위 외 질문 판단: 당신의 지식 범위(학사, 주변 맛집)와 명백히 관련 없는 질문(예: 금융, 스포츠)에는 "죄송합니다. 저는 한국외국어대학교 학사 및 캠퍼스 생활 정보에 대해서만 답변할 수 있습니다." 라고 답변하세요. '제공된 정보에 없다'는 식의 부연 설명은 절대 덧붙이지 마세요.

3. 정보 우선순위 판별: 여러 개의 참고 문서가 주어지면, 그중에서 사용자의 질문에 가장 직접적으로 답할 수 있는 핵심 정보를 먼저 식별하세요. 관련성이 떨어지거나 부차적인 정보는 답변에 포함하지 않거나, 꼭 필요한 경우에만 간략하게 덧붙여 설명하세요.

4. 표(Table) 분석: 참고 문서에 표가 포함된 경우, 당신은 표 분석 전문가로서 행과 열의 관계를 정확히 해석하여 답변해야 합니다.

5. 조건부 답변: 만약 표나 텍스트에 학과, 학번 등 세부 조건이 명시되어 있지 않다면, "제시된 자료에 따르면 일반적으로" 또는 "2025학년도 기준으로는" 과 같이 정보의 출처나 기준을 명확히 밝히며 답변하세요.

6.  포괄적 기준 적용: 사용자가 특정 학번(예: 2025학번)에 대해 질문했을 때, 참고 문서에 해당 학번이 명시적으로 존재하지 않더라도 '2015학번부터'와 같이 포괄적인 기준이 있다면, 그 기준이 질문에 해당하는 최신 규정이라고 판단하고 답변해야 합니다. 이때, '2015학번부터 적용되는 규정에 따르면'과 같이 어떤 기준을 사용했는지 명확히 밝혀주세요.

7. 다중 정보 처리: 만약 사용자의 질문에 대해 여러 문서에서 서로 다른 정보가 검색될 경우, 하나의 정보만 선택하지 마세요. 대신, 각각의 조건과 내용을 명확히 구분하여 모든 정보를 종합적으로 안내해야 합니다.

8. 예외 가능성 인지: 학사 규정은 단과대학, 학과, 학번별로 예외 규칙이 존재할 수 있다는 사실을 항상 인지하세요. 만약 일반적인 규칙을 찾았더라도, "일반적으로는 OO학점이 필요하지만, 소속 단과대학이나 학과에 따라 다를 수 있으니 정확한 정보는 학교 공식 문서를 확인하시거나 학과 사무실에 문의하는 것을 권장합니다" 와 같이 답변에 '주의사항'과 '한계'를 명시하세요.

9. 정보 부재 시: 위의 모든 노력에도 불구하고 질문에 대한 답변을 참고 문서에서 찾을 수 없는 경우에만, "죄송합니다. 문의하신 내용에 대한 정보는 제가 가진 자료에서 확인할 수 없습니다."라고 답변하세요.

            """

            
            ),
    }
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# retriever = MultiQueryRetriever.from_llm(
#     retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
#     llm=llm
# )

# 사용자 세션별 대화를 저장
chat_sessions: Dict[str, ConversationalRetrievalChain] = {}

def get_or_create_chain(session_id: str) -> ConversationalRetrievalChain:
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,
    input_key="question",
    output_key="answer" )
        new_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True, # 참고 문서 반환 활성화
            output_key="answer"
        )
        chat_sessions[session_id] = new_chain
        print(f"새로운 세션 ID 생성: {session_id}")
    return chat_sessions[session_id]


class ChatMessage(BaseModel):
    message: str
    session_id: str
    user_id: str # 사용자 식별

class ChatResponse(BaseModel):
    response: str
    success: bool
    source_documents: List[Dict[str, str]] = [] # 문서 내용과 메타데이터





@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("15/minute")
async def chat_with_gemini(request: Request):
    start_time = time.time()

    try:

        body = await request.json()
        chat_message = ChatMessage(**body)

        qa_chain = get_or_create_chain(chat_message.session_id)
        result = qa_chain.invoke({"question": chat_message.message})

        # 참고 문서 추출 및 로그 출력
        source_documents_for_response: List[Dict[str, str]] = []
        if 'source_documents' in result and result['source_documents']:
            print("\n--- 참고 문서 ---")
            for i, doc in enumerate(result['source_documents']):
                print(f"문서 {i+1}:")
                print(f"  소스: {doc.metadata.get('source', '알 수 없음')}")
                print(f"  내용 (일부): {doc.page_content[:200]}...")
                source_documents_for_response.append({
                    "source": doc.metadata.get('source', '알 수 없음'),
                    "content": doc.page_content 
                })
            print("---------------\n")
        
        response_time_ms = int((time.time() - start_time) * 1000)

        # DB에 로그 저장
        try:
            db_conn_str = CONNECTION_STRING.replace("postgresql+psycopg2", "postgresql")
            conn = psycopg2.connect(db_conn_str)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_logs (session_id, user_id, user_question, bot_answer, retrieved_sources, response_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (chat_message.session_id, chat_message.user_id, chat_message.message, result['answer'], json.dumps(source_documents_for_response), response_time_ms)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_error:
            print(f"DB 로그 저장 실패: {db_error}")
        

        return ChatResponse(
            response=result['answer'],
            success=True,
            source_documents=source_documents_for_response # 응답에 참고 문서 추가
        )
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return ChatResponse(
            response=f"오류가 발생했습니다: {str(e)}",
            success=False,
            source_documents=[]
        )

@app.get("/")
async def root():
    return {"message": "한국외국어대학교(서울) 학사 챗봇 API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render 사용시 PORT
    uvicorn.run(app, host="0.0.0.0", port=port)

