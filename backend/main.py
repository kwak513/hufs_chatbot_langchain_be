import os
from typing import Dict
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.messages import SystemMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pdf_importer import create_vector_store, CONNECTION_STRING, COLLECTION_NAME

# 환경 변수 로드
load_dotenv()

# RAG 구성 요소를 프로그램 시작 시 한 번만 초기화
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
    print("Vector store loaded from PostgreSQL.")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    print("Creating a new vector store...")
    vector_store = create_vector_store()
    if vector_store is None:
        exit("Error: Vector store could not be created.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    model_kwargs={
        "system_instruction": SystemMessage(content="""당신은 한국외국어대학교(서울) 학사 전문가입니다. 답변 원칙: 1. 한국외국어대학교(서울) 관련 질문에 정확히 답변합니다. 2. 이전 대화 맥락을 기억하고 유연하게 응답합니다. 3. 친절하고 이해하기 쉬운 말투를 사용하며, 반드시 완전한 문장으로 답변합니다. 4. 참고 정보에 없는 내용은 절대 추측하거나 임의로 답변하지 않습니다. 답변 규칙: - 한국외국어대학교(서울)과 관련 없는 질문: "죄송합니다. 한국외국어대학교(서울) 관련 질문에만 답변드릴 수 있습니다."라고 답변하세요. - 사용자의 질문과 관련된 정보가 참고 문서에 명확하게 존재하지 않는 경우, 어떤 내용도 추론하거나 덧붙이지 말고 무조건 "죄송합니다. 해당 정보를 확인할 수 없습니다."라고 답변하세요.""")
    }
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 사용자 세션별 대화 체인을 저장할 딕셔너리
chat_sessions: Dict[str, ConversationalRetrievalChain] = {}

def get_or_create_chain(session_id: str) -> ConversationalRetrievalChain:
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        new_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            # return_source_documents=True <-- 이 부분을 제거했습니다.
        )
        chat_sessions[session_id] = new_chain
        print(f"새로운 세션 ID 생성: {session_id}")
    return chat_sessions[session_id]

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    success: bool
    
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatMessage):
    try:
        qa_chain = get_or_create_chain(request.session_id)
        result = qa_chain.invoke({"question": request.message})
        
        return ChatResponse(
            response=result['answer'],
            success=True
        )
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return ChatResponse(
            response=f"오류가 발생했습니다: {str(e)}", 
            success=False
        )

@app.get("/")
async def root():
    return {"message": "한국외국어대학교(서울) 학사 챗봇 API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)