import os
import time
import pandas as pd
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import Document

# 업종별 파일 설정
industry_files = {
    "강선_건조업_안전보건": ["./data/강선 건조업 안전보건관리체계 구축 가이드.md", "./data/선박건조 및 수리업.csv"],
    "벌목업_안전보건": ["./data/벌목업 안전보건관리체계 구축 가이드.md", "./data/임업.csv"],
    "섬유_및_섬유제품_제조업(표백_및_염색가공업)": ["./data/섬유제품 염색, 정리 및 마무리 가공업 안전보건관리체계 구축 가이드.md", "./data/섬유 및 섬유제품제조업.csv"],
    "인쇄업_안전보건": ["./data/인쇄업 안전보건관리체계 구축 가이드.md", "./data/출판 인쇄업.csv"],
    "플라스틱제품_안전보건": ["./data/플라스틱 제품 제조업 안전보건관리체계 구축 가이드.md", "./data/플라스틱 가공 제품제조업.csv"],
    "자동차부품_안전보건": ["./data/자동차 부품 안전보건관리체계 구축 가이드.md", "./data/차량 부품 제조업.csv"],
}

common_file_path = "./data/공통.csv"

# 텍스트 분할 설정
def create_text_splitter(context_length=None):
    chunk_size = 300
    chunk_overlap = 50
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )

# CSV 파일 로드 함수
def load_csv_as_documents(file_path, metadata=None):
    try:
        df = pd.read_csv(file_path)
        documents = []
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            doc_metadata = metadata or {}
            documents.append(Document(page_content=content, metadata=doc_metadata))
        return documents
    except Exception as e:
        st.error(f"CSV 파일 로드 실패: {file_path}\n{e}")
        return []

# 벡터 스토어 생성
def create_vector_store(files, embeddings, source_type):
    all_documents = []
    for name, file_paths in files.items():
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        for file_path in file_paths:
            if not os.path.exists(file_path):
                st.warning(f"파일이 존재하지 않습니다: {file_path}")
                continue
            if file_path.endswith(".csv"):
                documents = load_csv_as_documents(file_path, metadata={"source": name, "type": source_type})
            else:
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source"] = name
                    doc.metadata["type"] = source_type
            all_documents.extend(documents)
            time.sleep(2)  # 각 파일 처리 후 딜레이 추가
    text_splitter = create_text_splitter()
    split_texts = text_splitter.split_documents(all_documents)
    return FAISS.from_documents(split_texts, embeddings)

# OpenAI 초기화
embeddings = OpenAIEmbeddings()
try:
    industry_vector_store = create_vector_store(industry_files, embeddings, "industry")
    common_vector_store = create_vector_store({"공통 사례": common_file_path}, embeddings, "common")
except Exception as e:
    st.error(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
    st.stop()

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>업종별 중대재해 사례 및 안전보건관리체계 질의응답</h1>", unsafe_allow_html=True)
selected_industry = st.selectbox("업종을 선택하세요", list(industry_files.keys()))
query = st.text_input("질문을 입력하세요:")

if st.button("검색"):
    if not query:
        st.warning("질문을 입력하세요.")
    else:
        industry_retriever = industry_vector_store.as_retriever(search_kwargs={"k": 1})
        time.sleep(2)  # API 호출 전 딜레이 추가
        industry_results = industry_retriever.get_relevant_documents(query)
        
        common_retriever = common_vector_store.as_retriever(search_kwargs={"k": 1})
        time.sleep(2)  # API 호출 전 딜레이 추가
        common_results = common_retriever.get_relevant_documents(query)

        print(f"Industry results: {len(industry_results)}, Common results: {len(common_results)}")


