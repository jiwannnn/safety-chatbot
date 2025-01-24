# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader

# Streamlit 페이지 설정
st.set_page_config(page_title="업종별 중대재해 사례 및 안전보건관리체계 질의응답", page_icon="🤖")

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: #555;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="bottom-right">by 김지완</div>', unsafe_allow_html=True)

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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
    # 기본 분할 설정
    chunk_size = 200
    chunk_overlap = 50

    if context_length and context_length > 32000:
        chunk_size = 150
        chunk_overlap = 50

    # RecursiveCharacterTextSplitter 사용
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # 구분자 순서대로 시도
    )

# 벡터 스토어 생성
def create_vector_store(files, embeddings, source_type):
    all_documents = []
    for name, file_paths in files.items():
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    st.warning(f"파일이 존재하지 않습니다: {file_path}")
                    continue

                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = name
                    doc.metadata["type"] = source_type

                all_documents.extend(documents)
            except Exception as e:
                st.error(f"파일 로드 실패: {file_path}\n{str(e)}")

    text_splitter = create_text_splitter(len(" ".join([doc.page_content for doc in all_documents]).split()))
    split_texts = text_splitter.split_documents(all_documents)
    return FAISS.from_documents(split_texts, embeddings)

# 요약 함수 (GPT-4-32k 사용)
def summarize_context(llm, context):
    prompt = PromptTemplate(
        input_variables=["context"],
        template=(
            "다음 텍스트를 간략히 요약하세요:\n\n"
            "{context}\n\n"
            "요약:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    max_context_length = 20000  # 안전한 입력 길이 제한
    truncated_context = context[:max_context_length]
    summary = chain.run({"context": truncated_context})
    return summary

# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings()

try:
    # 벡터 스토어 생성
    industry_vector_store = create_vector_store(industry_files, embeddings, "industry")
    common_vector_store = create_vector_store({"공통 사례": common_file_path}, embeddings, "common")
except Exception as e:
    st.error(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
    st.stop()

# Streamlit UI 구성
st.markdown("<h1 style='text-align: center;'>업종별 중대재해 사례 및 안전보건관리체계 질의응답</h1>", unsafe_allow_html=True)

selected_industry = st.selectbox("업종을 선택하세요", list(industry_files.keys()))
query = st.text_input("질문을 입력하세요:")

if st.button("검색"):
    if not query:
        st.warning("질문을 입력하세요.")
    else:
        try:
            # 검색 수행
            industry_retriever = industry_vector_store.as_retriever(search_kwargs={"k": 1})
            industry_results = industry_retriever.get_relevant_documents(query)

            common_retriever = common_vector_store.as_retriever(search_kwargs={"k": 1})
            common_results = common_retriever.get_relevant_documents(query)

            all_results = industry_results + common_results
            combined_context = "\n".join([doc.page_content for doc in all_results])

            # 요약 단계
            llm_summary = ChatOpenAI(model_name="gpt-4-turbo-32k", temperature=0, max_tokens=1000)
            summarized_context = summarize_context(llm_summary, combined_context)

            # 최종 답변 생성
            prompt_template = """다음 문서를 참고하여 질문에 답변하세요:
            {context}
            질문: {question}
            답변:"""

            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            llm = ChatOpenAI(model_name="gpt-4-32k", temperature=0, max_tokens=500)

            chain = LLMChain(llm=llm, prompt=prompt)

            # **청크 크기를 줄이고 요청 간 딜레이 추가**
            final_response = ""
            for chunk in summarized_context.split("\n"):
                if chunk.strip():  # 빈 청크 제외
                    response = chain.run({"context": chunk, "question": query})
                    final_response += response + "\n"
                    time.sleep(2)  # 요청 간 2초 대기

            # 결과 출력
            st.subheader("답변")
            st.write(final_response)

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

