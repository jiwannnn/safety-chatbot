# -*- coding: utf-8 -*-
"""Untitled23.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16Mnsdmkw5XqZmEHJTvGn7gp2pm4-qhiQ
"""

import os
import time
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "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"  # 여기에 본인의 OpenAI API 키를 입력

# 업종별 파일 설정
industry_files = {
    "강선_건조업_안전보건": ["./data/강선 건조업 안전보건관리체계 구축 가이드.md", "./data/선박건조 및 수리업.csv"],
    "벌목업_안전보건": ["./data/벌목업 안전보건관리체계 구축 가이드.md", "./data/임업.csv"],
    "섬유_및_섬유제품_제조업(표백_및_염색가공업)": ["./data/섬유제품 염색, 정리 및 마무리 가공업 안전보건관리체계 구축 가이드.md", "./data/섬유 및 섬유제품제조업.csv"],
    "인쇄업_안전보건": ["./data/인쇄업 안전보건관리체계 구축 가이드.md", "./data/출판 인쇄업.csv"],
    "플라스틱제품_안전보건": ["./data/플라스틱 제품 제조업 안전보건관리체계 구축 가이드.md", "./data/플라스틱 가공 제품제조업.csv"],
    "자동차부품_안전보건": ["./data/자동차 부품 안전보건관리체계 구축 가이드.md", "./data/차량 부품 제조업.csv"],
}

# 공통 사례 파일 경로
common_file_path = "./data/공통.csv"

# 텍스트 분할 설정
def create_text_splitter():
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

text_splitter = create_text_splitter()

# 벡터 스토어 생성 함수
def create_vector_store(files, embeddings, source_type):
    all_documents = []
    for name, file_paths in files.items():
        if not isinstance(file_paths, list):  # 파일이 리스트가 아니면 리스트로 변환
            file_paths = [file_paths]

        for file_path in file_paths:
            try:
                # 텍스트 로더를 사용하여 문서 로드
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()

                # 메타데이터 추가
                for doc in documents:
                    doc.metadata["source"] = name
                    doc.metadata["type"] = source_type

                # 텍스트 분할
                split_texts = text_splitter.split_documents(documents)
                all_documents.extend(split_texts)
            except Exception as e:
                st.error(f"파일 로드 실패: {file_path}\n{str(e)}")

    return Chroma.from_documents(all_documents, embeddings)

# 업종별 데이터와 공통 데이터 각각의 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
industry_vector_store = create_vector_store(industry_files, embeddings, "industry")
common_vector_store = create_vector_store({"공통 사례": common_file_path}, embeddings, "common")

import os
import time
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

# Streamlit 페이지 설정
st.set_page_config(page_title="중대재해 사례 질의응답", page_icon="🤖")

# 텍스트 분할 설정
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 벡터 스토어 초기화 (업종별 및 공통 사례)
industry_vector_store = create_vector_store(industry_files, embeddings, "industry")
common_vector_store = create_vector_store({"공통 사례": common_file_path}, embeddings, "common")

# Streamlit 앱 제목
st.title("중대재해 사례 및 안전보건관리체계 질의응답 시스템")

# 업종 선택
selected_industry = st.selectbox("업종을 선택하세요", list(industry_files.keys()))

# 사용자 질문 입력
query = st.text_input("질문을 입력하세요:")

# 검색 버튼 클릭 시 실행
if st.button("검색"):
    if not query:
        st.warning("질문을 입력하세요.")
    else:
        try:
            # 업종 데이터 검색
            industry_retriever = industry_vector_store.as_retriever(search_kwargs={"k": 3})
            industry_results = industry_retriever.get_relevant_documents(query)

            # 대기 시간 추가 (API 요청 간 속도 제한 방지)
            time.sleep(2)

            # 공통 데이터 검색
            common_retriever = common_vector_store.as_retriever(search_kwargs={"k": 3})
            common_results = common_retriever.get_relevant_documents(query)

            # 선택된 업종의 데이터와 공통 데이터를 함께 결합
            all_results = industry_results + common_results

            # 검색 결과 결합
            combined_context = "\n".join([doc.page_content for doc in all_results])

            # 텍스트 분할
            split_contexts = text_splitter.split_text(combined_context)

            # 프롬프트 템플릿 설정
            prompt_template = """다음 문서를 참고하여 질문에 답변하세요:

            {context}

            질문: {question}
            답변:"""

            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

            # GPT-4 모델 설정
            llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=500)

            # 분할된 텍스트에서 답변 생성
            final_response = ""
            for chunk in split_contexts:
                chain = LLMChain(llm=llm, prompt=prompt)
                response = chain.run({"context": chunk, "question": query})
                final_response += response + "\n"

                # 대기 시간 추가 (속도 제한 방지)
                time.sleep(2)

            # 결과 출력
            st.subheader("답변")
            st.write(final_response)

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
