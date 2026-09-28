import os

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.retriever import search_chunks
from src.text_splitter import split_text


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="DocuRAG", page_icon="📄")
st.title("DocuRAG")
st.write("Upload a PDF, ask questions, or get a summary.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    ) as document:
        pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({"page": page_number, "text": text})

    if not pages:
        st.warning("I couldn't find selectable text in this PDF.")
    else:
        chunks = []

        for page in pages:
            page_chunks = split_text(
                page["text"],
                chunk_size=40,
                overlap=8,
            )

            for chunk_text in page_chunks:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text,
                })

        st.success(f"Read {len(pages)} page(s) from {uploaded_file.name}.")

        if st.button("Summarize this PDF"):
            if not api_key:
                st.error("The Gemini key wasn't found in your .env file.")
            else:
                document_text = "\n\n".join(
                    f"Page {page['page']}:\n{page['text']}"
                    for page in pages
                )

                summary_prompt = f"""
Summarize this PDF using only the text below.
Use clear bullet points and include page numbers for important details.
If a detail is unclear or missing, say so.

PDF text:
{document_text}
"""

                try:
                    with st.spinner("Summarizing the PDF..."):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-3.5-flash-lite",
                            contents=summary_prompt,
                        )

                    st.subheader("Document summary")
                    st.write(response.text or "Gemini returned no summary.")

                except Exception as error:
                    st.error(
                        "Gemini couldn't summarize this PDF. "
                        "It may be temporarily busy or at its free-tier limit."
                    )
                    with st.expander("Show technical details"):
                        st.code(str(error))

        with st.form("question_form"):
            question = st.text_input("Ask a question about this PDF")
            submitted = st.form_submit_button("Get answer")

        if submitted and question.strip():
            if not api_key:
                st.error("The Gemini key wasn't found in your .env file.")
            else:
                try:
                    with st.spinner("Searching the PDF..."):
                        matching_chunks = search_chunks(chunks, question,limit = 5)

                except Exception as error:
                    st.error(
                        "Couldn't search the PDF. Gemini may be busy or its "
                        "free-tier request limit may have been reached. "
                        "Wait a minute and try again."
                    )
                    with st.expander("Show technical details"):
                        st.code(str(error))
                    st.stop()

                if not matching_chunks:
                    st.info("I couldn't find matching passages in the PDF.")
                else:
                    context = "\n\n".join(
                        f"Source page {chunk['page']}:\n{chunk['text']}"
                        for chunk in matching_chunks
                    )

                    answer_prompt = f"""
Answer the question using only the PDF passages below.
If they do not contain the answer, say that you could not find it in the PDF.
Include the source page number in your answer.

Question:
{question}

PDF passages:
{context}
"""

                    try:
                        with st.spinner("Writing an answer..."):
                            client = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model="gemini-3.5-flash-lite",
                                contents=answer_prompt,
                            )

                        st.subheader("Answer")
                        st.write(response.text or "Gemini returned no text.")

                        with st.expander("Show the passages used"):
                            for chunk in matching_chunks:
                                st.markdown(
                                    f"**Source: page {chunk['page']}**"
                                )
                                st.write(chunk["text"])

                    except Exception as error:
                        st.error(
                            "Gemini couldn't answer this request. "
                            "It may be temporarily busy or at its free-tier limit."
                        )
                        with st.expander("Show technical details"):
                            st.code(str(error))