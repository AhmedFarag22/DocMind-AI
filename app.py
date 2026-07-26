
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="📄"
)


# =========================
# Application Title
# =========================

st.title("📄 DocMind AI")

st.write(
    "Ask questions about your PDF using Retrieval-Augmented Generation."
)


# =========================
# Load Models
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_generator():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )


embedding_model = load_embedding_model()

generator = load_generator()


# =========================
# Upload PDF
# =========================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # =========================
    # Read PDF
    # =========================

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )


    # =========================
    # Extract Text
    # =========================

    documents = []

    for page_number, page in enumerate(doc):

        text = page.get_text()

        if text.strip():

            documents.append({

                "page": page_number + 1,

                "text": text

            })


    # =========================
    # Split Text into Chunks
    # =========================

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = []


    for document in documents:

        page_chunks = text_splitter.split_text(

            document["text"]

        )


        for chunk_text in page_chunks:

            chunks.append({

                "text": chunk_text,

                "page": document["page"]

            })


    # =========================
    # Create Embeddings
    # =========================

    texts = [

        chunk["text"]

        for chunk in chunks

    ]


    embeddings = embedding_model.encode(

        texts,

        show_progress_bar=False

    )


    embeddings = np.array(

        embeddings

    ).astype("float32")


    # =========================
    # Create FAISS Index
    # =========================

    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(

        dimension

    )


    index.add(

        embeddings

    )


    st.success(

        f"Document processed successfully! "

        f"{len(chunks)} chunks created."

    )


    # =========================
    # Ask Question
    # =========================

    question = st.text_input(

        "Ask a question about your document"

    )


    if st.button("Ask"):

        if question.strip():


            # =========================
            # Question Embedding
            # =========================

            question_embedding = embedding_model.encode(

                [question]

            )


            question_embedding = np.array(

                question_embedding

            ).astype("float32")


            # =========================
            # Retrieve Relevant Chunks
            # =========================

            k = min(

                5,

                len(chunks)

            )


            distances, indices = index.search(

                question_embedding,

                k

            )


            retrieved_chunks = []


            for index_number in indices[0]:

                if 0 <= index_number < len(chunks):

                    retrieved_chunks.append(

                        chunks[index_number]

                    )


            # =========================
            # Build Context
            # =========================

            context = "\n\n".join(

                [

                    f"[Page {chunk['page']}]\n"

                    f"{chunk['text']}"

                    for chunk in retrieved_chunks

                ]

            )


            # =========================
            # Build Prompt
            # =========================

            prompt = f"""

Answer the question using only the context below.

If the answer cannot be found in the context,

say:

"The answer is not available in the provided context."

Context:

{context}

Question:

{question}

Answer:

"""


            # =========================
            # Generate Answer
            # =========================

            with st.spinner(

                "Generating answer..."

            ):

                result = generator(

                    prompt,

                    max_new_tokens=150,

                    do_sample=False

                )


            answer = result[0][

                "generated_text"

            ]


            # =========================
            # Display Answer
            # =========================

            st.subheader(

                "🤖 Answer"

            )


            st.write(

                answer

            )


            # =========================
            # Display Sources
            # =========================

            st.subheader(

                "📚 Sources"

            )


            for chunk in retrieved_chunks:

                with st.expander(

                    f"Page {chunk['page']}"

                ):

                    st.write(

                        chunk["text"]

                    )