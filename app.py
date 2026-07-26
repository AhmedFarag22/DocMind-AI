
import re
import json

import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# Application Title
# =========================================================

st.title("📄 DocMind AI")

st.write(
    "AI Document Intelligence System powered by RAG."
)


# =========================================================
# Load Models
# =========================================================

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


# =========================================================
# Text Splitter
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200

)


# =========================================================
# PDF FUNCTIONS
# =========================================================

def extract_pdf_text(uploaded_file):

    pdf_bytes = uploaded_file.read()


    doc = fitz.open(

        stream=pdf_bytes,

        filetype="pdf"

    )


    documents = []


    for page_number, page in enumerate(doc):

        text = page.get_text()


        if text.strip():

            documents.append({

                "page": page_number + 1,

                "text": text.strip()

            })


    return documents


# ---------------------------------------------------------


def create_chunks(documents):

    chunks = []


    for document in documents:

        page_chunks = text_splitter.split_text(

            document["text"]

        )


        for chunk_text in page_chunks:

            chunks.append({

                "text": chunk_text.strip(),

                "page": document["page"]

            })


    return chunks


# ---------------------------------------------------------


def build_context(chunks):

    return "\n\n".join(

        [

            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"

            for chunk in chunks

        ]

    )


# =========================================================
# FAISS / RAG FUNCTIONS
# =========================================================

def create_faiss_index(chunks):

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


    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(

        dimension

    )


    index.add(

        embeddings

    )


    return index


# ---------------------------------------------------------


def retrieve_relevant_chunks(

    query,

    index,

    chunks,

    k=5

):

    query_embedding = embedding_model.encode(

        [query]

    )


    query_embedding = np.array(

        query_embedding

    ).astype("float32")


    k = min(

        k,

        len(chunks)

    )


    distances, indices = index.search(

        query_embedding,

        k

    )


    retrieved_chunks = []


    for index_number in indices[0]:

        if 0 <= index_number < len(chunks):

            retrieved_chunks.append(

                chunks[index_number]

            )


    return retrieved_chunks


# =========================================================
# LLM FUNCTION
# =========================================================

def generate_text(

    prompt,

    max_new_tokens=250

):

    result = generator(

        prompt,

        max_new_tokens=max_new_tokens,

        do_sample=False

    )


    return result[0][

        "generated_text"

    ].strip()


# =========================================================
# OUTPUT PARSER
# =========================================================

def parse_qa_response(raw_output):

    """
    Parse and validate the LLM response.

    Expected format:

    {
        "answer": "...",
        "confidence": "high",
        "sources": [
            {
                "page": 1,
                "reason": "..."
            }
        ]
    }

    If parsing fails, return a safe fallback
    instead of crashing the application.
    """

    try:

        cleaned_output = raw_output.strip()


        # Remove Markdown code fences
        cleaned_output = re.sub(

            r"```json|```",

            "",

            cleaned_output,

            flags=re.IGNORECASE

        ).strip()


        # Try to find a JSON object
        json_match = re.search(

            r"\{.*\}",

            cleaned_output,

            re.DOTALL

        )


        if json_match:

            cleaned_output = json_match.group(0)


        parsed_data = json.loads(

            cleaned_output

        )


        answer = parsed_data.get(

            "answer",

            "The answer is not available in the provided context."

        )


        confidence = parsed_data.get(

            "confidence",

            "unknown"

        )


        sources = parsed_data.get(

            "sources",

            []

        )


        # Validate sources safely
        validated_sources = []


        if isinstance(sources, list):

            for source in sources:

                if isinstance(source, dict):

                    page = source.get(

                        "page"

                    )


                    reason = source.get(

                        "reason",

                        "Relevant information found on this page."

                    )


                    if isinstance(page, int):

                        validated_sources.append({

                            "page": page,

                            "reason": reason

                        })


        return {

            "answer": str(answer),

            "confidence": str(confidence),

            "sources": validated_sources

        }


    except Exception:

        # Safe fallback if the model does not return valid JSON

        return {

            "answer": raw_output,

            "confidence": "unknown",

            "sources": []

        }


# =========================================================
# KEY POINTS
# =========================================================

def extract_key_points(documents):

    """
    Extract important facts from labeled document sections.

    This is deterministic and does not depend on
    the small FLAN-T5 model.
    """

    key_points = []


    for document in documents:

        text = document["text"]


        # Payment Terms

        match = re.search(

            r"Payment Terms:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

            text,

            re.IGNORECASE | re.DOTALL

        )


        if match:

            key_points.append(

                match.group(1).strip()

            )


        # Late Payment

        match = re.search(

            r"Late Payment:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

            text,

            re.IGNORECASE | re.DOTALL

        )


        if match:

            key_points.append(

                match.group(1).strip()

            )


        # Contract Duration

        match = re.search(

            r"Contract Duration:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

            text,

            re.IGNORECASE | re.DOTALL

        )


        if match:

            key_points.append(

                match.group(1).strip()

            )


        # Termination

        match = re.search(

            r"Termination:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

            text,

            re.IGNORECASE | re.DOTALL

        )


        if match:

            key_points.append(

                match.group(1).strip()

            )


        # Data Protection

        match = re.search(

            r"Data Protection:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

            text,

            re.IGNORECASE | re.DOTALL

        )


        if match:

            key_points.append(

                match.group(1).strip()

            )


        # Governing Law

        match = re.search(

            r"Governing Law:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

            text,

            re.IGNORECASE | re.DOTALL

        )


        if match:

            key_points.append(

                match.group(1).strip()

            )


    return key_points


# =========================================================
# RISK ANALYSIS
# =========================================================

def analyze_risks(documents):

    risks = []


    full_text = "\n".join(

        document["text"]

        for document in documents

    )


    # -----------------------------------------------------
    # Late Payment Risk
    # -----------------------------------------------------

    late_payment_match = re.search(

        r"Late Payment:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

        full_text,

        re.IGNORECASE | re.DOTALL

    )


    if late_payment_match:

        late_payment_text = (

            late_payment_match.group(1).strip()

        )


        percentage_match = re.search(

            r"(\d+)%",

            late_payment_text

        )


        percentage = (

            percentage_match.group(1)

            if percentage_match

            else "a specified"

        )


        risks.append({

            "risk": "Late Payment Penalty",

            "explanation": (

                f"The client may face a {percentage}% "
                "penalty if payment is not received "
                "within the agreed period."

            ),

            "severity": "Medium"

        })


    # -----------------------------------------------------
    # Confidentiality Risk
    # -----------------------------------------------------

    confidentiality_match = re.search(

        r"Data Protection:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

        full_text,

        re.IGNORECASE | re.DOTALL

    )


    if confidentiality_match:

        risks.append({

            "risk": "Confidentiality Breach",

            "explanation": (

                "Unauthorized disclosure of confidential "
                "information could violate the data "
                "protection requirements."

            ),

            "severity": "Medium"

        })


    # -----------------------------------------------------
    # Termination Risk
    # -----------------------------------------------------

    termination_match = re.search(

        r"Termination:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",

        full_text,

        re.IGNORECASE | re.DOTALL

    )


    if termination_match:

        risks.append({

            "risk": "Contract Termination Risk",

            "explanation": (

                "Either party can terminate the agreement "
                "by providing the required written notice."

            ),

            "severity": "Low"

        })


    return risks


# =========================================================
# DOCUMENT COMPARISON
# =========================================================

def extract_sections(text):

    """
    Extract labeled sections such as:

    Payment Terms:
    Late Payment:
    Contract Duration:
    """

    sections = {}


    pattern = (

        r"(?m)^"

        r"([A-Za-z ]+):"

        r"\s*"

        r"(.*?)"

        r"(?=^[A-Za-z ]+:|\Z)"

    )


    matches = re.findall(

        pattern,

        text,

        re.MULTILINE | re.DOTALL

    )


    for title, content in matches:

        normalized_title = title.strip().lower()


        sections[normalized_title] = (

            content.strip()

        )


    return sections


# ---------------------------------------------------------


def compare_documents(documents_a, documents_b):

    text_a = "\n".join(

        document["text"]

        for document in documents_a

    )


    text_b = "\n".join(

        document["text"]

        for document in documents_b

    )


    sections_a = extract_sections(

        text_a

    )


    sections_b = extract_sections(

        text_b

    )


    similarities = []

    differences = []

    important_changes = []

    potential_risks = []


    # -----------------------------------------------------
    # Compare Sections
    # -----------------------------------------------------

    all_sections = sorted(

        set(sections_a.keys())

        |

        set(sections_b.keys())

    )


    for section in all_sections:

        value_a = sections_a.get(

            section

        )


        value_b = sections_b.get(

            section

        )


        if (

            value_a is not None

            and

            value_b is not None

        ):


            if value_a == value_b:

                similarities.append({

                    "section": section,

                    "value": value_a

                })


            else:

                differences.append({

                    "section": section,

                    "document_a": value_a,

                    "document_b": value_b

                })


        elif value_a is None:

            differences.append({

                "section": section,

                "document_a": "Not present",

                "document_b": value_b

            })


        else:

            differences.append({

                "section": section,

                "document_a": value_a,

                "document_b": "Not present"

            })


    # -----------------------------------------------------
    # Detect Specific Numeric Changes
    # -----------------------------------------------------

    for difference in differences:

        section = difference["section"]

        value_a = difference["document_a"]

        value_b = difference["document_b"]


        # Payment period

        if section == "payment terms":

            days_a = re.search(

                r"(\d+)\s+days",

                value_a,

                re.IGNORECASE

            )


            days_b = re.search(

                r"(\d+)\s+days",

                value_b,

                re.IGNORECASE

            )


            if days_a and days_b:

                old_value = days_a.group(1)

                new_value = days_b.group(1)


                important_changes.append(

                    f"Payment period changed from "
                    f"{old_value} days to "
                    f"{new_value} days."

                )


                if int(new_value) > int(old_value):

                    potential_risks.append(

                        "The longer payment period may "
                        "delay cash collection."

                    )


        # Late payment percentage

        if section == "late payment":

            percentage_a = re.search(

                r"(\d+)%",

                value_a

            )


            percentage_b = re.search(

                r"(\d+)%",

                value_b

            )


            if percentage_a and percentage_b:

                old_value = percentage_a.group(1)

                new_value = percentage_b.group(1)


                important_changes.append(

                    f"Late payment penalty changed from "
                    f"{old_value}% to "
                    f"{new_value}%."

                )


                if int(new_value) > int(old_value):

                    potential_risks.append(

                        "The higher late payment penalty "
                        "increases the financial impact "
                        "of delayed payment."

                    )


        # Contract duration

        if section == "contract duration":

            months_a = re.search(

                r"(\d+)\s+months",

                value_a,

                re.IGNORECASE

            )


            months_b = re.search(

                r"(\d+)\s+months",

                value_b,

                re.IGNORECASE

            )


            if months_a and months_b:

                old_value = months_a.group(1)

                new_value = months_b.group(1)


                important_changes.append(

                    f"Contract duration changed from "
                    f"{old_value} months to "
                    f"{new_value} months."

                )


                if int(new_value) > int(old_value):

                    potential_risks.append(

                        "The longer contract duration "
                        "creates a longer contractual "
                        "commitment."

                    )


        # Termination notice

        if section == "termination":

            notice_a = re.search(

                r"(\d+)\s+days",

                value_a,

                re.IGNORECASE

            )


            notice_b = re.search(

                r"(\d+)\s+days",

                value_b,

                re.IGNORECASE

            )


            if notice_a and notice_b:

                old_value = notice_a.group(1)

                new_value = notice_b.group(1)


                important_changes.append(

                    f"Termination notice changed from "
                    f"{old_value} days to "
                    f"{new_value} days."

                )


                if int(new_value) > int(old_value):

                    potential_risks.append(

                        "The longer termination notice "
                        "period may reduce flexibility "
                        "to exit the agreement quickly."

                    )


    return (

        similarities,

        differences,

        important_changes,

        potential_risks

    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(

    "⚙️ AI Operations"

)


task = st.sidebar.selectbox(

    "Select an operation",

    [

        "Question Answering",

        "Summarization",

        "Key Points",

        "Risk Analysis",

        "Document Comparison"

    ]

)


# =========================================================
# DOCUMENT COMPARISON
# =========================================================

if task == "Document Comparison":

    st.header(

        "📊 Document Comparison"

    )


    st.write(

        "Upload two PDF documents to compare their content."

    )


    uploaded_file_a = st.file_uploader(

        "Upload Document A",

        type=["pdf"],

        key="document_a"

    )


    uploaded_file_b = st.file_uploader(

        "Upload Document B",

        type=["pdf"],

        key="document_b"

    )


    if (

        uploaded_file_a is not None

        and

        uploaded_file_b is not None

    ):


        if st.button(

            "Compare Documents"

        ):


            with st.spinner(

                "Comparing documents..."

            ):


                documents_a = extract_pdf_text(

                    uploaded_file_a

                )


                documents_b = extract_pdf_text(

                    uploaded_file_b

                )


                (

                    similarities,

                    differences,

                    important_changes,

                    potential_risks

                ) = compare_documents(

                    documents_a,

                    documents_b

                )


            st.subheader(

                "📊 Comparison Result"

            )


            # -------------------------------------------------
            # Similarities
            # -------------------------------------------------

            st.markdown(

                "### ✅ Similarities"

            )


            if similarities:

                for item in similarities:

                    st.write(

                        f"- **{item['section'].title()}:** "
                        f"{item['value']}"

                    )

            else:

                st.write(

                    "No identical sections were found."

                )


            # -------------------------------------------------
            # Differences
            # -------------------------------------------------

            st.markdown(

                "### 🔄 Differences"

            )


            if differences:

                for item in differences:

                    st.write(

                        f"**{item['section'].title()}**"

                    )


                    st.write(

                        f"**Document A:** "
                        f"{item['document_a']}"

                    )


                    st.write(

                        f"**Document B:** "
                        f"{item['document_b']}"

                    )

                    st.divider()

            else:

                st.write(

                    "No differences were found."

                )


            # -------------------------------------------------
            # Important Changes
            # -------------------------------------------------

            st.markdown(

                "### ⚠️ Important Changes"

            )


            if important_changes:

                for change in important_changes:

                    st.write(

                        f"- {change}"

                    )

            else:

                st.write(

                    "No important changes were detected."

                )


            # -------------------------------------------------
            # Potential Risks
            # -------------------------------------------------

            st.markdown(

                "### 🚨 Potential Risks"

            )


            if potential_risks:

                for risk in potential_risks:

                    st.write(

                        f"- {risk}"

                    )

            else:

                st.write(

                    "No new risks caused by the differences "
                    "were detected."

                )


# =========================================================
# SINGLE DOCUMENT OPERATIONS
# =========================================================

else:

    st.header(

        "📄 Document Analysis"

    )


    uploaded_file = st.file_uploader(

        "Upload a PDF",

        type=["pdf"],

        key="single_document"

    )


    if uploaded_file is not None:


        with st.spinner(

            "Processing document..."

        ):


            documents = extract_pdf_text(

                uploaded_file

            )


            chunks = create_chunks(

                documents

            )


            index = create_faiss_index(

                chunks

            )


        st.success(

            f"Document processed successfully! "
            f"{len(chunks)} chunks created."

        )


        # =================================================
        # QUESTION ANSWERING
        # =================================================

        if task == "Question Answering":

            st.subheader(

                "💬 Ask a Question"

            )


            question = st.text_input(

                "Ask a question about your document"

            )


            if st.button(

                "Ask"

            ):


                if question.strip():


                    retrieved_chunks = (

                        retrieve_relevant_chunks(

                            question,

                            index,

                            chunks,

                            k=5

                        )

                    )


                    context = build_context(

                        retrieved_chunks

                    )


                    prompt = f"""
Answer the question using only the context below.

Return ONLY valid JSON.

The JSON must follow exactly this structure:

{{
    "answer": "your answer here",
    "confidence": "high, medium, or low",
    "sources": [
        {{
            "page": 1,
            "reason": "why this page supports the answer"
        }}
    ]
}}

Rules:

1. Use only the provided context.
2. Do not invent information.
3. If the answer is not available, write exactly:
   "The answer is not available in the provided context."
4. Use only page numbers that appear in the context.
5. Return valid JSON only.
6. Do not use Markdown.
7. Do not add explanations outside the JSON.

Context:

{context}

Question:

{question}

JSON:
"""


                    with st.spinner(

                        "Generating answer..."

                    ):


                        raw_answer = generate_text(

                            prompt,

                            max_new_tokens=250

                        )


                    # Output Parser

                    parsed_response = parse_qa_response(

                        raw_answer

                    )


                    st.subheader(

                        "🤖 Answer"

                    )


                    st.write(

                        parsed_response["answer"]

                    )


                    st.write(

                        f"**Confidence:** "
                        f"{parsed_response['confidence']}"

                    )


                    st.subheader(

                        "📚 References"

                    )


                    if parsed_response["sources"]:

                        for source in parsed_response["sources"]:

                            page_number = source["page"]

                            reason = source["reason"]


                            st.write(

                                f"📄 **Page {page_number}** — "
                                f"{reason}"

                            )


                            matching_chunks = [

                                chunk

                                for chunk in retrieved_chunks

                                if chunk["page"] == page_number

                            ]


                            for chunk in matching_chunks:

                                with st.expander(

                                    f"View source from Page {page_number}"

                                ):

                                    st.write(

                                        chunk["text"]

                                    )

                    else:

                        st.info(

                            "No structured references were returned. "
                            "Showing retrieved context below."

                        )


                        for chunk in retrieved_chunks:

                            with st.expander(

                                f"Page {chunk['page']}"

                            ):

                                st.write(

                                    chunk["text"]

                                )


        # =================================================
        # SUMMARIZATION
        # =================================================

        elif task == "Summarization":

            st.subheader(

                "📝 Document Summarization"

            )


            if st.button(

                "Summarize Document"

            ):


                summaries = []


                with st.spinner(

                    "Generating summaries..."

                ):


                    for chunk in chunks:

                        prompt = f"""
Summarize the following document section.

Keep the summary concise.

Focus only on important information.

Text:

{chunk['text']}

Summary:
"""


                        summary = generate_text(

                            prompt,

                            max_new_tokens=100

                        )


                        summaries.append({

                            "page": chunk["page"],

                            "summary": summary

                        })


                combined_summary = "\n\n".join(

                    [

                        item["summary"]

                        for item in summaries

                    ]

                )


                final_prompt = f"""
Create a concise final summary from the following text.

Focus on the main topic and the most important facts.

Do not copy the text word for word.

Text:

{combined_summary}

Final Summary:
"""


                with st.spinner(

                    "Creating final summary..."

                ):


                    final_summary = generate_text(

                        final_prompt,

                        max_new_tokens=300

                    )


                st.subheader(

                    "📝 Final Summary"

                )


                st.write(

                    final_summary

                )


                st.subheader(

                    "📚 Summary Sources"

                )


                for item in summaries:

                    with st.expander(

                        f"Page {item['page']}"

                    ):

                        st.write(

                            item["summary"]

                        )


        # =================================================
        # KEY POINTS
        # =================================================

        elif task == "Key Points":

            st.subheader(

                "🔑 Key Points Extraction"

            )


            if st.button(

                "Extract Key Points"

            ):


                key_points = extract_key_points(

                    documents

                )


                st.subheader(

                    "🔑 Key Points"

                )


                if key_points:

                    for point in key_points:

                        st.write(

                            f"- {point}"

                        )

                else:

                    st.info(

                        "No structured key points were detected."

                    )


                st.subheader(

                    "📚 Sources"

                )


                for document in documents:

                    with st.expander(

                        f"Page {document['page']}"

                    ):

                        st.write(

                            document["text"]

                        )


        # =================================================
        # RISK ANALYSIS
        # =================================================

        elif task == "Risk Analysis":

            st.subheader(

                "⚠️ Risk Analysis"

            )


            if st.button(

                "Analyze Risks"

            ):


                risks = analyze_risks(

                    documents

                )


                st.subheader(

                    "⚠️ Identified Risks"

                )


                if risks:

                    for risk in risks:

                        st.markdown(

                            f"### {risk['risk']}"

                        )


                        st.write(

                            f"**Explanation:** "
                            f"{risk['explanation']}"

                        )


                        st.write(

                            f"**Severity:** "
                            f"{risk['severity']}"

                        )


                else:

                    st.success(

                        "No significant risks were identified."

                    )


                st.subheader(

                    "📚 Risk Sources"

                )


                for document in documents:

                    with st.expander(

                        f"Page {document['page']}"

                    ):

                        st.write(

                            document["text"]
                        )

