
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# إعداد Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# تحميل النموذج
model = SentenceTransformer("all-MiniLM-L6-v2")

# تحميل PDF وتحويله إلى نص
@st.cache_data
def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# تقسيم النص إلى Chunks
def split_text_into_chunks(text, chunk_size=5):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# إنشاء Embeddings
@st.cache_resource
def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=False)

# استرجاع السياق من chunks
def retrieve_chunks(query, chunks, chunk_embeddings, top_n=5):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(chunks[i], similarities[i]) for i in top_indices]

# توليد إجابة من Groq
def generate_answer(context_chunks, question):
    context = "\n".join([f"- {chunk}" for chunk, _ in context_chunks])
    prompt = f"""You are a helpful assistant.
Use only the following context to answer the user's question. Do not invent or assume any information.

Context:
{context}

Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ======================= واجهة Streamlit =========================
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("🤖 اسألني عن لائحة حوكمة البيانات (NDMO)")

pdf_path = "ndmo_en.pdf"
with st.spinner("📄 جارٍ تحميل وتحليل الوثيقة..."):
    text = load_pdf_text(pdf_path)
    chunks = split_text_into_chunks(text)
    embeddings = embed_chunks(chunks)

st.success("✅ تم تجهيز النظام! يمكنك البدء بطرح الأسئلة.")

question = st.text_input("❓ اكتب سؤالك هنا:", placeholder="مثال: What are the objectives?")
if question:
    with st.spinner("💡 جاري توليد الإجابة..."):
        top_chunks = retrieve_chunks(question, chunks, embeddings)
        answer = generate_answer(top_chunks, question)
        st.markdown("### 📌 الإجابة:")
        st.write(answer)

        with st.expander("📚 عرض المصادر المسترجعة"):
            for i, (chunk, score) in enumerate(top_chunks, 1):
                st.markdown(f"""**🔹 القطعة {i} (التشابه: {score:.2f})**  
{chunk}""")


{chunk}")
