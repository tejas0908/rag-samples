from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import glob
from text_chunker import sentences
import nltk
import cleantext
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

class RnsitRagApplication:
    def __init__(self) -> None:
        nltk.download('punkt_tab')
        self.dataset = self.load_dataset()
        self.dataset_chunked = self.chunk_dataset()
        self.clean_dataset()
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = self.generate_embeddings()
        self.vector_database = self.initialize_vector_database(embeddings)

    def chunk_dataset(self):
        chunked_dataset = {}
        count = 0
        for doc_id in self.dataset:
            for sentence in sentences(self.dataset[doc_id]):
                chunked_dataset[count] = {'sentence': sentence, 'doc_id': doc_id} 
                count += 1
        return chunked_dataset
    
    def clean_dataset(self):
        for i in self.dataset_chunked:
            self.dataset_chunked[i]['sentence'] = cleantext.clean(
                self.dataset_chunked[i]['sentence'],
                extra_spaces=True,
                lowercase=True,
                punct=True
            )
    
    def load_dataset(self):
        dataset = {}
        for doc_id in glob.glob("dataset/*.txt"):
            with open(doc_id) as f:
                dataset[doc_id] = f.read()
        return dataset
    
    def generate_embeddings(self):
        sentences = []
        for i in self.dataset_chunked:
            sentences.append(self.dataset_chunked[i]['sentence'])
        embeddings = self.embedding_model.encode(sentences)
        return embeddings
    
    def initialize_vector_database(self, embeddings):
        index = faiss.index_factory(384, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def get_answer_from_llm(self, context, query):
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""Answer the user's question given the below context
                 
                 {context}"""},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content
    
    def get_answer(self, query):
        query_embeddings = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embeddings)
        _, index = self.vector_database.search(query_embeddings, 5)
        selected_chunk_id = index[0][0]
        selected_doc_id = self.dataset_chunked[selected_chunk_id]['doc_id']
        context = self.dataset[selected_doc_id]
        llm_answer = self.get_answer_from_llm(context, query)
        return llm_answer

app = RnsitRagApplication()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        question = gr.Textbox(label="Ask a Question")
    with gr.Row():
        submit = gr.Button("Submit")
    with gr.Row():
        answer = gr.TextArea(label="Answer")
    
    def ask_question(question):
        return app.get_answer(question)
    
    submit.click(fn=ask_question, inputs=question, outputs=answer)

demo.launch()