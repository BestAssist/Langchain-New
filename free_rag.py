from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# No API key needed 🚀
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

generator = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=generator)
