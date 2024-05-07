import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM

documents = SimpleDirectoryReader("data/").load_data()

#from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.core import PromptTemplate

system_prompt = "You are a question and answer assistant who speaks Turkish and works in the banking sector. Using the data you have obtained, can you write one question whose answers are included in this data? Let your writing language be Turkish."


# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.1,"top_p":0.0, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="tiiuae/falcon-7b",
    model_name="tiiuae/falcon-7b",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

# Commented out IPython magic to ensure Python compatibility.
# %pip install llama-index-embeddings-langchain

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import ServiceContext
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name= "sentence-transformers/LaBSE")
                        # "intfloat/multilingual-e5-large")
                        #"loodos/electra-small-turkish-cased-discriminator")
                        #"sentence-transformers/distiluse-base-multilingual-cased-v2")
                        #"sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
                        #"intfloat/multilingual-e5-large")
                        #"distilbert-base-multilingual-cased"
)

service_context = ServiceContext.from_defaults(
    chunk_size=4096,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
questions = ["Bankalarca bir gerçek ya da tüzel kişiye veya bir risk grubuna kullandırılabilecek kredilerin risk tutarları toplamı ne kadardır?",
             "Sorunlu alacakların çözümlenmesi açısından ilk savunma hattı nedir?",
             "Sorunlu alacak ölçütleri nedir?"]
for q in questions:
  print("Soru:", q)
  response = query_engine.query(q)
  print("Cevap:", response)

