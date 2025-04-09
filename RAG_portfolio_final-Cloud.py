#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pymupdf
pip install faiss-cpu
pip install sentence_transformers
pip install streamlit
pip install transformers


# In[1]:


import pandas as pd
import numpy as np
import fitz
import re
from sentence_transformers import SentenceTransformer


# In[2]:


import faiss
import pickle


# In[5]:


resume_dict= {
    "CONTACT": {
        "NAME": "Juan Medina",
        "CITY": "Toronto, Canada",
        "EMAIL": "juanandresmedina125@gmail.com"
    },
    "RELEVANT WORK EXPERIENCE": {
        "Vector Institute": {
            "CITY": "Toronto, ON",
            "ROLE": "Faculty Affiliate Researcher",
            "DATE": "Sep 2023 - Feb 2025",
            "DESCRIPTION": "Developed and implemented a comprehensive Natural Language Processing and Machine Learning pipeline utilizing large language models (LLM) in few shot, pre-training and fine-tuning settings, resulting in valuable policy insights on post-COVID conditions (PCC). Implemented a novel named entity recognition (NER) tool for identifying social determinants of PCC, facilitating the analysis of unexplored relationships between PCC and sociodemographic dimensions. Designed and executed a language entailment pipeline to automate a granular analysis of annotated data, providing actionable insights for over 26 SDOH entity dimensions in over 7,000 texts."
        },
        "Rubik": {
            "CITY": "London, UK (Remote)",
            "ROLE": "AI Product Strategy [Consulting Project]",
            "DATE": "Sep 2024 - Dec 2024",
            "DESCRIPTION": "Developed a comprehensive business intelligence AI implementation strategy tailored to the waste management sector, including scalability in cloud systems, interoperability, and product differentiation."
        },
        "J. Roy Gillis Lab, University of Toronto": {
            "CITY": "Toronto, ON",
            "ROLE": "Data Science, Quantitative Analysis Specialist",
            "DATE": "Jun 2024 - Dec 2024",
            "DESCRIPTION": "Designed an end-to-end sentiment analysis pipeline to analyze discourse around vaccination hesitancy in Canada, entailing the extraction, cleaning, annotation, modeling and visualization of over 100,000 data points from the Reddit API. Led an interdisciplinary team of 8 researchers, including engineers, social scientists and designers. Created an interactive visual story showcasing key trends and contextual patterns related to vaccination hesitancy, enhancing understanding and decision-making for stakeholders."
        },
        "i4Health Research lab, York University": {
            "CITY": "Toronto, ON",
            "ROLE": "Machine Learning Research Assistant",
            "DATE": "Oct 2023 - Sep 2024",
            "DESCRIPTION": "Explored and developed ML-driven disparity analysis pipelines with Natural Language Processing and Causal Inference to assess discriminatory relationships in health. Collaborated in the development of a question-answering model for medical images."
        }
    },
    "EDUCATION": {
        "University of Toronto": {
            "CITY": "Toronto, ON, CA",
            "DEGREE": "Master of Science (M.Sc.): Health Systems Artificial Intelligence emphasis",
            "START DATE": "Sep 2023",
            "GRADUATION DATE": "Mar 2025"
        },
        "University of California, San Diego": {
            "CITY": "Remote",
            "DEGREE": "Coursework in Object-Oriented Programming, Natural Language Processing, Probability and Statistics for Deep Learning, and Discrete Mathematics",
            "START DATE": "Jun 2022",
            "GRADUATION DATE": "Nov 2022"
        },
        "Wesleyan University": {
            "CITY": "Middletown, CT, USA",
            "DEGREE": "Bachelor of Arts Double Major: Economics, Science in Society Program (Mathematics/Neuroscience & Sociology emphases)",
            "START DATE": "Aug 2018",
            "GRADUATION DATE": "May 2022"
        }
    },
    "SKILLS": {
        "Technical": "Python, R, SQL, Tableau, SAS, SLURM (HPC), AWS, Spark, PowerBI, Stata",
        "Relevant Courses": "Machine Learning, Deep Learning, Statistical Learning, Data Visualization, Causal Inference, AI Implementation, Biostatistics, Innovation Management, Health Policy"
    }
}


# In[6]:


resume_chunks=[]
for k, val in resume_dict.items():
  string= str(k) + ': ' + str(val) +' }}'
  resume_chunks.append(string)


# In[8]:


# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
resume_embeddings = model.encode(resume_chunks)

print("Embedding shape:", resume_embeddings.shape)  # Should be (num_chunks, 384)


# In[9]:


# Initialize FAISS index
dimension = resume_embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)
index.add(resume_embeddings)

# Save index and text chunks for future use
with open("resume_faiss.pkl", "wb") as f:
    pickle.dump((index, resume_chunks), f)

print("FAISS index built and saved.")


# In[10]:


# Load the saved FAISS index and text chunks
with open("resume_faiss.pkl", "rb") as f:
    index, resume_chunks = pickle.load(f)


# In[11]:


import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# In[12]:


embedder = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


# In[14]:


def retrieve_and_generate(query, top_k=2):
    # Embed the query
    query_embedding = embedder.encode([query])

    # Search FAISS for top matching chunks
    distances, indices = index.search(query_embedding, k=top_k)
    retrieved_texts = [resume_chunks[i] for i in indices[0] if i < len(resume_chunks)]

    if not retrieved_texts:
        return "Sorry, I couldn't find relevant information in the resume."

    # Combine context
    context = "\n".join(retrieved_texts)

    # Manually build the prompt for Qwen2.5
    prompt = (
        "You are Juan, a recent master's graduate. Based on your resume information below (in python dictionary format), "
        "answer the user's question truthfully and concisely in first person, checking for the right key in the dictionary. Let's think step by step.\n\n"
        f"Resume:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # Tokenize and generate
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    output = llm.generate(input_ids, max_new_tokens=200)

    # Only decode the newly generated part
    generated_tokens = output[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


# In[16]:


import streamlit as st


# In[ ]:


qa_dict = {"Work experience": "My experience spans both academia and industry, including my role as a Faculty Affiliate Researcher at the Vector Institute for Artificial Intelligence, where I focused on Natural Language Processing in healthcare, and as an AI strategy consultant for Rubik, a London-based tech startup.",
           "Education": "I recently earned an M.Sc. with an emphasis in Health Systems Artificial Intelligence from the University of Toronto and hold a B.A. in Economics and Science & Technology Studies from Wesleyan University.",
           "Skillset": "With a multidisciplinary background in AI, statistics, economics, and health systems, I bring a unique perspective to solving complex problems.",
           "Other": "Please reach out directly at juanandresmedina125@gmail.com, and I would be delighted to discuss my background and experiencesin greater detail."
}


# In[17]:


# 🧠 Title and Intro
st.title("🧠 Ask My Resume")
st.write("Hello! What would you like to know about my experience?")



# ➤ Section: Predefined questions
st.subheader("🔸 Topics")
selected_question = st.selectbox("Choose a topic:", ["Select..."] + list(qa_dict.keys()))

if selected_question != "Select...":
    st.markdown(f"**Answer:** {qa_dict[selected_question]}")

# ➤ Section: Custom question
st.subheader("🔍 Ask Your Own Question")
query = st.text_input("Enter your question here")

if query:
    with st.spinner("Thinking..."):
        answer = retrieve_and_generate(query)
        st.success(answer)


# In[ ]:




