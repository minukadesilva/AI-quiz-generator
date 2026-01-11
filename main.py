import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from fpdf import FPDF

# Convert quiz to pdf
def make_my_pdf(quiz_stuff):
    pdf = FPDF()
    pdf.add_page()
    
    # --- PAGE 1: THE QUESTIONS ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "EXAM PAPER", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12)
    
    # Safe margin from the left
    x_offset = 20 

    for i, q in enumerate(quiz_stuff.questions):
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 10, f"{i+1}. {q.question}")
        
        pdf.set_font("Helvetica", size=11)
        letters = ["A", "B", "C", "D"]
        
        for index, opt in enumerate(q.options):
            # We clean the option text just in case AI added "A." or "A)" already
            clean_opt = opt.replace(f"{letters[index]})", "").replace(f"{letters[index]}.", "").strip()
            
            # Move the "cursor" to the right before printing
            pdf.set_x(x_offset) 
            pdf.cell(0, 8, f"{letters[index]}) {clean_opt}", ln=True)
        
        pdf.ln(5)

    # --- PAGE 2: ANSWER KEY ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ANSWER KEY", ln=True, align='C')
    pdf.ln(10)
    
    for i, q in enumerate(quiz_stuff.questions):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(x_offset)
        pdf.write(10, f"Q{i+1} Correct Answer: {q.answer}\n")
        
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_x(x_offset)
        pdf.write(8, f"Explanation: {q.explanation}\n")
        pdf.ln(8)
        
    return pdf.output()

load_dotenv() 

# Setup
class MyQuestion(BaseModel):
    question: str = Field(description="the text of the question")
    options: List[str] = Field(description="list of 4 options")
    answer: str = Field(description="the letter of the correct answer")
    explanation: str = Field(description="why it is correct")

class MyQuiz(BaseModel):
    questions: List[MyQuestion]

# UI
st.title("AI Quiz Maker")
st.write("Upload a PDF and I will make a quiz for you!")

num = st.sidebar.slider("How many questions do you need?", 1, 10, 5)

my_file = st.file_uploader("Upload your PDF", type="pdf")

if my_file is not None:
    #Temporary save
    with open("temp.pdf", "wb") as f:
        f.write(my_file.getbuffer())

    if st.button("Generate Quiz"):
        st.write("Please wait...")
        
        #LangChain
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        text_for_ai = ""
        for i in range(min(5, len(chunks))):
            text_for_ai += chunks[i].page_content
            
        # AI used
        brain = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        quiz_brain = brain.with_structured_output(MyQuiz)
        
        # Prompt
        template = """
        You are a teacher. Make a quiz with {count} questions from this text.
        Make sure you give me 4 options for every question.
        TEXT: {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = prompt | quiz_brain
        result = chain.invoke({"context": text_for_ai, "count": num})
        
        # Result
        st.session_state.my_quiz = result

        st.success("Done!")

        for i, item in enumerate(result.questions):
            st.write("---")
            st.markdown(f"**Question {i+1}: {item.question}**")
            letters = ["A", "B", "C", "D"]
            for index, opt in enumerate(item.options):
                st.write(f"{letters[index]}) {opt}")
            
            with st.expander("See the answer"):
                st.write(f"Answer: {item.answer}")
                st.write(f"Explanation: {item.explanation}")

        #Download Button
        st.write("### Download section")
        final_pdf = make_my_pdf(result)
        st.download_button("Download PDF", data=bytes(final_pdf), file_name="quiz.pdf")

    # Delete the file
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")