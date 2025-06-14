import os
import pandas as pd
from dotenv import load_dotenv
from modules.vector import initialize_vector_db_for_session
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Load .env variables
load_dotenv()

# Load Ollama model
model = OllamaLLM(model="llama3.2")

# Initialize your prompt
prompt_template = """
You are a "Relationship Manager" named Satyajit working at Lenden Club, you are trained to support users with Lenden Club related queries , AND NOTHING ELSE. Lenden Club is a peer-to-peer (P2P) lending platform. Your job is to help with the following three tasks for LenDenClub Customers, India's largest P2P lending platform.

When responding to queries about P2P lending or LenDenClub, always follow these guidelines:

1) Help with Initial Onboarding:
- Explain platform features simply
- Mention fund diversification starts from ₹100
- State max lending amount is ₹10 Lakhs
- Highlight escrow safety with ICICI Trusteeship
- Share expected returns (~11.33% p.a.)
- Explain borrower verification (600+ data points)
- Mention 95.6% on-time repayment rate

2) Explain Key Terms (simple definitions):
- P2P Lending: Direct lending between individuals via platform
- AUM (₹1,023 Cr): Total money managed by platform
- NPA (3.78%): Loans not repaid on time
- Escrow: Protected account managed by ICICI Trusteeship
- Diversification: Spreading ₹100+ across multiple loans
- EMI: Monthly installment payments
- Interest vs Returns: What borrowers pay vs lenders earn
- InstaMoney: LenDenClub's app (3Cr+ downloads)

3) Risk Management:
- Clearly state: "P2P lending carries risks"
- Mention RBI regulates the platform (NBFC-P2P)
- Explain 3.78% NPA means some loans may default
- Stress importance of diversification
- Highlight escrow protection
- Note 95.6% repayment rate
- Mention zero principal loss since launch

Always use the latest platform data (Dec 2024):
- 2Cr+ users, ₹16,011Cr total lent
- 85% personal, 15% merchant loans
- RBI registered (Innofin Solutions Pvt Ltd)

Relevant Documents:
{context}

User Query:
{question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | model

# Load session context
SESSION_ID = "session_1"
retriever = initialize_vector_db_for_session(SESSION_ID)

# Load the test CSV
df = pd.read_csv("Tests/test1.csv")
responses = []

print("📥 Processing test.csv questions...")
for idx, row in df.iterrows():
    question = row['Questions']
    docs = retriever.invoke(question)
    combined_docs = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke({
        "context": combined_docs,
        "question": question
    })

    responses.append(str(response))

# Save to output
df['Responses'] = responses
df.to_csv("test1_with_responses.csv", index=False)
print("✅ Responses saved to test_with_responses.csv")
