import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Load API Key ---
# We specify the path to the .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Please check your backend/.env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API Key configured.")

# --- Configure Gemini Model ---
generation_config = {
  "temperature": 0.2, # Low temp for factual medical answers
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def build_rag_prompt(query: str, retrieved_chunks: list[dict], mode: str) -> tuple[str, str]:
    """
    Builds the final prompt for the LLM, including context.
    Returns the prompt string and a formatted sources string.
    """
    
    # 1. Format the retrieved context
    context_str = ""
    sources_str = ""
    seen_sources = set()
    
    for i, chunk in enumerate(retrieved_chunks):
        # Add the text chunk to the context
        context_str += f"\n--- Context {i+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
        context_str += chunk['text']
        
        # Collect unique sources for display
        source_id = f"{chunk['source']} (Page {chunk['page']})"
        if source_id not in seen_sources:
            sources_str += f"- {source_id}\n"
            seen_sources.add(source_id)

    # 2. Select the prompt "persona" based on the mode
    if mode == "Patient":
        persona = "You are a helpful medical assistant. Explain the following to a patient in simple, clear, and empathetic language. Do not use complex medical jargon. Focus on the main points."
    else: # Default to "Doctor" mode
        persona = "You are an expert medical specialty assistant. Provide a concise, accurate, and evidence-based answer for a medical professional. Cite the sources provided."

    # 3. Build the final prompt
    prompt = f"""
    {persona}
    
    You MUST answer the query using ONLY the provided context.
    If the answer is not found in the context, state that clearly.
    
    **QUERY:**
    {query}
    
    **PROVIDED CONTEXT:**
    {context_str}
    
    **ANSWER:**
    """
    
    # Return both the prompt and the formatted sources string
    return prompt, sources_str

def generate_answer(query: str, retrieved_chunks: list[dict], mode: str) -> dict:
    """
    Generates an answer from the LLM using the RAG prompt.
    """
    if not GEMINI_API_KEY:
         return {
            "answer": "Error: Gemini API Key is not configured.",
            "sources": ""
        }
        
    prompt, sources = build_rag_prompt(query, retrieved_chunks, mode)
    
    try:
        print("Generating answer from Gemini...")
        convo = model.start_chat()
        convo.send_message(prompt)
        
        return {
            "answer": convo.last.text,
            "sources": sources
        }
    except Exception as e:
        print(f"Error generating answer from Gemini: {e}")
        return {
            "answer": "Error: Could not generate an answer from the AI model.",
            "sources": sources # Still return the sources
        }