import pdfplumber
from chunker import chunk_text # Import from your other file
import os

def extract_and_chunk_pdfs(pdf_folder_path: str) -> list[dict]:
    """
    Extracts text from all PDFs in a folder and splits them into chunks.
    Returns a list of dictionaries, where each dict is a chunk.
    """
    print(f"Starting PDF extraction from: {pdf_folder_path}")
    all_chunks = []
    
    # Check if the path is a directory
    if not os.path.isdir(pdf_folder_path):
        print(f"Error: Path '{pdf_folder_path}' is not a valid directory.")
        return []

    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            print(f"Processing: {filename}...")
            
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        
                        if page_text: # If text was successfully extracted
                            page_number = i + 1
                            # Use the chunk_text function
                            chunks = chunk_text(page_text) 
                            
                            for chunk_index, chunk in enumerate(chunks):
                                # Create a dictionary for each chunk
                                all_chunks.append({
                                    "text": chunk,
                                    "metadata": {
                                        "source": filename,
                                        "page": page_number,
                                        "chunk_id": f"{filename}-p{page_number}-c{chunk_index}"
                                    }
                                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks