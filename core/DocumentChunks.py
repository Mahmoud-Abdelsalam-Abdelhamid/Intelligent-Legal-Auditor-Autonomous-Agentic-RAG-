import json
from langchain_text_splitters import (RecursiveCharacterTextSplitter, 
                                      MarkdownHeaderTextSplitter)

def clean_markdown(text):
    # (Your existing cleaning logic is fine, keeping it same)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('!['): continue
        if line.strip().startswith('* [') or line.strip().startswith('+ ['): continue
        if line.strip() == "*": continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def chunk_legal_text(raw_text, company_name, source_url):
    print(" - Splitting text into chunks...")
    
    # 1. MARKDOWN SPLITTING (Kept same)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    md_header_splits = md_splitter.split_text(raw_text)
    print(f"   > Created {len(md_header_splits)} sections via Markdown headers.")
    
    # 2. RECURSIVE SPLITTING (Optimized)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,  # <--- INCREASED from 200 to 400 (Critical for Legal)
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(md_header_splits)
    print(f"   > Created {len(final_chunks)} total chunks.")
    
    # 3. METADATA & CONTEXT INJECTION (The Accuracy Booster)
    for doc in final_chunks:
        doc.metadata["company"] = company_name
        doc.metadata["url"] = source_url
        
        # Extract headers from metadata to inject into content
        # This fixes "The Orphaned Chunk" problem by forcing context into every chunk
        h1 = doc.metadata.get("Header 1", "")
        h2 = doc.metadata.get("Header 2", "")
        h3 = doc.metadata.get("Header 3", "")
        
        # We prepend the section title to the text content.
        # Even if the chunk is just "we will provide notice", 
        # it now becomes: "Context: Termination Policy. we will provide notice"
        # This makes it searchable!
        context_header = f"Context: {h1} > {h2} > {h3}\n"
        doc.page_content = context_header + doc.page_content
    
    return final_chunks