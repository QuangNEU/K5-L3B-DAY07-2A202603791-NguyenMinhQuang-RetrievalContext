"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage

def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    chunks = []
    text = page.text  
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + size
        chunk_text = text[start:end]
        
        chunks.append(Chunk(
            chunk_id=f"{page.document_id}_p{page.page}_{chunk_idx}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata=page.metadata  
        ))
        start += (size - overlap)
        chunk_idx += 1
        
    return chunks

def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata."""
    chunks = []
    text = page.text
    if not text:
        return chunks
        
    paragraphs = text.split('\n\n')
    current_chunk_text = ""
    chunk_idx = 0 
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(para) > max_size:
            if current_chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"{page.document_id}_p{page.page}_{chunk_idx}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=current_chunk_text,
                    metadata=page.metadata
                ))
                chunk_idx += 1
                current_chunk_text = ""
            
            for i in range(0, len(para), max_size):
                chunks.append(Chunk(
                    chunk_id=f"{page.document_id}_p{page.page}_{chunk_idx}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=para[i:i + max_size],
                    metadata=page.metadata
                ))
                chunk_idx += 1
            continue
            
        if current_chunk_text and len(current_chunk_text) + len(para) + 2 > max_size:
            chunks.append(Chunk(
                chunk_id=f"{page.document_id}_p{page.page}_{chunk_idx}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=current_chunk_text,
                metadata=page.metadata
            ))
            chunk_idx += 1
            current_chunk_text = para
        else:
            if current_chunk_text:
                current_chunk_text += "\n\n" + para
            else:
                current_chunk_text = para
                
    if current_chunk_text:
        chunks.append(Chunk(
            chunk_id=f"{page.document_id}_p{page.page}_{chunk_idx}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=current_chunk_text,
            metadata=page.metadata
        ))
        
    return chunks