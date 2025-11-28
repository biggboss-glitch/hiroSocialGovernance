"""
文本提取模块
支持从 .md, .txt, .pdf 文件中提取纯文本
"""

import os
from pathlib import Path
from typing import Optional


def extract_from_txt(file_path: str) -> str:
    """从TXT文件提取文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_from_md(file_path: str) -> str:
    """从Markdown文件提取文本（保留原始格式，不转换HTML）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_from_pdf(file_path: str) -> str:
    """从PDF文件提取文本"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")
    
    text_parts = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
    
    return "\n\n".join(text_parts)


def extract_text(file_path: str) -> str:
    """
    根据文件扩展名自动选择提取方法
    
    Args:
        file_path: 文件路径
        
    Returns:
        提取的纯文本内容
        
    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = path.suffix.lower()
    
    extractors = {
        '.txt': extract_from_txt,
        '.md': extract_from_md,
        '.markdown': extract_from_md,
        '.pdf': extract_from_pdf,
    }
    
    extractor = extractors.get(suffix)
    if extractor is None:
        supported = ', '.join(extractors.keys())
        raise ValueError(f"不支持的文件格式: {suffix}。支持的格式: {supported}")
    
    return extractor(file_path)


def split_text_into_chunks(text: str, max_chunk_size: int = 8000, overlap: int = 200) -> list[str]:
    """
    将长文本分割成多个小块，适合Zep处理
    
    Args:
        text: 原始文本
        max_chunk_size: 每个块的最大字符数
        overlap: 块之间的重叠字符数
        
    Returns:
        文本块列表
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # 尝试在句子边界处分割
        if end < len(text):
            # 查找最近的句子结束符
            for sep in ['。', '！', '？', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > max_chunk_size * 0.5:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 下一个块从重叠位置开始
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        text = extract_text(file_path)
        print(f"提取了 {len(text)} 个字符")
        print(f"前500字符:\n{text[:500]}")



