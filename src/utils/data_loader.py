"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings, cache_dir: str = None):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    Tự động lưu và tải lại từ local cache để tiết kiệm quota API.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())
        cache_dir : thư mục lưu FAISS index cache

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from pathlib import Path
    import time
    from langchain_community.vectorstores import FAISS

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"
    else:
        cache_dir = Path(cache_dir)

    if (cache_dir / "index.faiss").exists():
        print(f"Đã tìm thấy FAISS index tại {cache_dir}, đang tải lại...")
        return FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    batch_size = 40
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"   Đang index batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch, embeddings)
                else:
                    vectorstore.add_texts(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    wait_time = 25 * (attempt + 1)
                    print(f"   ⏳ Gặp rate limit quota, đang tạm dừng {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        if i + batch_size < len(chunks):
            time.sleep(5)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))
        print(f"Đã lưu cache FAISS index vào {cache_dir}")
    except Exception as e:
        print(f"Không thể lưu cache FAISS: {e}")

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
