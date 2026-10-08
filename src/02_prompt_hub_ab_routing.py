"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# TODO: Đổi thành tên của bạn — phải là duy nhất trong Hub của bạn
PROMPT_V1_NAME = "bui-le-gia-huy-rag-prompt-v1"
PROMPT_V2_NAME = "bui-le-gia-huy-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# TODO: Viết SYSTEM_V1 — phong cách ngắn gọn, trả lời 2-4 câu
# Gợi ý: "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.
#          Giữ câu trả lời ngắn gọn (2-4 câu). ...\n\nContext:\n{context}"
# ⚠️ BẮT BUỘC có {context} trong SYSTEM — thiếu thì LLM không nhận được tài liệu
#    mà chương trình KHÔNG báo lỗi (câu trả lời bịa, điểm RAGAS thấp).
SYSTEM_V1 = (
    "Bạn là một trợ lý AI hữu ích và súc tích. Hãy trả lời câu hỏi dựa CHỈ vào ngữ cảnh được cung cấp. "
    "Giữ câu trả lời ngắn gọn, trực diện trong khoảng 2-4 câu, không lan man.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# TODO: Viết SYSTEM_V2 — phong cách có cấu trúc, expert tone, 3-5 câu
# Gợi ý: "Bạn là chuyên gia AI. Đọc kỹ context, xác định facts liên quan,
#          viết câu trả lời rõ ràng và có tổ chức (3-5 câu). ...\n\nContext:\n{context}"
# ⚠️ BẮT BUỘC có {context} (giống SYSTEM_V1)
SYSTEM_V2 = (
    "Bạn là một chuyên gia AI giàu kinh nghiệm. Hãy đọc kỹ ngữ cảnh, phân tích các thông tin liên quan "
    "và đưa ra câu trả lời có cấu trúc rõ ràng, chuyên sâu và mang tính học thuật (3-5 câu), có thể sử dụng gạch đầu dòng nếu phù hợp.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    # TODO: Push PROMPT_V1 — bọc trong try/except để xử lý lỗi
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn, súc tích (2-4 câu)")
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    # TODO: Push PROMPT_V2 — bọc trong try/except
    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc, giọng chuyên gia (3-5 câu)")
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    # TODO: Pull PROMPT_V1_NAME, fallback về PROMPT_V1 nếu lỗi
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}'")

    # TODO: Pull PROMPT_V2_NAME, fallback về PROMPT_V2 nếu lỗi
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).

    Gợi ý:
        hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME
    """
    # TODO: Tính MD5 hash của request_id và chuyển thành số nguyên
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # TODO: Trả về PROMPT_V1_NAME nếu chẵn, PROMPT_V2_NAME nếu lẻ
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
# TODO: Thêm @traceable(name="ab-rag-query", tags=["ab-test", "step2"])
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    # TODO: Retrieve docs từ retriever
    docs = retriever.invoke(question)

    # TODO: Ghép page_content thành 1 string (dùng "\n\n".join)
    context = "\n\n".join(doc.page_content for doc in docs)

    # TODO: Chạy chain và lấy answer
    chain = prompt | llm | StrOutputParser()

    import time
    for attempt in range(5):
        try:
            answer = chain.invoke({"context": context, "question": question})
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                wait_time = 5 * (attempt + 1)
                print(f"    Rate limit, đang chờ {wait_time}s để thử lại...")
                time.sleep(wait_time)
            else:
                raise

    # TODO: Trả về dict kết quả
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo LangSmith Client với API key từ config
    # Gợi ý: client = Client(api_key=config.LANGSMITH_API_KEY)
    client = Client(api_key=config.LANGSMITH_API_KEY)

    # TODO: Push cả 2 prompts lên Hub
    push_prompts_to_hub(client)

    # TODO: Pull cả 2 prompts từ Hub (dùng dict trả về)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    # TODO: Tạo retriever từ vectorstore (k=3)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    import time
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        # TODO: Lấy version key từ request_id qua get_prompt_version()
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        # TODO: Gọi ask_ab() với đúng arguments
        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
        if config.PROVIDER == "gemini":
            time.sleep(3)

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
