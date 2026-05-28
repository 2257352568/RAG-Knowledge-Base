"""End-to-end RAG test: parse → embed → store → retrieve → LLM answer."""
import sys, os, shutil
sys.path.insert(0, '.')

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

persist_dir = '/tmp/test_rag_db'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

from parsing.pipeline import process_file
from vectordb import DummyEmbedder, VectorStore, Retriever
from llm import DeepSeekLLM, RAGEngine

passed = failed = 0
def check(name, condition):
    global passed, failed
    if condition: passed += 1; print(f'  PASS: {name}')
    else: failed += 1; print(f'  FAIL: {name}')

# === 1. Build knowledge base from test PDF ===
print('=== 1. Build Knowledge Base ===')
doc, chunks = process_file('/tmp/test_paper.pdf', chunk_size=600, chunk_overlap=100)
print(f'  Loaded PDF: {doc.pages} pages, {len(chunks)} chunks')

# Use DummyEmbedder for speed (offline, no API)
embedder = DummyEmbedder(dim=384)
store = VectorStore(persist_dir=persist_dir, collection_name='rag_test', embedder=embedder)
store.add_chunks(chunks)
check('Index built', store.count() == 5)

# For demo purposes, verify retrieval still works
retriever = Retriever(store)
results = retriever.retrieve('What methodology was used?', top_k=3)
check('Retrieval works', len(results) == 3)

# === 2. Initialize LLM and RAG Engine ===
print('\n=== 2. RAG Engine with DeepSeek ===')
llm = DeepSeekLLM(api_key=DEEPSEEK_KEY, model='deepseek-chat')
engine = RAGEngine(retriever, llm)
check('RAG engine initialized', engine is not None)

# === 3. Test basic Q&A ===
print('\n=== 3. Basic Q&A ===')
result = engine.query(
    '这篇论文的研究方法是什么？包括哪些数据库？',
    top_k=3,
    max_tokens=512,
)
answer = result['answer']
sources = result['sources']
print(f'  Answer: {answer[:300]}...')
print(f'  Sources used: {len(sources)}')
for i, s in enumerate(sources):
    meta = s['metadata']
    print(f'    [{i+1}] section="{meta.get("section_title")}" | score={s["score"]}')

check('Answer is not empty', len(answer) > 0)
check('Sources returned', len(sources) == 3)
check('Methodology mentioned', 'methodology' in answer.lower() or '方法' in answer or 'PRISMA' in answer or '系统' in answer)

# === 4. Test "out of scope" question (should not hallucinate) ===
print('\n=== 4. Out-of-scope question (hallucination check) ===')
result2 = engine.query(
    '这篇论文中提到量子计算机的内容有哪些？',
    top_k=3,
    max_tokens=256,
)
print(f'  Answer: {result2["answer"][:200]}...')
# Should indicate no relevant info was found
check('No hallucination on OOS question',
     '没有' in result2['answer'] or '未提及' in result2['answer'] or '未找到' in result2['answer'] or '没有提供' in result2['answer'] or '没有相关' in result2['answer'])

# === 5. Test streaming ===
print('\n=== 5. Streaming ===')
chunks_received = []
for token in engine.query_stream('什么是machine learning?', top_k=2, max_tokens=100):
    chunks_received.append(token)
full = ''.join(chunks_received)
check('Streaming produces output', len(full) > 0)
print(f'  Streamed: {full[:150]}...')

# === 6. Verify prompt structure ===
print('\n=== 6. Prompt structure ===')
messages = result['prompt_messages']
check('Has system message', messages[0]['role'] == 'system')
check('Has user message', messages[1]['role'] == 'user')
check('System msg contains context', '参考资料' in messages[0]['content'])
check('Context contains source info', '来源' in messages[0]['content'] or '/tmp/test_paper.pdf' in messages[0]['content'])

# Summary
print(f'\n{"="*40}')
print(f'Results: {passed} passed, {failed} failed')

# Cleanup
store._client.delete_collection('rag_test')
shutil.rmtree(persist_dir)
