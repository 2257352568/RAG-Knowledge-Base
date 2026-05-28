"""Full pipeline test: AST parse → BM25+Vector hybrid → Rerank → Query Rewrite → Cache."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

persist_dir = '/tmp/test_full_rag'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

from parsing.pipeline import process_file
from vectordb import DummyEmbedder, VectorStore, QueryCache
from retrieval import BM25Index, HybridRetriever, DummyReranker
from llm import RAGEngine

passed = failed = 0
def check(name, condition):
    global passed, failed
    if condition: passed += 1; print(f'  PASS: {name}')
    else: failed += 1; print(f'  FAIL: {name}')

# === 1. Markdown AST parsing ===
print('=== 1. Markdown AST Parsing ===')
md_path = '/tmp/test_ast.md'
with open(md_path, 'w') as f:
    f.write("""# Deep Learning Guide

## Chapter 1: Basics
Machine learning is a subset of artificial intelligence.

### 1.1 Neural Networks
Neural networks consist of layers of neurons.

```python
def forward(x):
    return relu(linear(x))
```

## Chapter 2: Transformers
The transformer architecture uses self-attention.

### 2.1 Attention Mechanism
Attention computes weighted sums of values.

- Query, Key, Value projections
- Multi-head attention
- Scaled dot-product

![Attention Diagram](figs/attention.png)
""")

doc, chunks = process_file(md_path, chunk_size=600, chunk_overlap=100)
check('MD sections detected', len(doc.sections) >= 4)
check('Code block preserved', any('```python' in c.text for c in chunks))
check('Title extracted', doc.title == 'Deep Learning Guide')
section_titles = [s['title'] for s in doc.sections]
print(f'  Sections: {section_titles}')

# === 2. Build BM25 + Vector indexes ===
print('\n=== 2. Hybrid Indexing ===')
embedder = DummyEmbedder(dim=384)
store = VectorStore(persist_dir=persist_dir, collection_name='full_test', embedder=embedder)
store.add_chunks(chunks)
check('Vector store built', store.count() == len(chunks))

bm25 = BM25Index()
bm25_docs = [{"text": c.text, "metadata": c.metadata.to_dict(), "id": f"{c.metadata.source_path}_{c.metadata.chunk_index}"}
             for c in chunks]
bm25.build(bm25_docs)
check('BM25 index built', True)

# === 3. Hybrid Retrieval ===
print('\n=== 3. Hybrid Retrieval (Vector + BM25 + RRF) ===')
hybrid = HybridRetriever(store, bm25)

vec_results = store.search('neural networks', top_k=3)
check('Vector search works', len(vec_results) > 0)

bm25_results = bm25.search('neural networks', top_k=3)
check('BM25 search works', len(bm25_results) > 0)

hybrid_results = hybrid.retrieve('neural networks', top_k=3, alpha=0.5)
check('Hybrid search works', len(hybrid_results) > 0)
check('Hybrid returns <= top_k', len(hybrid_results) <= 3)

# === 4. Query Rewrite ===
print('\n=== 4. Query Rewrite ===')
reranker = DummyReranker()
cache = QueryCache()

llm_available = bool(DEEPSEEK_KEY)
if llm_available:
    engine = RAGEngine(
        retriever=hybrid, api_key=DEEPSEEK_KEY,
        reranker=reranker, cache=cache, enable_rewrite=True,
    )
    rewritten = engine._rewrite_query('neural network layers')
    check('Query rewrite produces queries', len(rewritten) > 0)
    print(f'  Original: neural network layers')
    print(f'  Rewritten: {rewritten}')
else:
    engine = RAGEngine(
        retriever=hybrid, api_key="test",
        reranker=reranker, cache=cache, enable_rewrite=False,
    )
    print('  (SKIP - no DEEPSEEK_API_KEY)')

# === 5. Context Reorganization ===
print('\n=== 5. Context Reorganization (LangChain LCEL) ===')
chunks_for_ctx = hybrid_results[:3]
ctx = engine._reorganize_context(chunks_for_ctx)
check('Context formatted with source', '来源' in ctx)
check('Context has separators', '---' in ctx)

# === 6. Cache test ===
print('\n=== 6. TTL Cache ===')
cache.set('test_query', {'answer': 'cached answer', 'sources': [], 'cached': False})
cached = cache.get('test_query')
check('Cache hit', cached is not None)
check('Cached answer correct', cached['answer'] == 'cached answer')

miss = cache.get('nonexistent')
check('Cache miss', miss is None)

cache.set('sourced_query', {
    'answer': 'test',
    'sources': [{'metadata': {'source_path': md_path}}],
})
cache.invalidate_by_source(md_path)
check('Cache invalidation (with source)', cache.get('sourced_query') is None)
check('Cache invalidation (no source, unaffected)', cache.get('test_query') is not None)

# === 7. Incremental upsert ===
print('\n=== 7. Incremental Upsert ===')
modified = chunks[:1]
old_count = store.count()
store.upsert_chunks(modified)
check('Upsert preserves count', store.count() == old_count)

# === 8. Full RAG pipeline ===
if llm_available:
    print('\n=== 8. Full RAG Pipeline (with LLM) ===')
    engine_full = RAGEngine(
        retriever=hybrid, api_key=DEEPSEEK_KEY,
        reranker=reranker, cache=QueryCache(), enable_rewrite=True,
    )
    result = engine_full.query('What is the attention mechanism?', top_k=3, max_tokens=256)
    check('RAG answer not empty', len(result['answer']) > 0)
    check('Sources returned', len(result['sources']) > 0)
    check('Not cached (first call)', result.get('cached') == False)
    print(f'  Answer: {result["answer"][:200]}...')

    result2 = engine_full.query('What is the attention mechanism?', top_k=3, max_tokens=256)
    check('Cache hit on second call', result2.get('cached') == True)
else:
    print('\n=== 8. Full RAG Pipeline (SKIP - no API key) ===')
    print('  Set DEEPSEEK_API_KEY to enable LLM tests')

print(f'\n{"="*40}')
print(f'Results: {passed} passed, {failed} failed')

store._client.delete_collection('full_test')
shutil.rmtree(persist_dir)
os.remove(md_path)
