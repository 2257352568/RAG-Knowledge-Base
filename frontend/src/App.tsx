import { useState, useRef, useEffect } from 'react';

interface Message { id: string; role: 'user' | 'assistant'; content: string; }
interface DocInfo { name: string; source_path: string; chunks: number }
interface Settings { llm_api_key: string; llm_model: string; llm_base_url: string; top_k: number; rerank_enabled: boolean; stream_enabled: boolean }

const API = '/api';
async function api(path: string, opts?: RequestInit) {
  const res = await fetch(API + path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const [page, setPage] = useState<'chat' | 'docs' | 'settings'>('chat');
  const [msgs, setMsgs] = useState<Message[]>([]);   // lifted state — survives page switches

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* ---- Sidebar ---- */}
      <aside style={{ width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column', background: '#f9fafb', borderRight: '1px solid #e5e7eb' }}>
        {/* Brand */}
        <div style={{ padding: '20px 16px 12px', borderBottom: '1px solid #e5e7eb' }}>
          <h1 style={{ fontSize: 15, fontWeight: 600, color: '#111827', lineHeight: 1.4 }}>
            面向个人的RAG<br />知识库问答系统
          </h1>
          <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>混合检索 · 智能问答</p>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: 8, display: 'flex', flexDirection: 'column', gap: 2, overflowY: 'auto' }}>
          {([
            { id: 'chat' as const, label: '对话', emoji: '💬' },
            { id: 'docs' as const, label: '文档管理', emoji: '📁' },
            { id: 'settings' as const, label: '设置', emoji: '⚙️' },
          ]).map(({ id, label, emoji }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 12px', borderRadius: 8, border: 'none',
                background: page === id ? '#e5e7eb' : 'transparent',
                color: page === id ? '#111827' : '#4b5563',
                fontWeight: page === id ? 500 : 400,
                fontSize: 14, cursor: 'pointer', textAlign: 'left', width: '100%',
              }}
            >
              <span style={{ width: 20, textAlign: 'center' }}>{emoji}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div style={{ padding: 12, borderTop: '1px solid #e5e7eb', fontSize: 11, color: '#9ca3af', textAlign: 'center' }}>
          v1.0 · RAG 本地知识库
        </div>
      </aside>

      {/* ---- Main (keep all views mounted, show/hide via display) ---- */}
      <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ display: page === 'chat' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
          <ChatView msgs={msgs} setMsgs={setMsgs} />
        </div>
        {page === 'docs' && <DocsView />}
        {page === 'settings' && <SettingsView />}
      </main>
    </div>
  );
}

// ========================= Chat =========================

const WELCOME_QS = [
  '什么是 RAG？它和微调有什么区别？',
  'Transformer 的注意力机制是如何工作的？',
  'ChromaDB 的 HNSW 索引原理是什么？',
];

function ChatView({ msgs, setMsgs }: { msgs: Message[]; setMsgs: (v: Message[] | ((p: Message[]) => Message[])) => void }) {
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);

  const send = async (text?: string) => {
    const q = (text || input).trim();
    if (!q || sending) return;
    const uid = Date.now().toString();
    const aid = (Date.now() + 1).toString();
    setMsgs(prev => [...prev, { id: uid, role: 'user', content: q }, { id: aid, role: 'assistant', content: '' }]);
    setInput('');
    setSending(true);

    // Schedule SSE streaming OUTSIDE React event context
    // so React 18 auto-batching doesn't merge all token renders.
    setTimeout(() => { streamChat(q, aid); }, 0);
  };

  async function streamChat(q: string, aid: string) {
    try {
      const res = await fetch(API + '/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: 5, stream: true }),
      });
      if (!res.ok) throw new Error(await res.text());
      const reader = res.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const d = JSON.parse(line.slice(6));
            if (d.status === 'thinking') {
              setMsgs(prev => prev.map(m => m.id === aid ? { ...m, content: '思考中...' } : m));
            }
            if (d.token) {
              setMsgs(prev => prev.map(m => m.id === aid ? {
                ...m, content: m.content === '思考中...' ? d.token : m.content + d.token
              } : m));
            }
            if (d.error) {
              setMsgs(prev => prev.map(m => m.id === aid ? { ...m, content: d.error } : m));
              setSending(false);
              return;
            }
          } catch { /* */ }
        }
      }
    } catch (e: any) {
      setMsgs(prev => prev.map(m => m.id === aid ? { ...m, content: `错误: ${e.message}` } : m));
    }
    setSending(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {msgs.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '0 24px' }}>
            <div style={{ textAlign: 'center', maxWidth: 640, width: '100%' }}>
              <h2 style={{ fontSize: 24, fontWeight: 600, color: '#1f2937', marginBottom: 8 }}>知识库问答</h2>
              <p style={{ fontSize: 14, color: '#6b7280', marginBottom: 32 }}>
                基于 RAG 检索增强生成，从文档中检索并生成精准回答
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {WELCOME_QS.map(q => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    style={{
                      width: '100%', textAlign: 'left', padding: '14px 20px',
                      borderRadius: 12, border: '1px solid #e5e7eb', background: '#fff',
                      fontSize: 14, color: '#4b5563', cursor: 'pointer',
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ maxWidth: 768, margin: '0 auto', width: '100%', padding: '24px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {msgs.map(msg => (
                <div key={msg.id} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  <div style={{
                    maxWidth: '80%', borderRadius: 16,
                    padding: '14px 20px', fontSize: 15, lineHeight: 1.65,
                    background: msg.role === 'user' ? '#f3f4f6' : 'transparent',
                    color: '#1f2937',
                    borderBottomRightRadius: msg.role === 'user' ? 4 : 16,
                    borderBottomLeftRadius: msg.role === 'user' ? 16 : 4,
                  }}>
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content || (
                      <span style={{ display: 'inline-block', width: 2, height: 16, background: '#9ca3af', animation: 'pulse 1s infinite', borderRadius: 2 }} />
                    )}</div>
                  </div>
                </div>
              ))}
              <div ref={endRef} />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div style={{ flexShrink: 0, padding: '0 24px 24px' }}>
        <div style={{ maxWidth: 768, margin: '0 auto', width: '100%', position: 'relative' }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="向知识库提问…"
            rows={1}
            disabled={sending}
            style={{
              width: '100%', resize: 'none', background: '#fff',
              border: '1px solid #d1d5db', borderRadius: 12,
              padding: '14px 48px 14px 16px', fontSize: 15,
              color: '#1f2937', lineHeight: 1.5,
              fontFamily: 'inherit', outline: 'none',
              boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
              minHeight: 52, boxSizing: 'border-box',
            }}
          />
          <button
            onClick={() => send()}
            disabled={sending || !input.trim()}
            style={{
              position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
              width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center',
              borderRadius: 8, border: 'none',
              background: sending || !input.trim() ? '#e5e7eb' : '#111827',
              color: sending || !input.trim() ? '#9ca3af' : '#fff',
              cursor: sending || !input.trim() ? 'default' : 'pointer',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
          </button>
        </div>
        <p style={{ fontSize: 11, color: '#9ca3af', textAlign: 'center', marginTop: 10 }}>
          RAG 知识库 · 答案基于已索引的文档生成
        </p>
      </div>
    </div>
  );
}

// ========================= Docs =========================

function DocsView() {
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async () => { setLoading(true); try { setDocs(await api('/documents')); } catch { } setLoading(false); };
  useEffect(() => { load(); }, []);

  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');

  const upload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    setUploading(true);
    setUploadMsg(`上传 ${e.target.files.length} 个文件中...`);
    const fd = new FormData();
    for (const f of e.target.files) fd.append('files', f);
    try {
      const res = await fetch('http://localhost:8000/api/ingest', { method: 'POST', body: fd });
      const data = await res.json();
      if (res.ok && data.errors.length === 0) {
        setUploadMsg(`成功: ${data.indexed} 个文件, ${data.total_chunks} 个分块`);
        load();
      } else if (res.ok && data.errors.length > 0) {
        setUploadMsg(`部分失败: ${data.errors.join('; ')}`);
        load();
      } else {
        setUploadMsg(`失败: ${data.detail || res.statusText}`);
      }
    } catch (err: any) {
      setUploadMsg(`网络错误: ${err.message}`);
    }
    setUploading(false);
    e.target.value = '';
  };

  const del = async (name: string) => {
    if (!confirm(`删除 "${name}"？`)) return;
    await fetch(API + `/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
    load();
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      <div style={{ maxWidth: 640, margin: '0 auto', padding: '32px 24px' }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, color: '#1f2937', marginBottom: 24 }}>文档管理</h1>

        <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb', padding: 20, marginBottom: 24 }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 20px', background: uploading ? '#9ca3af' : '#111827', color: '#fff', borderRadius: 8, fontSize: 14, cursor: uploading ? 'default' : 'pointer', border: 'none' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>
            {uploading ? '处理中...' : '上传文件'}
            <input type="file" accept=".pdf,.md,.markdown" multiple onChange={upload} disabled={uploading} style={{ display: 'none' }} />
          </label>
          {uploadMsg && <p style={{ fontSize: 12, color: uploadMsg.startsWith('成功') ? '#16a34a' : '#dc2626', marginTop: 8 }}>{uploadMsg}</p>}
          {!uploadMsg && <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 8 }}>支持 PDF · Markdown</p>}
        </div>

        <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb' }}>
          {loading ? <div style={{ padding: 32, textAlign: 'center', fontSize: 14, color: '#9ca3af' }}>加载中…</div>
          : docs.length === 0 ? <div style={{ padding: 32, textAlign: 'center', fontSize: 14, color: '#9ca3af' }}>暂无文档</div>
          : docs.map(d => (
            <div key={d.source_path} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 20px', borderBottom: '1px solid #f3f4f6' }}>
              <div><p style={{ fontSize: 14, fontWeight: 500, color: '#1f2937' }}>{d.name}</p><p style={{ fontSize: 12, color: '#9ca3af', marginTop: 2 }}>{d.chunks} 个分块</p></div>
              <button onClick={() => del(d.name)} style={{ fontSize: 13, color: '#9ca3af', background: 'none', border: 'none', cursor: 'pointer' }}>删除</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ========================= Settings =========================

function SettingsView() {
  const [s, setS] = useState<Settings | null>(null);
  const [ok, setOk] = useState(false);
  useEffect(() => { api('/settings').then(setS).catch(() => {}); }, []);

  const save = async () => {
    if (!s) return;
    await fetch(API + '/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(s) });
    setOk(true); setTimeout(() => setOk(false), 2000);
  };
  if (!s) return <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, color: '#9ca3af' }}>加载中…</div>;

  const inputStyle: React.CSSProperties = {
    width: '100%', background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 8,
    padding: '10px 12px', fontSize: 14, color: '#1f2937', outline: 'none',
    fontFamily: 'inherit', boxSizing: 'border-box',
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      <div style={{ maxWidth: 560, margin: '0 auto', padding: '32px 24px' }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, color: '#1f2937', marginBottom: 24 }}>设置</h1>

        <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb', padding: 20, marginBottom: 20 }}>
          <h3 style={{ fontSize: 12, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 16, marginTop: 0 }}>LLM 配置</h3>
          {[
            { label: 'API Key', key: 'llm_api_key', type: 'password' },
            { label: 'Base URL', key: 'llm_base_url', type: 'text' },
            { label: 'Model', key: 'llm_model', type: 'text' },
            { label: 'Top-K', key: 'top_k', type: 'number' },
          ].map(({ label, key, type }) => (
            <div key={key} style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 12, color: '#6b7280', marginBottom: 4, display: 'block' }}>{label}</label>
              <input
                type={type}
                value={String((s as any)[key] ?? '')}
                onChange={e => setS({ ...s, [key]: key === 'top_k' ? Number(e.target.value) : e.target.value })}
                style={inputStyle}
              />
            </div>
          ))}
        </div>

        <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e5e7eb', padding: 20, marginBottom: 20 }}>
          <h3 style={{ fontSize: 12, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 16, marginTop: 0 }}>检索设置</h3>
          {[
            ['rerank_enabled', 'Rerank 精排 (Cross-Encoder)'],
            ['stream_enabled', '流式输出 (SSE)'],
          ].map(([key, label]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 0', borderBottom: '1px solid #f3f4f6' }}>
              <span style={{ fontSize: 14, color: '#4b5563' }}>{label}</span>
              <button
                onClick={() => setS({ ...s, [key]: !(s as any)[key] })}
                style={{
                  width: 40, height: 24, borderRadius: 12, border: 'none',
                  background: (s as any)[key] ? '#111827' : '#d1d5db',
                  position: 'relative', cursor: 'pointer',
                }}
              >
                <span style={{
                  display: 'block', width: 20, height: 20, borderRadius: '50%',
                  background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
                  position: 'absolute', top: 2, left: 2,
                  transform: (s as any)[key] ? 'translateX(16px)' : 'none',
                  transition: 'transform 0.15s',
                }} />
              </button>
            </div>
          ))}
        </div>

        <button
          onClick={save}
          style={{
            width: '100%', padding: 14, background: '#111827', color: '#fff',
            borderRadius: 12, border: 'none', fontSize: 14, fontWeight: 500,
            cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          {ok ? '已保存 ✓' : '保存设置'}
        </button>
      </div>
    </div>
  );
}
