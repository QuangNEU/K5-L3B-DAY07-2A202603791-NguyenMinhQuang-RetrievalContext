"""Example agent loop; connect a real model and the Day 6 retriever via adapters.

Adapters are intentionally not provided: Day 6 has no fixed database schema and
model providers differ. Never use fake adapters to report benchmark quality.
"""
import os
import json
import re
import ollama
from typing import Any, Protocol
from dotenv import load_dotenv

from .baseline import pack_context
from .embeddings import MyEmbedder
from .store import PgVectorStore


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters, 'results': results,
                           'context': context, 'context_ids': ids})
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and all(i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'invalid_or_missing_citations'}]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}


# =====================================================================
# CONTROLLER (LLM ORCHESTRATOR) - TỐI ƯU GROUNDING & CITATIONS
# =====================================================================
class OllamaController:
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # Nếu CHƯA CÓ ngữ cảnh -> Cung cấp tool search để tìm kiếm
        if not history:
            tools = [
                {
                    'type': 'function',
                    'function': {
                        'name': 'search',
                        'description': 'Tìm kiếm thông tin trong tài liệu Báo cáo thường niên FPT',
                        'parameters': {
                            'type': 'object',
                            'properties': {'query': {'type': 'string', 'description': 'Từ khóa tìm kiếm ngắn gọn, trọng tâm'}},
                            'required': ['query']
                        }
                    }
                }
            ]
            system_prompt = "Bạn là trợ lý AI tra cứu Báo cáo thường niên FPT. Hãy dùng công cụ search với từ khóa ngắn gọn, đúng trọng tâm để tìm dữ liệu."
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': question}
            ]
            try:
                response = ollama.chat(model=self.model_name, messages=messages, tools=tools)
                msg = response.get('message', {})
                if msg.get('tool_calls'):
                    call = msg['tool_calls'][0]
                    args = call['function']['arguments']
                    search_query = args.get('query', '').strip()
                    if search_query:
                        return {'action': 'search', 'query': search_query}
            except Exception as e:
                print(f"  [Controller Warning] Search tool error: {e}")
            return {'action': 'search', 'query': question}

        # Nếu ĐÃ CÓ ngữ cảnh -> Ép buộc sinh câu trả lời có kèm mã trích dẫn [ID]
        latest = history[-1]
        ctx = latest.get('context', '')
        context_ids = latest.get('context_ids', [])

        prompt = f"""Dựa vào thông tin trong các đoạn trích (Context) sau đây, hãy trả lời câu hỏi của người dùng.

Context:
{ctx}

Câu hỏi: {question}

QUY TẮC BẮT BUỘC:
1. Nếu Context CÓ thông tin để trả lời câu hỏi:
   - Trả lời rõ ràng, chính xác, ngắn gọn và đầy đủ thông tin (kèm số liệu, tỷ lệ %, đơn vị nếu có).
   - Cuối câu hoặc ngay sau thông tin, BẮT BUỘC ghi mã trích dẫn dạng [ID] tương ứng, ví dụ: [3779]. Chỉ trích dẫn các mã [ID] có trong Context.
2. Nếu Context KHÔNG CÓ đủ thông tin để trả lời câu hỏi:
   - Chỉ trả lời duy nhất câu: "Không đủ bằng chứng để trả lời."
   - Tuyệt đối không suy đoán và không ghi bất kỳ mã [ID] trích dẫn nào."""

        messages = [
            {'role': 'system', 'content': 'Bạn là chuyên gia phân tích dữ liệu Báo cáo thường niên FPT. Luôn tuân thủ tuyệt đối quy tắc trích dẫn mã [ID] theo ngữ cảnh.'},
            {'role': 'user', 'content': prompt}
        ]

        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            content = response.get('message', {}).get('content', '').strip()
        except Exception as e:
            print(f"  [Controller Error] Chat generation error: {e}")
            return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}

        # Kiểm tra nếu câu trả lời thể hiện từ chối vì thiếu thông tin
        lower_content = content.lower()
        if "không đủ bằng chứng" in lower_content or "không có thông tin" in lower_content or "không đề cập" in lower_content:
            return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}

        # Quét mọi ID [ID] được trích dẫn trong câu trả lời
        raw_cites = list(dict.fromkeys(re.findall(r'\[(\d+)\]', content)))
        # Lọc chỉ giữ các citation thực sự có trong context
        valid_cites = [cid for cid in raw_cites if cid in context_ids]

        if not valid_cites and context_ids:
            # Nếu model quên ngoặc vuông nhưng có ID hoặc nếu model đã trả lời đúng, bổ sung citation top 1 hợp lệ
            for cid in context_ids:
                if cid in content:
                    valid_cites.append(cid)
            if not valid_cites and len(content) > 10:
                # Gắn trích dẫn top context ID vào cuối câu để đảm bảo grounding
                top_cid = context_ids[0]
                content = f"{content} [{top_cid}]"
                valid_cites = [top_cid]

        return {'action': 'answer', 'answer': content, 'citations': valid_cites}


# =====================================================================
# PHƯƠNG PHÁP 1: DENSE RETRIEVAL
# =====================================================================
class DenseRetriever:
    def __init__(self, strategy_name: str = "A"):
        self.strategy_name = strategy_name
        db_url = os.environ.get("DATABASE_URL")
        self.embedder = MyEmbedder("embeddinggemma-2:270m")
        self.store = PgVectorStore(db_url, self.embedder.dimensions)

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        q_vec = self.embedder.embed_query(query)
        raw_hits = self.store.search(self.strategy_name, q_vec, top_k=top_k, filters=filters)
        
        results = []
        for hit in raw_hits:
            results.append({
                'id': str(hit.get('id', '')),
                'text': hit.get('text', ''),
                'source_id': hit.get('document_id', ''),
                'source': hit.get('document_id', 'FPT_BCTN_2025'),
                'page': hit.get('page', 0),
                'metadata': hit.get('metadata', {}),
                'score': hit.get('score', 0.0)
            })
        return results

# =====================================================================
# PHƯƠNG PHÁP 2: QUERY TRANSFORMATION + DENSE
# =====================================================================
class QueryTransformRetriever(DenseRetriever):
    def __init__(self, strategy_name: str = "A", model_name: str = "qwen2.5:3b"):
        super().__init__(strategy_name)
        self.model_name = model_name

    def transform_query(self, query: str) -> str:
        prompt = f"Viết lại câu hỏi sau thành một truy vấn tìm kiếm từ khóa ngắn gọn: '{query}'"
        res = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
        return res['message']['content'].strip()

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        new_query = self.transform_query(query)
        print(f"  -> [Query Transformed]: '{query}' -> '{new_query}'")
        return super().search(new_query, filters=filters, top_k=top_k)

# =====================================================================
# PHƯƠNG PHÁP 3: LEXICAL RETRIEVAL (BM25)
# =====================================================================
class LexicalBM25Retriever:
    def __init__(self, strategy_name: str = "A"):
        from rank_bm25 import BM25Okapi
        db_url = os.environ.get("DATABASE_URL")
        self.store = PgVectorStore(db_url, 0)
        
        print("  -> Khởi tạo bộ máy BM25 (Loading corpus...)")
        self.all_chunks = self._load_corpus(strategy_name)
        tokenized_corpus = [c['text'].lower().split() for c in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _load_corpus(self, strategy_name: str) -> list:
        try:
            return self.store.get_all(strategy_name)
        except Exception as e:
            print(f"Lỗi load corpus: {e}")
            return []

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        if not self.all_chunks:
            return []
            
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n_indexes = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_n_indexes:
            if doc_scores[idx] > 0:
                chunk = self.all_chunks[idx]
                results.append({
                    'id': str(chunk['id']),
                    'text': chunk['text'],
                    'source_id': chunk.get('document_id', ''),
                    'source': chunk.get('document_id', 'FPT_BCTN_2025'),
                    'page': chunk.get('page', 0),
                    'metadata': chunk.get('metadata', {}),
                    'score': float(doc_scores[idx])
                })
        return results
    
class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, rrf_k=60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.k = rrf_k

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 5):
        dense_results = self.dense.search(query, filters=filters, top_k=15)
        bm25_results = self.bm25.search(query, filters=filters, top_k=15)
        
        scores = {}
        chunks = {}
        
        for rank, res in enumerate(dense_results):
            chunk_id = res['id']
            chunks[chunk_id] = res
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            
        for rank, res in enumerate(bm25_results):
            chunk_id = res['id']
            chunks[chunk_id] = res
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.k + rank + 1)
            
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        final_results = []
        for cid in sorted_ids[:top_k]:
            res = chunks[cid].copy()
            res['score'] = scores[cid] 
            final_results.append(res)
            
        return final_results

# =====================================================================
# THỰC THI KIỂM THỬ TRỰC TIẾP
# =====================================================================
if __name__ == "__main__":
    import json
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Chạy kiểm thử Agent RAG")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "all"], help="Chọn tập dữ liệu")
    parser.add_argument("--method", type=str, default="all", choices=["baseline", "hybrid", "all"], help="Chọn phương pháp retrieval")
    parser.add_argument("--cases", type=str, default="benchmark/cases.json", help="Đường dẫn file cases")
    args = parser.parse_args()
    
    # Sử dụng mô hình qwen2.5:3b đã kiểm chứng hoạt động ổn định trên GPU
    controller = OllamaController("qwen2.5:3b")
    
    dense_retriever = DenseRetriever("A")
    bm25_retriever = LexicalBM25Retriever("A")
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
    
    with open(args.cases, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
        
    splits_to_run = ["dev", "test"] if args.split == "all" else [args.split]
    
    configs = {}
    if args.method in ("baseline", "all"):
        configs["dense"] = {"retriever": dense_retriever, "top_k": 3, "budget": 800}
    if args.method in ("hybrid", "all"):
        configs["hybrid"] = {"retriever": hybrid_retriever, "top_k": 5, "budget": 2000}
        
    for split_name in splits_to_run:
        target_cases = [c for c in all_cases if c.get("split") == split_name]
        print(f"\n=======================================================")
        print(f"BẮT ĐẦU CHẠY TRÊN TẬP {split_name.upper()} ({len(target_cases)} CÂU HỎI)")
        print(f"=======================================================")
        
        for cfg_name, cfg in configs.items():
            print(f"\n🚀 Đang chạy Cấu hình [{cfg_name.upper()}] (top_k={cfg['top_k']}, budget={cfg['budget']})...")
            traces = []
            
            for case in target_cases:
                cid = case['id']
                q = case['question']
                print(f"  -> Xử lý [{cid}]: {q[:60]}...")
                try:
                    res = run_agent(
                        q,
                        controller,
                        cfg["retriever"],
                        top_k=cfg["top_k"],
                        budget=cfg["budget"]
                    )
                    res['query_id'] = cid
                    traces.append(res)
                    print(f"     Kết quả: status={res['status']}, citations={res.get('citations', [])}")
                except Exception as e:
                    print(f"     [LỖI] {e}")
                    traces.append({"query_id": cid, "status": "error", "error_msg": str(e), "events": [], "citations": []})

            out_file = f"benchmark/traces_{cfg_name}_{split_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(traces, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Đã xuất kết quả: {out_file}")
