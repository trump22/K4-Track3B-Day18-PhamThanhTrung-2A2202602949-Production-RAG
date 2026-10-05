# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phạm Thanh Trung  
**MSSV:** 2A2202602949  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026

## Phần 1: Lecture Mapping

| Khái niệm | Module / hàm | Cách áp dụng và giới hạn |
|---|---|---|
| Semantic chunking | M1 — `chunk_semantic()` | Tách câu bằng regex, mã hóa MiniLM, so sánh cosine giữa hai câu liên tiếp; ngưỡng mặc định 0.85. Chưa kết luận cải thiện chất lượng chỉ từ ngưỡng; cần đánh giá trên corpus tiếng Việt. |
| Parent–child retrieval | M1 — `chunk_hierarchical()`; `pipeline.build_pipeline()` / `run_query()` | Index con tối đa 256 ký tự, sau rerank dùng cha tối đa 2048 ký tự làm context. ID cha trong pipeline gồm tên nguồn để không trùng giữa tài liệu. |
| Structure-aware chunking | M1 — `chunk_structure_aware()` | Tách theo `#`, `##`, `###`, giữ bảng/list và không coi heading trong fenced code block là section. |
| Hybrid Search | M2 — `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | underthesea chuẩn hóa `_` thành khoảng trắng; BM25 bắt từ khóa, bge-m3 tạo vector 1024 chiều; Qdrant `query_points()` tìm vector. RRF cộng `1/(60+rank+1)` và lấy top ứng viên. |
| Cross-Encoder | M3 — `CrossEncoderReranker.rerank()` | Chấm trực tiếp cặp query–chunk bằng `BAAI/bge-reranker-v2-m3`, chọn top 3. Benchmark trước tích hợp khoảng 615ms/3 ứng viên trên CPU, chưa đạt mục tiêu 150ms; không suy rộng thành latency cho 20 ứng viên. |
| RAGAS | M4 — `evaluate_ragas()` | Gemini chấm faithfulness, answer relevancy, context precision và context recall; embeddings local tránh phụ thuộc embedding API. Lưu điểm từng câu và phân biệt `status=error` với điểm đo bằng 0. |
| Diagnostic Tree | M4 — `failure_analysis()` | Tính trung bình bốn metric, chọn metric thấp nhất, tra nguyên nhân/gợi ý và lấy Bottom-N. Đây là dấu hiệu để kiểm tra câu trả lời/context, không tự chứng minh nguyên nhân. |
| Enrichment | M5 — `_enrich_single_call()`, `enrich_chunks()` | Một request/chunk lấy summary, HyQA, context, metadata; context và HyQA được đưa vào text index. Giữ nguyên nguồn/parent_id và fallback khi thiếu key, timeout hoặc JSON sai. |

## Phần 2: Challenges & Debugging

1. **NumPy/Python:** `OverflowError: cannot convert longdouble infinity to integer` khi import NumPy 1.26.4 trong Python 3.14. Cập nhật NumPy 2.5.3 giúp các model chạy, nhưng LangChain 0.2 vẫn khai báo NumPy <2. Không coi việc import thành công là đã giải quyết xong tương thích môi trường; nên chốt môi trường Python 3.11/3.12 với bộ dependency tương thích và lockfile ở vòng triển khai tiếp theo.
2. **Adapter embeddings của RAGAS 0.1:** `The truth value of an empty array is ambiguous. Use array.size > 0 to check that an array is not empty.` Nguyên nhân là adapter cũ dùng bool trên mảng NumPy rỗng. Thay bằng adapter LangChain bọc SentenceTransformer trực tiếp, giữ embeddings bge-m3 local và có test hồi quy.
3. **Asyncio trên Python 3.14:** RAGAS 0.1 tạo `asyncio.as_completed` trước event loop và phát sinh `RuntimeError` cùng cảnh báo coroutine chưa được await. Thử event loop đang chạy và nested loop, nhưng tích hợp thật còn lỗi `Timeout should be used inside a task`. Unit test giả lập đã không phát hiện lỗi này. Chuyển chạy tích hợp sang Python 3.11 trong `.venv311`, giữ môi trường cũ và bổ sung smoke test RAGAS thật trước chạy toàn bộ.
4. **Provider LLM:** Chuyển từ OpenAI sang Gemini không chỉ đổi tên biến key. Cần truyền key rõ ràng, đổi base URL sang Google, chọn GEMINI_MODEL, cấu hình cùng provider cho RAGAS và giữ embeddings local. Gemini Flash Lite báo `Multiple candidates is not enabled for this model` khi RAGAS gửi n=3. Adapter giữ ba mẫu Answer Relevancy bằng ba request n=1 riêng và có test hồi quy. Request giả lập kiểm tra endpoint/key/model; preflight Gemini thực tế đã trả lời OK trước chạy tích hợp. Model ban đầu `gemini-3.8-flash` có quota free tier 5 request/phút; chuyển cả Baseline và Production sang `gemini-3.1-flash-lite`, điều tiết request và cache enrichment thành công để giữ phép so sánh cùng model.
5. **Ghép parent–child và báo cáo:** Mã `parent_0` bị lặp nếu chunk từng tài liệu riêng, và pipeline ban đầu chỉ gửi child cho LLM. Bổ sung ID có namespace nguồn, map parent context và lưu `per_question` trong JSON để phân tích có bằng chứng.
6. **Bằng chứng đánh giá:** Unit test API giả lập xác minh logic, không chứng minh chất lượng LLM thật. So sánh Baseline/Production chỉ hợp lệ khi cả hai báo cáo có status OK, cùng 20 câu hỏi và có dữ liệu từng câu. Không dùng bảng điểm minh họa của đề bài làm số liệu thực nghiệm.
7. **Timeout và quota ngày khi tích hợp:** Production chấm được 60/80 metric rồi nhận `TimeoutError`. Kiểm tra API xác nhận HTTP 429 với `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, giới hạn 500 request/ngày của `gemini-3.1-flash-lite`. Phản hồi `retryDelay: 65205s` khiến hook chờ quá lâu nếu xử lý như quota phút. Sửa để quota ngày không tạo thời gian chờ trong hook; giữ timeout từng request 30 giây vì RAGAS tự ghi đè nó bằng timeout toàn metric. Tăng timeout toàn metric lên 600 giây. Thử chấm bằng `gemini-2.5-flash-lite` nhưng model này chỉ cho 20 request/ngày nên cũng dừng. Giữ nguyên các câu trả lời đã sinh; khi đổi model chấm phải chấm lại cả Baseline/Production cùng model, không so sánh điểm của hai model chấm khác nhau.

Môi trường tích hợp cuối cùng sử dụng Python 3.11.9 trong `.venv311`, NumPy 1.26.4 và RAGAS 0.1.22; `pip check` không báo dependency bị thiếu hoặc xung đột. Quota thực tế của Gemini Flash Lite trong lần chạy là 15 request/phút; client tăng khoảng cách request lên 4.2 giây khi nhận phản hồi 429. Cache enrichment chỉ lưu kết quả API hợp lệ, không lưu fallback làm kết quả AI.

Kiến thức cần bổ sung: tương thích dependency, quản lý quota/timeout và concurrency của LLM, cách RAGAS đánh giá phép tính/suy luận, retrieval có version/effective date, và đo p50/p95 latency trên cấu hình phần cứng cố định.

## Phần 3: Action Plan

### Dự án đề xuất: Trợ lý tra cứu chính sách nhân sự và CNTT

Đây là kế hoạch áp dụng dựa trên corpus của lab; chưa khẳng định có một dự án cá nhân đã triển khai ngoài bài tập.

**Hiện trạng:** Corpus có quy chế nhiều phiên bản, bảng lương và điều khoản loại trừ. Baseline dùng dense retrieval; pipeline nâng cao có hybrid, rerank và enrichment. Các rủi ro cần đo là chọn văn bản hết hiệu lực, thiếu bằng chứng cho câu hỏi nhiều bước, tính toán sai và latency CPU.

**Kế hoạch:**

1. Dùng structure-aware cho chính sách Markdown, kết hợp parent–child; giữ heading, bảng, phiên bản và ngày hiệu lực trong metadata.
2. Dùng BM25 + bge-m3 + RRF. Bổ sung kiểm soát tài liệu hiện hành với metadata version/effective_date/superseded; vẫn cho phép truy vấn lịch sử nếu người dùng hỏi rõ thời điểm.
3. Rerank top 20 bằng Cross-Encoder, chọn top 3 và lấy parent context. Đo latency thật trước khi quyết định GPU/ONNX hoặc giảm số ứng viên; không đặt ngưỡng chất lượng dựa vào benchmark chưa đo.
4. Đánh giá cùng bộ 20 câu bằng RAGAS và đối chiếu thủ công các câu version, negation, multi-hop, numeric. Mục tiêu thử nghiệm faithfulness/context precision >=0.75; chỉ xác nhận sau khi đo và so sánh baseline.
5. Enrichment một lần/chunk, cache theo hash nội dung và model; không ghi đè metadata nguồn. Chỉ cho LLM suy luận từ tài liệu đã cung cấp và lưu trạng thái fallback để kiểm toán.

**Timeline:**

- **Tuần 1:** Chuẩn hóa môi trường/lockfile, schema metadata và ngày hiệu lực; dựng Qdrant, thiết lập bộ câu hỏi và baseline có thể tái chạy.
- **Tuần 2:** Bổ sung version-aware retrieval, thử nhiều chunk/top-k, cache enrichment, đo chất lượng/latency/chi phí; hoàn thiện Bottom-5 và thêm test hồi quy từ lỗi thật.

**Tiêu chí nghiệm thu:** báo cáo đủ 20 câu và 4 metric, lưu answer/context/ground truth; kết quả không suy giảm trên nhóm câu hỏi phủ định/phiên bản; có log latency và lỗi dịch vụ; thông tin chưa tìm được phải được nêu rõ thay vì suy đoán.

### Kết quả chạy tích hợp ngày 05/10/2026

Chạy `main.py` thành công trên Python 3.11.9 với cùng 20 câu và model Gemini `gemini-3.1-flash-lite` cho Baseline và Production. Cả hai report có đủ 20 điểm hợp lệ cho từng metric. Baseline đạt Faithfulness 0.9000, Answer Relevancy 0.7864, Context Precision 0.7500, Context Recall 0.8250; Production đạt lần lượt 0.8667, 0.8346, 0.7750 và 0.9250. Production tăng ba metric (đặc biệt Context Recall +0.1000) nhưng Faithfulness giảm 0.0333. Bottom-5 cũng cho thấy có lỗi tính toán 2%/tháng cần xử lý ở bước sinh đáp án, cùng một số điểm RAGAS thấp chưa khớp với context/answer khi kiểm tra thủ công; vì vậy cần xác thực evaluator trước khi tối ưu retrieval chỉ theo điểm tổng hợp.
