# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phạm Thanh Trung  
**MSSV:** 2A2202602949  
**Khóa:** K4 - Track 3B  
**Ngày chạy:** 05/10/2026

## RAGAS Scores

Lần chạy `main.py` dùng Python 3.11.9, Gemini `gemini-3.1-flash-lite` và cùng 20 câu hỏi cho cả hai hệ thống. Cả hai báo cáo có `status: ok`, đủ 20 điểm hợp lệ cho mỗi metric.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|--:|
| Faithfulness | 0.9000 | 0.8667 | -0.0333 |
| Answer Relevancy | 0.7864 | 0.8346 | +0.0483 |
| Context Precision | 0.7500 | 0.7750 | +0.0250 |
| Context Recall | 0.8250 | 0.9250 | +0.1000 |

Production cải thiện ba metric, đặc biệt Context Recall, nhưng Faithfulness giảm nhẹ. Vì vậy chưa thể kết luận pipeline mới tốt hơn trên mọi khía cạnh; cần ưu tiên kiểm tra lỗi tính toán và xác nhận các điểm RAGAS thấp có khớp với bằng chứng được truy hồi hay không.

## Bottom-5 Failures

Danh sách dưới đây theo thứ tự `reports/ragas_report.json`. “Worst metric” là metric thấp nhất mà RAGAS ghi nhận cho từng câu. Với các trường hợp có câu trả lời/context đúng khi kiểm tra thủ công, phân tích phân biệt rõ điểm thấp của metric với lỗi thực tế, không mặc định chẩn đoán tự động là nguyên nhân gốc.

### #1 — Câu hỏi nhiều bước: ngày phép và khoảng lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép theo chính sách 2024 và lương Senior (P3-P4) 20–35 triệu VNĐ/tháng.
- **Got:** Trả lời đúng 18 ngày phép, nhưng nói không tìm thấy khoảng lương.
- **Worst metric:** Answer Relevancy = 0.2917 (Faithfulness 0.6667, Context Precision 0, Context Recall 0.5).
- **Câu trả lời có đúng không?** Đúng một phần; phép tính ngày phép đúng, câu trả lời thiếu dữ liệu lương mà câu hỏi yêu cầu.
- **Các đoạn trích dẫn có đáp án không?** Có phần ngày phép, gồm cả quy chế 2024 và 2023; không có đoạn về dải lương Senior, nên mô hình không thể trả lời đủ từ context đã cấp.
- **Câu hỏi có cần viết lại không?** Câu hỏi rõ nhưng gộp hai thông tin ở hai tài liệu. Có thể giữ nguyên cho bài test multi-hop; hệ thống nên tách thành các truy vấn con thay vì bắt người dùng phải viết lại.
- **Cây lỗi:** Output thiếu một nửa → Context đầy đủ? Không, thiếu đoạn chính sách lương → Query có rõ không? Có, nhưng gồm hai chủ đề → Root cause: retrieval không lấy được tài liệu lương cùng tài liệu phép năm.
- **Module cần sửa:** Ưu tiên M2 (decompose/expand truy vấn và hợp nhất kết quả từ cả hai chủ đề), sau đó kiểm tra M3 để tránh loại mất chunk lương khi rerank. Bổ sung test multi-hop để xác nhận context cuối có cả hai nguồn.

### #2 — Tính phí tạm ứng quá hạn

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng trên 15 triệu là 300.000 VNĐ/tháng, tính pro-rata khoảng 50.000 VNĐ cho 5 ngày.
- **Got:** Tính 15.000.000 × 2% = 300.000 VNĐ và kết luận đó là khoản phạt phải trả.
- **Worst metric:** Faithfulness = 0.5914 (Faithfulness điểm từng câu = 0, Context Recall = 0.5).
- **Câu trả lời có đúng không?** Không. Mô hình coi mức 2% mỗi tháng là mức phạt cố định một lần, đồng thời bỏ qua 5 ngày quá hạn thay vì một tháng.
- **Các đoạn trích dẫn có đáp án không?** Có. Context nêu hạn thanh toán 15 ngày và phí 2%/tháng trên số tiền chưa hoàn ứng. Phần pro-rata 5 ngày không được diễn đạt thành phép tính mẫu trong đoạn trích, nhưng có thể suy ra từ quy tắc theo tháng.
- **Câu hỏi có cần viết lại không?** Câu hỏi đủ rõ; để giảm mơ hồ về cơ sở thời gian, có thể hỏi “tính phí pro-rata cho 5 ngày quá hạn theo mức 2%/tháng là bao nhiêu?”
- **Cây lỗi:** Output sai → Context có quy tắc chính nhưng không có ví dụ pro-rata → Query cơ bản rõ → Root cause: bước sinh câu trả lời/tính toán không diễn giải “/tháng” và thời gian quá hạn chính xác.
- **Module cần sửa:** Đây là lỗi ở bước sinh đáp án sau M3, không phải bằng chứng lỗi M1–M3/M5. Làm rõ prompt yêu cầu nêu công thức và kỳ tính phí; với phép tính tiền, dùng hàm tính toán xác định hoặc kiểm tra số học, rồi thêm ca kiểm thử ở M4.

### #3 — Ngưỡng thâm niên cộng phép

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách hiện hành v2024, từ 3 năm trở lên; mỗi 3 năm cộng 1 ngày. Quy chế cũ v2023 dùng ngưỡng 5 năm.
- **Got:** Nêu đúng ngưỡng 3 năm theo v2024 và ghi chú ngưỡng cũ là 5 năm.
- **Worst metric:** Context Precision = 0.7037 (Faithfulness = 1, Context Recall = 1).
- **Câu trả lời có đúng không?** Có; nội dung trả lời phân biệt đúng phiên bản mới và cũ.
- **Các đoạn trích dẫn có đáp án không?** Có. Context có cả quy chế 2024 (3 năm) và quy chế 2023 (5 năm), đều liên quan trực tiếp đến xung đột phiên bản.
- **Câu hỏi có cần viết lại không?** Không bắt buộc; có thể thêm “theo quy chế hiện hành” nếu chỉ muốn câu trả lời hiện hành, nhưng bộ test có chủ ý kiểm tra phân biệt phiên bản.
- **Cây lỗi:** Output đúng → Context có cả hai phiên bản và ngày hiệu lực → Query hơi thiếu mốc thời gian nhưng có thể xử lý theo chính sách hiện hành → Điểm Precision thấp không phù hợp rõ ràng với hai context liên quan khi kiểm tra thủ công.
- **Module cần sửa:** Trước khi thay đổi M2/M3, kiểm tra cách chấm Context Precision trong M4 và đối chiếu tiêu chí “relevant” của evaluator với câu hỏi có xung đột phiên bản. Không loại tài liệu cũ một cách tuyệt đối vì nó cần thiết khi giải thích khác biệt phiên bản.

### #4 — Phân loại thông tin lương

- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Dữ liệu Bí mật (cấp 3); cần hạn chế quyền truy cập và mã hóa khi truyền.
- **Got:** “Thông tin lương thuộc cấp độ Bí mật.”
- **Worst metric:** Context Precision = 0.7049 (Faithfulness = 1, Context Recall = 1).
- **Câu trả lời có đúng không?** Có. Câu trả lời trực tiếp đáp ứng điều được hỏi; phần ground truth còn nêu thêm quy tắc xử lý dữ liệu.
- **Các đoạn trích dẫn có đáp án không?** Có. Quy chế chi trả lương nói rõ thông tin lương là dữ liệu Bí mật; chính sách phân loại nêu quy tắc cho dữ liệu Bí mật/cấp 3.
- **Câu hỏi có cần viết lại không?** Không, câu hỏi đơn nghĩa. Nếu muốn chấm đủ các ý trong ground truth, cần hỏi thêm “và cần xử lý như thế nào?”
- **Cây lỗi:** Output đúng với câu hỏi → Context chứa phân loại cụ thể và quy tắc áp dụng → Query rõ → Context Precision thấp nhưng các context được trả về đều có bằng chứng liên quan.
- **Module cần sửa:** Kiểm tra M4 (Context Precision judge/định nghĩa relevance và mức độ khớp với câu hỏi), không mặc định cần chỉnh M2/M3. Có thể mở rộng câu hỏi/ground truth đồng bộ nếu kỳ vọng câu trả lời có thêm quy tắc mã hóa và quyền truy cập.

### #5 — Hoàn trả chi phí đào tạo

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc tối thiểu 1 năm; nghỉ sau 8 tháng nên phải hoàn trả 100%, tức 25.000.000 VNĐ.
- **Got:** Trả lời hoàn trả 25.000.000 VNĐ (100%).
- **Worst metric:** Faithfulness = 0.7255 (Faithfulness điểm từng câu = 0, Context Precision = 1, Context Recall = 1).
- **Câu trả lời có đúng không?** Có. Số tiền và tỷ lệ hoàn trả đều đúng theo chính sách.
- **Các đoạn trích dẫn có đáp án không?** Có đầy đủ: cam kết ít nhất 1 năm sau khóa học và hoàn trả 100% nếu nghỉ trước hạn.
- **Câu hỏi có cần viết lại không?** Không; đủ dữ kiện về số tiền tài trợ và thời gian nghỉ việc.
- **Cây lỗi:** Output đúng → Context chứa trực tiếp điều kiện 1 năm và hoàn trả 100% → Query rõ → Điểm Faithfulness thấp mâu thuẫn với câu trả lời và bằng chứng đang lưu.
- **Module cần sửa:** Kiểm tra M4/evaluator và dữ liệu đầu vào của lần chấm (đặc biệt prompt/judge output cho câu này); không sửa retrieval chỉ dựa trên chẩn đoán “hallucination” tự động khi context precision/recall đều bằng 1 và câu trả lời được context hỗ trợ.

## Case Study (cho presentation)

**Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

**Error Tree walkthrough:**
1. Output đúng? → Không. Mô hình tính 300.000 VNĐ như khoản phạt một lần; mức 2% trong context là theo tháng, và chỉ có 5 ngày quá hạn.
2. Context đúng? → Có quy tắc thời hạn 15 ngày và phí 2%/tháng trên số tiền chưa hoàn ứng; thiếu ví dụ tính pro-rata.
3. Query rewrite OK? → Câu hỏi gốc hiểu được; có thể làm rõ “phí pro-rata cho 5 ngày” nhưng không nên coi việc viết lại câu hỏi là cách chữa chính.
4. Fix ở bước: → Bước sinh đáp án/tính toán sau reranking: yêu cầu mô hình nhận diện đơn vị thời gian, trình bày phép tính; nên chuyển tính toán sang logic xác định và kiểm thử bằng số.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm kiểm thử số học cho các câu hỏi tiền tệ có tỷ lệ theo tháng/ngày; so sánh kết quả LLM với phép tính xác định.
- Kiểm tra thủ công ba trường hợp có context/answer đúng nhưng RAGAS chấm thấp để xác thực M4 trước khi thay đổi retrieval.
- Bổ sung truy vấn multi-hop kiểm tra khả năng lấy đồng thời dữ liệu từ chính sách phép năm và bảng lương.
