# MMDS Final Project

Đây là README cho bài nộp cuối kỳ môn Mining Massive Data Sets. Hai mục cần đọc/chấm chính là:

1. `final_2_523K0011/` - thư mục nộp bài chính, gồm source code, notebook đã chạy, dữ liệu, kết quả, hình ảnh và báo cáo.
2. `project_documentation.docx` - tài liệu mô tả quá trình làm bài, ý tưởng triển khai, quyết định kỹ thuật và bằng chứng kiểm tra.

## Thông tin nhóm

- Chau Pham Tuan Kiet - 523K0011
- Nguyen Bao Long - 523K0014
- Nguyen Ba Hung - 523K0006
- Giảng viên: Hoang Anh

## Tóm tắt project

Project giải quyết 3 bài toán trong đề final:

| Task | Nội dung | Kết quả chính |
| --- | --- | --- |
| Task 01 | Hierarchical clustering trong không gian phi Euclidean với chuỗi, 4-shingle và Jaccard distance | 10,000 chuỗi, 116 bước merge, dừng tại 9,884 clusters |
| Task 02 | Dự đoán giá vàng bằng PySpark Linear Regression sau khi giảm chiều CUR từ 15 về 5 | Best setting `k = 6`, test RMSE `0.3169` |
| Task 03 | Collaborative filtering với user-based similarity và lookup nhanh ứng viên tương tự | Best `N = 10`, RMSE `0.9532`, indexed lookup `2.832s` so với baseline `3.815s` |

## Cấu trúc thư mục nộp bài

```text
final_2_523K0011/
|-- Report/
|   |-- report.pdf
|   |-- report.tex
|   |-- references.bib
|   |-- task01_global_avg_dist.png
|   |-- task01_tsne_3d.png
|   |-- task02_loss_curves.png
|   |-- task02_twin_bars.png
|   |-- task03_rmse_by_n.png
|   `-- task03_runtime_compare.png
`-- Source/
    |-- Task01/
    |   |-- Task01.ipynb
    |   |-- task01_strings.csv
    |   |-- task01_metrics.csv
    |   |-- task01_global_avg_dist.png
    |   `-- task01_tsne_3d.png
    |-- Task02/
    |   |-- Task02.ipynb
    |   |-- gold_prices.csv
    |   |-- task02_results.csv
    |   |-- task02_loss_curves.png
    |   `-- task02_twin_bars.png
    `-- Task03/
        |-- Task03.ipynb
        |-- ratings2k.csv
        |-- task03_results.csv
        |-- task03_rmse_by_n.png
        `-- task03_runtime_compare.png
```

## File nên đọc theo thứ tự

1. `project_documentation.docx`: đọc trước để hiểu cách nhóm phân tích rubric, thiết kế pipeline, chọn thuật toán và kiểm tra kết quả.
2. `final_2_523K0011/Report/report.pdf`: báo cáo IEEE ngắn gọn, chứa kết quả cuối cùng và hình minh họa.
3. `final_2_523K0011/Source/Task01/Task01.ipynb`: notebook clustering chuỗi bằng 4-shingle và Jaccard distance.
4. `final_2_523K0011/Source/Task02/Task02.ipynb`: notebook dự đoán giá vàng bằng CUR reduction và Linear Regression.
5. `final_2_523K0011/Source/Task03/Task03.ipynb`: notebook collaborative filtering và so sánh runtime lookup.

Các thư mục như `docs/`, `lectures/`, `tmp/`, `output/` chỉ là tài liệu hoặc file làm việc phụ, không phải phần cần đọc chính khi chấm bài nộp cuối.

## Môi trường chạy

Các notebook có phần dependency setup ở đầu file. Môi trường cần:

- Python 3.x
- Jupyter Notebook hoặc Google Colab
- Java runtime để chạy PySpark
- Các thư viện Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyspark`

Nếu chạy trên Google Colab, mở từng notebook và chạy từ trên xuống. Nếu file dữ liệu chưa có trong runtime, notebook Task 02 và Task 03 có logic yêu cầu upload `gold_prices.csv` hoặc `ratings2k.csv`.

## Cách chạy lại kết quả

Chạy từng notebook theo thứ tự sau:

1. Mở `final_2_523K0011/Source/Task01/Task01.ipynb`, chọn `Run All`. Notebook sẽ sinh lại `task01_strings.csv`, `task01_metrics.csv` và hai hình của Task 01.
2. Mở `final_2_523K0011/Source/Task02/Task02.ipynb`, chọn `Run All`. Notebook dùng `gold_prices.csv`, tạo lag features, chạy CUR reduction và xuất `task02_results.csv` cùng hai hình kết quả.
3. Mở `final_2_523K0011/Source/Task03/Task03.ipynb`, chọn `Run All`. Notebook dùng `ratings2k.csv`, chạy collaborative filtering, xuất `task03_results.csv` và hai hình đánh giá.

Sau khi chạy lại, các hình trong từng thư mục `Source/TaskXX/` có thể được copy sang `Report/` để rebuild `report.pdf` từ `report.tex` nếu cần.

## Ghi chú kỹ thuật

- Task 01 dùng PySpark cho bước tạo dữ liệu và shingling, sau đó chạy agglomerative clustering trong bộ nhớ theo yêu cầu đề bài.
- Task 02 dùng 15 giá bán trước đó làm feature, giá bán tại ngày hiện tại làm label, split train/test theo tỉ lệ 7:3 và đánh giá các chiều giảm từ 15 đến 5.
- Task 03 biểu diễn user profile bằng sparse dictionary; giá trị 0 được hiểu là chưa rating, không phải đánh giá thấp. Similarity chỉ tính trên item cùng được rating.
- Các notebook đã lưu output, CSV và hình ảnh để thuận tiện kiểm tra mà không cần chạy lại toàn bộ.

## Bằng chứng đầu ra

- Task 01: `task01_strings.csv`, `task01_metrics.csv`, `task01_global_avg_dist.png`, `task01_tsne_3d.png`
- Task 02: `task02_results.csv`, `task02_loss_curves.png`, `task02_twin_bars.png`
- Task 03: `task03_results.csv`, `task03_rmse_by_n.png`, `task03_runtime_compare.png`
- Báo cáo cuối: `final_2_523K0011/Report/report.pdf`
- Tài liệu quá trình: `project_documentation.docx`
