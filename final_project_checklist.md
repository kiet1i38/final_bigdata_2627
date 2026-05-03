# Final Project Submission Checklist - MMDS Endterm

**Check date:** 2025-05-01  
**Folder:** `final_2_523K0011/`  
**File đề bài:** `2526-HK2-MMDS-CK.pdf`  
**Status:** Chưa sửa, chỉ kiểm tra

---

## I. CẤU TRÚC NỘP BÀI (Theo đề bài - Page 5)

### Yêu cầu:
- Tên folder: `final_<Project group ID>_<student ID>`
- `Source/`: chứa source code, từng task trong sub-directory riêng, preserve outputs của tất cả cells trong ipynb
- `Report/`: report source (từ Overleaf), file report.pdf

### Thực tế:
```
final_2_523K0011/
├── Report/
│   ├── references.bib
│   ├── report.pdf (565KB)
│   ├── report.tex
│   ├── task01_global_avg_dist.png
│   ├── task01_tsne_3d.png
│   ├── task02_twin_bars.png
│   ├── task03_rmse_by_n.png
│   └── task03_runtime_compare.png
└── Source/
    ├── Task01/
    │   ├── Task01.ipynb (290KB)
    │   ├── task01_global_avg_dist.png
    │   ├── task01_metrics.csv
    │   ├── task01_strings.csv (2.7MB - 10,000 strings)
    │   └── task01_tsne_3d.png
    ├── Task02/
    │   ├── Task02.ipynb (117KB)
    │   ├── gold_prices.csv
    │   ├── task02_loss_curves.png
    │   ├── task02_results.csv
    │   └── task02_twin_bars.png
    └── Task03/
        ├── Task03.ipynb (79KB)
        ├── ratings2k.csv
        ├── task03_results.csv
        ├── task03_rmse_by_n.png
        └── task03_runtime_compare.png
```

**✓ Đạt:** Cấu trúc đúng quy định

---

## II. TASK 1: HIERARCHICAL CLUSTERING (3.0 điểm)

### A. Data (1.0 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Tạo 10,000 chuỗi alphabetical, độ dài [32,64] | 0.25 | Random length [32,64], không phân biệt hoa thường | Có file `task01_strings.csv` (10,000 strings) | ✓ Có (cần check trong notebook) |
| 4-shingle tokenization | 0.25 | Áp dụng 4-shingle cho mỗi string | Báo cáo xác nhận | ✓ Có |
| Jaccard distance | 0.25 | Dùng Jaccard để đo dissimilarity | Báo cáo xác nhận | ✓ Có |
| Lưu csv: index, string, shingles | 0.25 | 3 cột: index, string, shingles | Có `task01_strings.csv` | ✓ Có |

### B. Algorithms (1.5 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Agglomerative (bottom-up) | 0.5 | Algorithm structure: agglomerative, data space: non-Euclidean, in-memory | Báo cáo: "final agglomerative loop runs in memory" | ✓ Có |
| Threshold t để merge | 0.5 | Merge if distance ≤ t | **⚠️ Chưa rõ** - Báo cáo chỉ nói "merge threshold is chosen with data-driven threshold" | ⚠️ Cần check notebook |
| Approach 2 (non-Euclidean) | 0.5 | Cluster representation & distance theo Approach 2 | Báo cáo: "Follow Approach 2 in the lecture" | ✓ Có |
| Terminate khi diameter jump | 0.25 | Phát hiện abnormal jump trong cluster diameter | Báo cáo: "abnormal jump in cluster diameter" | ✓ Có |
| OOP model, code compact | 0.25 | Tổ chức theo OOP, code gọn | Báo cáo: "compact OOP design" | ✓ Có |

**⚠️ LƯU Ý QUAN TRỌNG:** Đề bài yêu cầu CẢ:
1. "Propose a distance threshold t such that clusters can be merged if and only if their distance does not exceed t" (trong phần Algorithm structure)
2. "The algorithm terminates when a cluster diameter suddenly jumps" (0.25 điểm riêng)

Báo cáo chỉ nhắc đến diameter jump. Cần kiểm tra `Task01.ipynb` xem có implement cả 2 tiêu chí này không hay chỉ có 1. Nếu thiếu cái đầu (threshold t) thì **mất 0.5 điểm**.

### C. Experiments (0.5 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Line chart: global_avg_dist qua các bước | 0.25 | Tính avg_dist từng cluster, global_avg_dist, vẽ line chart | Có `task01_global_avg_dist.png` | ✓ Có |
| 3D t-SNE visualization | 0.25 | Dùng t-SNE để visualize kết quả clustering trong 3D | Có `task01_tsne_3d.png` | ✓ Có |

---

## III. TASK 2: LINEAR REGRESSION - GOLD PRICE (3.0 điểm)

### A. Data (0.5 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Đọc gold_prices.csv với PySpark | 0.25 | Date, Buy Price, Sell Price; PySpark | Có `gold_prices.csv`, báo cáo xác nhận | ✓ Có |
| Tạo samples: 15 previous dates làm features, date t làm label | 0.25 | Features: 15 consecutive previous dates; Label: date t | Báo cáo: "previous 15 gold prices as features" | ✓ Có |
| Split 7:3 (train:test) | (trong 0.25 trên) | Random split 7:3 | Báo cáo: "7:3 as required", 3960 train/1590 test | ✓ Có |

### B. Algorithms (1.5 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Dùng PySpark LinearRegression có sẵn | 0.5 | pyspark.ml.regression.LinearRegression | Báo cáo xác nhận | ✓ Có |
| CUR class giảm chiều 15→5, step=1 | 1.0 | Implement CUR algorithm dạng class, giảm từ 15 xuống 5, bước nhảy 1 | Báo cáo: "CUR-style class", "from 15 down to 5" | ✓ Có |
| Dùng RowMatrix | (trong 1.0) | pyspark.mllib.linalg.distributed.RowMatrix | Báo cáo: "RowMatrix.computeSVD" | ✓ Có |

### C. Experiments (1.0 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Infer row embedding cho train và test | 0.25 | New representation cho mỗi feature vector | Báo cáo: "infer row embeddings" | ✓ Có |
| Line chart: training losses (nhiều curves) | 0.25 | 1 chart duy nhất với nhiều loss curves | Có `task02_loss_curves.png` | ✓ Có |
| Twin-bar chart: train vs test results | 0.25 | 1 chart duy nhất với nhiều twin-bars | Có `task02_twin_bars.png` | ✓ Có |
| Experiments dạng classes với config parameters | 0.25 | Không replicate code, dùng class + config | **⚠️ Chưa rõ** - Báo cáo: "configuration-driven experiment pipeline" | ⚠️ Cần check notebook |

---

## IV. TASK 3: COLLABORATIVE FILTERING (3.0 điểm)

### A. Data (0.5 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| User vector: ratings của items | 0.25 | u[it] = rating của user u cho item it | Có `ratings2k.csv` (2365 ratings, 74 users, 467 items) | ✓ Có |
| Distance metric: 0 không phải negative | 0.25 | Đảm bảo 0s không có nghĩa là negative | Báo cáo: "missing entries never act as negative evidence" | ✓ Có |

### B. Algorithms (1.5 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Quick determine similar users (O(1) avg) | 0.5 | Không enumerate whole dataset, O(1) average | Báo cáo: "hash-based overlap index", "constant-time dictionary lookup" | ✓ Có |
| Implement proposed approach | 0.5 | Tích hợp vào pipeline CF | Báo cáo xác nhận | ✓ Có |
| Compute predicted rating dựa trên user-user similarity | 0.25 | Công thức weighted prediction | Báo cáo: "weighted prediction formula" | ✓ Có |
| OOP model, code compact | 0.25 | Tổ chức theo OOP | Báo cáo: "compact OOP-style units" | ✓ Có |

### C. Experiments (1.0 điểm)

| Tiêu chí | Điểm | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------|------------|---------|------------|
| Chia train/test sets | 0.25 | Divide into training and test | **⚠️ Chưa thấy nhắc trong báo cáo** | ⚠️ Cần check notebook |
| Experiments với nhiều giá trị N | 0.25 | Thử various N values | Báo cáo: "best N = 10", "several values of N" | ✓ Có |
| Bar chart: RMSE theo N | 0.25 | Vẽ bar chart so sánh RMSE | Có `task03_rmse_by_n.png` | ✓ Có |
| So sánh runtime: có vs không có quick approach | 0.25 | Chạy cùng hardware, so sánh thời gian | Có `task03_runtime_compare.png`, báo cáo: 7.916s vs 9.925s | ✓ Có |

---

## V. TASK 4: REPORT (1.0 điểm)

| Tiêu chí | Yêu cầu đề | Thực tế | Trạng thái |
|----------|------------|---------|------------|
| IEEE conference proceeding template | Page 4: "IEEE conference proceeding template" | Báo cáo có cấu trúc IEEE (Abstract, Sections, References) | ✓ Có |
| Overleaf export | "exported from Overleaf" | Có `report.tex` + `report.pdf` | ✓ Có |
| Title | Project title | "Distributed Solutions for the MMDS Final Project" | ✓ Có |
| Authors: members + lecturer last author | Lecturer appended as last author | 3 sinh viên + nguyenthanhan@tdtu.edu.vn là last author | ✓ Có |
| Abstract | Summarize requirements, approaches, results, completion | Có abstract đầy đủ | ✓ Có |
| Task sections với meaningful title | Mỗi section = 1 task, title có nghĩa | I. Task 1, II. Task 2, III. Task 3 | ✓ Có |
| Contributions section | Individual tasks, completion 0-100% | IV. CONTRIBUTIONS: 100% mỗi người | ✓ Có |
| Self-evaluation section | Self-evaluate và estimate scores | V. SELF-EVALUATION: 9.5-10.0/10 | ✓ Có |
| Conclusion section | Summarize lại | VI. CONCLUSION | ✓ Có |
| References IEEE format | IEEE format | Có 4 references định dạng IEEE | ✓ Có |
| Max 5 pages | "Maximal length is 05 pages" | Báo cáo 3 trang | ✓ Có |

---

## VI. CÁC VẤN ĐỀ CẦN KIỂM TRA TRONG NOTEBOOK

Để đạt **điểm tối đa (10/10)**, bạn bắt buộc phải mở và xác nhận các điểm sau trong notebook:

### 1. Task01.ipynb (quan trọng nhất)
- [ ] **Có cả 2 tiêu chí dừng không?**
  - Threshold t: merge if distance ≤ t
  - Terminate khi cluster diameter suddenly jumps
- [ ] **Cell outputs:** Tất cả cells đã được preserve chưa? (Đề yêu cầu "preserving the outputs of all cells")

### 2. Task02.ipynb
- [ ] **Code có tổ chức dạng class với config parameters không?** (0.25 điểm)
  - Không phải replicate blocks of code
  - Dùng class + configuration parameters
- [ ] **Cell outputs:** Tất cả cells đã được preserve chưa?

### 3. Task03.ipynb
- [ ] **Có chia train/test set không?** (0.25 điểm)
  - Đề yêu cầu "Divide the given data set into training and test sets"
  - Báo cáo KHÔNG nhắc đến việc này → Nguy cơ cao là thiếu!
- [ ] **Xử lý 0 = unknown (không phải negative):** Đã implement đúng chưa?
- [ ] **Cell outputs:** Tất cả cells đã được preserve chưa?

### 4. Tất cả notebook
- [ ] **Distributed computation:** Đề yêu cầu "distributed computation is required in this task" cho cả 3 tasks
  - Task 1: PySpark cho data generation và shingle extraction ✓ (báo cáo xác nhận)
  - Task 2: PySpark cho đọc data, CUR, LinearRegression ✓
  - Task 3: PySpark được nhắc trong đề, cần check có dùng không

---

## VII. TỔNG KẾT & ĐÁNH GIÁ RỦI RO

### Điểm dự kiến dựa trên báo cáo:
- Task 1: 2.75-3.0 / 3.0 (nếu thiếu threshold t → 2.5)
- Task 2: 2.75-3.0 / 3.0 (nếu thiếu class config → 2.75)
- Task 3: 2.75-3.0 / 3.0 (nếu thiếu train/test split → 2.75)
- Task 4: 1.0 / 1.0
- **Tổng: 9.25 - 10.0 / 10**

### Nguy cơ mất điểm (cần check gấp):
1. **Task 1 - Thiếu threshold t:** Mất 0.5 điểm (tổng còn 2.5/3)
2. **Task 2 - Thiếu class config:** Mất 0.25 điểm (tổng còn 2.75/3)
3. **Task 3 - Thiếu train/test split:** Mất 0.25 điểm (tổng còn 2.75/3)
4. **Notebook thiếu cell outputs:** Mất điểm tùy mức độ (đề nhấn mạnh "preserving the outputs of all cells")

### Khuyến nghị:
1. Mở `Task01.ipynb` → Check xem có implement cả threshold t và diameter jump không
2. Mở `Task02.ipynb` → Check code organization (class với config chưa)
3. Mở `Task03.ipynb` → Check có chia train/test set không (quan trọng nhất vì báo cáo không nhắc)
4. Kiểm tra tất cả notebook đã "Save and checkpoint" (đảm bảo outputs preserve)

---

**File này được tạo tự động dựa trên việc đối chiếu đề bài và cấu trúc folder nộp. Chưa thực hiện bất kỳ thay đổi nào.**
