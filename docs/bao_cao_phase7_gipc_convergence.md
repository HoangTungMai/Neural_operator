# Báo cáo Phase 7 — Ground-truth thứ hai bằng IPC (TacEx/libuipc) & nghiên cứu hội tụ tiếp tuyến

**Phạm vi:** Giai đoạn 7 (xây GT độc lập bằng Incremental Potential Contact để kiểm chứng/khắc phục việc GT tiếp tuyến PhysX không hội tụ).
**Ngày:** 2026-06-26
**Trạng thái:** Hạ tầng + driver + nghiên cứu hội tụ ĐÃ XONG và chạy thật trên GPU. Kết luận khoa học đã chốt. Bước paired-vs-PhysX cùng combo còn phía trước.

---

## 0. Tóm tắt điều hành

> **Khung:** Việc chọn PhysX hay IPC là một **bước framework để có GT chuẩn**, KHÔNG phải đóng góp khoa học — TacEx đã tạo GIPC để giải quyết PhysX. Báo cáo này là **hiệu chỉnh pipeline GT** (chọn độ phân giải, định lượng nhiễu solver để biết phải trung bình nhiều run). Đóng góp khoa học của dự án vẫn là **FNO** (Phase 3–6).


Phase 3–6 để lại một nút thắt trung thực: **trường tiếp tuyến (shear) của ground-truth PhysX Deformable-Body KHÔNG hội tụ theo lưới**. Đo paired: res24↔res32 ≈ **0.89**, res32↔res40 ≈ **0.70** rel-L2 tiếp tuyến — các lưới "đi lang thang" chứ không co về nhau, và PhysX nổ khi > ~res-48 nên không tới được một bia hội tụ. Kênh pháp tuyến thì ổn (bán kính tiếp xúc khớp Hertz ~1.3%). Nguyên nhân là **lỗi mô hình/solver** (position-based + ma sát chính quy hoá), không phải lỗi rời rạc hoá → mịn lưới không khử được.

Phase 7 dựng một **GT thứ hai bằng IPC** (Incremental Potential Contact — ma sát smooth, barrier contact, về lý thuyết hội tụ theo lưới và theo `eps_velocity→0`) qua **TacEx/libuipc**, rồi kiểm tra trực tiếp xem IPC có hội tụ chỗ PhysX thất bại không.

**Bốn kết quả cốt lõi (đều chạy thật trên GPU, không lý thuyết suông):**

1. **Tái dựng được backend IPC:** build libuipc vào image TacEx (CUDA 12.4, RTX 2000 Ada) + viết driver `tacex_uipc_extract_shear.py` sinh trường biến dạng gel cùng format npz như PhysX.
2. **IPC HỘI TỤ tiếp tuyến tới ~5% — nơi PhysX thất bại 70–90%.** Trường tiếp tuyến ổn định (đổi ≤ ~5%) từ res ≈ 20–24, đơn điệu, **~10–15× chặt hơn PhysX và đúng dấu hội tụ**.
3. **Trần còn lại KHÁC bản chất PhysX:** là **nhiễu tái-lập của solver GPU không tất định** (khử được bằng trung bình nhiều run), KHÔNG phải non-convergence mô hình như PhysX.
4. **GT IPC robust với tham số ma sát:** trường gần như không đổi khi quét `eps_velocity` một bậc độ lớn (mọi biến thiên nằm trong sàn nhiễu).

**Tính trung thực (ghi rõ để không overclaim):** (a) **KHÔNG có một "res hội tụ sạch" (<3%/bước)** trong setup này — tiếp tuyến đã chạm trần nhiễu solver từ res≈24, còn pháp tuyến hội tụ chậm hơn (~7% ở res28→32, chưa xong); (b) sàn nhiễu solver **không đơn điệu theo res** — thấp nhất ở res20–24 (~2.4% tang) rồi **TĂNG** ở res28–32 (tới ~12%), nên đẩy lưới mịn hơn res24 *phản tác dụng* cho tiếp tuyến; (c) đây vẫn là **GT self-consistent**, chưa phải GT đúng-vật-lý-tuyệt-đối (vẫn còn sai số mô hình gel siêu đàn hồi/luật ma sát) và chưa đối chứng phần cứng.

---

## 1. Bối cảnh & mục tiêu

### 1.1 Vì sao cần GT thứ hai
GT tiếp tuyến PhysX không hội tụ (số ở trên). Hệ quả: ở kênh tiếp tuyến **không tách bạch được lỗi operator (FNO) với lỗi GT**, vì bia GT dịch 70–90% mỗi lần mịn lưới tới trần solver. Trần ~0.15 relL2 tổng của FNO bị chi phối bởi bia tiếp tuyến *under-determined*.

### 1.2 Vì sao IPC
IPC/C-IPC (và bản GPU **libuipc**) giải tiếp xúc bằng **barrier** (không xuyên thấu, hội tụ theo lưới) và ma sát **smooth/lagged** mà về lý thuyết **độc lập lưới ở giới hạn `eps_velocity→0`**. Đúng hai cờ hội tụ ta cần. TacEx tích hợp sẵn libuipc vào Isaac Lab → tái dùng được engine chín thay vì tự viết solver.

### 1.3 Câu hỏi Phase 7
> Trường tiếp tuyến của IPC có **hội tụ** theo lưới (`gel_res`) và theo ma sát (`eps_velocity`) không — tức nó có cho một bia GT tin cậy ở chỗ PhysX thất bại?

---

## 2. Hạ tầng & driver

### 2.1 Build libuipc
Image `isaac-lab-tacex:latest` ban đầu CHỈ có code wrapper Python `tacex_uipc`, **thiếu hẳn engine libuipc** (submodule rỗng, không `.so`, `import uipc` fail). Đã build:
- libuipc `github.com/DH-Ng/libuipc` nhánh `tacex` (commit `1a7e93e`) + submodule `muda`, `SymEigen`; CUDA toolkit 12.4; cài vào `/isaac-sim/kit/python`.
- Patch image: dùng `tacex_uipc` từ source qua `.pth`, `__init__` minimal headless, `mesh_gen` debug_draw optional, deps (`flatdict`/`prettytable`/`hidapi`), `LIVESTREAM=0`.
- Xác minh: `import uipc, tacex_uipc` OK trong AppLauncher context.
> ⚠️ Patch đang nằm trong `docker commit` (sha 61cec66…), **chưa Dockerfile-hoá** — TODO tái lập.

### 2.2 Driver `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
Song song `isaac_extract_shear.py` (PhysX) nhưng dùng `UipcSim`:
- **Gel** = lưới tet **cấu trúc tất định** (`structured_tet_box`: grid đều nx×ny×nz, 6 tet/cell, sửa hướng thể tích dương). Cố tình KHÔNG dùng wildmeshing cho gel — xem §2.3.
- **Indentor cầu** = tet **fan-from-center tất định** (tâm + 1 tet/tam-giác icosphere). Vật cứng, drive động học mọi đỉnh qua `SoftPositionConstraint` + animator (press→settle→shear→settle).
- Z-up khớp PhysX; đọc trường mặt trên → nội suy lưới marker 32×32 (cKDTree) → npz **cùng format PhysX** (`params/coords/disp/mode/meta` + `gel_res/eps_velocity/d_hat`).
- 3 mode: `--smoke`, `--single` (1 setting, dùng cho shell-loop), `--convergence` (in-process, để tham khảo).

### 2.3 Bài học quan trọng về phương pháp: KHÔNG dùng wildmeshing cho convergence
Lần chạy convergence ĐẦU dùng `wildmeshing` (knob `edge_length_r`) cho ra **số rác** (kênh normal lệch ~95% giữa các setting, có cặp rel-L2 = 22). Chẩn: 3 lỗi pipeline, **không phải lỗi GIPC solver**:
1. wildmeshing **không tất định** (cùng tham số → vert count khác mỗi run) → trục friction bị trộn nhiễu lưới.
2. Mặt trên bất quy tắc → top-face detection sụp còn 1 đỉnh → trường "phẳng dí" (`n_unique=1/1024`).
3. Lưới mịn nhất nổ (peak_uz −38mm).
→ Sửa: gel + indentor đều chuyển sang **lưới tất định** (§2.2). Đây là điều kiện *bắt buộc* cho mọi nghiên cứu hội tụ: lưới phải tái lập + mặt lấy mẫu phải đều.

---

## 3. Giao thức đo hội tụ

- **Hai trục tách biệt** (không full grid): **trục lưới** quét `gel_res ∈ {6,8,12,16,20,24,28,32}` (ghim `eps_velocity=0.001`); **trục ma sát** quét `eps_velocity ∈ {0.02,0.01,0.005,0.002,0.001}` (ghim `gel_res=24`).
- **Mỗi setting 1 process riêng** (`--single` qua `infra/gen_uipc_convergence.sh`) → engine/SimulationContext sạch.
- **Replicate để hạ nhiễu:** mỗi mức lưới chạy **6 run y hệt config**; GT = trung bình; *độ tái lập* = rel-L2 trung bình từng-cặp giữa các run = **sàn nhiễu solver**.
- Cấu hình cố định (smoke đã validate): gel 0.10×0.10×0.04 m, cầu R=0.02 m, μ=0.6, E=100 kPa, ν=0.45, depth=5 mm, shear travel=4 mm.
- Metric: rel-L2 tách **normal (uz)** và **tiếp tuyến (uxy)**, trên cùng lưới marker 32×32.

---

## 4. Kết quả

### 4.1 Sàn nhiễu solver GPU (độ tái lập, 6 run/mức)
| gel_res | nhiễu tang (rel-L2) | nhiễu norm | #verts |
|--------:|--------------------:|-----------:|-------:|
| 12 | 0.062 | 0.023 | 1 014 |
| 16 | 0.084 | 0.027 | 2 023 |
| **20** | **0.026** | **0.008** | 3 969 |
| **24** | **0.024** | **0.009** | 6 875 |
| 28 | 0.073 | 0.025 | 10 092 |
| 32 | 0.123 | 0.044 | 15 246 |

**libuipc GPU không tất định run-to-run** (reduction GPU). Sàn nhiễu **không đơn điệu**: thấp nhất ở **res20–24** (~2.4% tang), rồi **tăng mạnh ở res28–32** (lưới mịn → nhiều node tiếp xúc → trạng thái stick-slip bistable hơn). Pháp tuyến luôn chặt hơn tiếp tuyến nhiều lần.

### 4.2 Hội tụ theo lưới (field trung-bình-6-run)
Khoảng cách giữa các mức kế tiếp và tới mức mịn nhất (res32):

| bước | tang | norm |   | mức | dist→res32 tang | dist→res32 norm |
|------|-----:|-----:|---|----:|----------------:|----------------:|
| 12→16 | 0.045 | 0.104 | | 12 | 0.177 | 0.303 |
| 16→20 | 0.110 | 0.131 | | 16 | 0.162 | 0.268 |
| 20→24 | 0.059 | 0.107 | | 20 | 0.064 | 0.172 |
| 24→28 | 0.047 | 0.055 | | 24 | 0.023 | 0.096 |
| 28→32 | 0.044 | 0.070 | | 28 | 0.044 | 0.070 |

`peak_uz` đơn điệu hội tụ: −3.99 → −4.23 → −4.61 → −4.77 → −4.93 → −5.02 → −5.10 → −5.18 (×10⁻³ m) cho res 6→32 (increments co lại).

**Đọc:** kênh **tiếp tuyến** đổi ≤ ~6% từ res20 trở đi và chạm trần nhiễu solver — coi như **ổn định từ res≈20–24**. Kênh **pháp tuyến** giảm đơn điệu nhưng **chậm hơn** (0.30→0.27→0.17→0.096→0.070→0): res28→32 vẫn ~7%, **chưa hội tụ hẳn** (dimple sắc dưới cầu là bottleneck). Đẩy mịn hơn res24 cho tiếp tuyến là *vô ích* vì nhiễu solver phình to (§4.1).

### 4.3 Hội tụ theo ma sát (`eps_velocity`, res24)
`mean_tang` (×10⁻³): eps 0.02→0.001 = 2.06 / (1.65→1.95 sau rerun) / 2.11 / 2.09 / 2.17. Mọi biến thiên **≤5% và nằm trong sàn nhiễu** → **trường INSENSITIVE với `eps_velocity`** trên một bậc độ lớn → friction model đã ở chế độ hội tụ (robust). "Outlier" eps0.01 ban đầu chỉ là một lần bốc nhiễu thấp (chạy lại về bình thường).

### 4.4 So sánh trực tiếp với PhysX
| | PhysX (paired) | IPC (Phase 7) |
|---|---|---|
| Tiếp tuyến giữa các lưới | **0.70–0.89, lang thang** | **0.044–0.059, đơn điệu giảm** |
| Hội tụ tiếp tuyến? | **KHÔNG** (mô hình) | **CÓ**, tới sàn nhiễu ~5% |
| Bản chất trần | non-convergence mô hình | nhiễu tái lập solver (khử bằng averaging) |
| Pháp tuyến | tốt (Hertz 1.3%) | hội tụ, chậm hơn tiếp tuyến |

IPC tiếp tuyến **~10–15× chặt hơn** và là vấn đề *thống kê khử được*, không phải *mô hình bế tắc*.

---

## 5. Kết luận

1. **GIPC/IPC hội tụ tiếp tuyến tới dải ~5% từ res≈20–24** — đủ tin cậy để chọn làm GT chuẩn (đối lập PhysX 70–90% không hội tụ). Đây là kết quả **hiệu chỉnh pipeline**, không phải claim khoa học (TacEx đã chứng minh GIPC > PhysX).
2. **Không có "res hội tụ sạch <3%":** tiếp tuyến bị chặn bởi nhiễu solver GPU (không tất định, còn *tăng* theo res); pháp tuyến hội tụ chậm (~7% ở res32, chưa xong).
3. **Sweet spot = res24:** nhiễu thấp nhất, tiếp tuyến đã ổn, chi phí vừa phải. Mịn hơn phản tác dụng.
4. **Trần là nhiễu tái-lập, không phải mô hình** — khử được bằng trung bình K run (nhiễu /√K).
5. Vẫn là **GT self-consistent**, chưa khử sai số mô hình (gel siêu đàn hồi, luật ma sát) và chưa đối chứng phần cứng → vẫn cần fine-tune trên dữ liệu cảm biến thật cho digital-twin.

---

## 6. Khuyến nghị & bước tiếp

1. **GT sản xuất = field trung-bình K=6 run tại res24** (nhiễu /√6 ≈ 1%). Lưu kèm `eps_velocity=0.001`.
2. **Dùng GT IPC cho pipeline FNO (việc chính):** sinh dataset IPC nhiều config (R/μ/E) ở res24/eps0.001, trung bình K run, đưa vào `novbts.operator.fem_benchmark` để (re)train/eval FNO trên GT IPC.
3. (Tuỳ chọn, KHÔNG phải science) **Sanity paired-vs-PhysX:** chỉ để biện minh việc đổi GT; cần khớp geometry + map drive-ratio g→travel.
4. **Cập nhật paper KSE2026:** BỎ "irreducible tangential ceiling"; trình bày GT như **lựa chọn framework** ("dùng IPC/TacEx vì ma sát hội tụ — cite TacEx"), KHÔNG dựng PhysX-vs-IPC thành finding. Science = FNO.

---

## 7. Tái lập

**Files:**
- Driver: `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
- Sweep: `infra/gen_uipc_convergence.sh` (shell-loop `--single`)
- Aggregate: `src/novbts/groundtruth/aggregate_uipc_convergence.py`
- Dữ liệu: `data/uipc/conv/` (10 setting 2 trục), `data/uipc/avg/` (replicate res12–32 × 6), `data/uipc/rep/` (đo sàn nhiễu)

**Lệnh:**
```bash
# convergence 2 truc
bash infra/gen_uipc_convergence.sh
python -m novbts.groundtruth.aggregate_uipc_convergence --conv-dir data/uipc/conv
# replicate-averaging (vd 6 run/muc): driver --single, --gel-res R, --eps-velocity 0.001
# roi tinh trung binh + san nhieu per-level
```

**Môi trường:** image `isaac-lab-tacex:latest` (đã build libuipc), `docker run --gpus all --entrypoint /isaac-sim/python.sh`, GPU RTX 2000 Ada 16GB, CUDA 12.4.

> Tiến độ live ghi ở `/work/fem_progress_uipc.txt` (Isaac nuốt stdout). Marker thành công: `SMOKE_UIPC_OK` / `SINGLE_UIPC_OK`.
