# Báo cáo tổng kết dự án — Neural-Operator surrogate cho cảm biến xúc giác VBTS

**Phạm vi:** Giai đoạn 3 → 6 (neural operator + cảm biến khả vi + điều khiển + framework)
**Ngày tổng kết:** 2026-06-17
**Trạng thái:** Gate 3 đóng; framework cảm biến VBTS khả vi (Nhánh A) đã đủ 4 mảnh; còn lộ trình Isaac-Sim robot-gắn-cảm-biến phía trước.

---

## 0. Tóm tắt điều hành

Dự án xây một **toán tử thần kinh (Fourier Neural Operator, FNO)** học trường dịch chuyển marker của một cảm biến xúc giác thị-giác (Vision-Based Tactile Sensor — VBTS), làm **surrogate khả vi, độ-trung-thực-cỡ-FEM, nhẹ phần cứng** thay cho solver tiếp xúc đắt đỏ. Đích cuối là hai nhánh ghép vào nhau:

- **Nhánh A — môi trường mô phỏng cảm biến VBTS** (sinh ground truth + xuất ảnh cảm biến + env điều khiển).
- **Nhánh B — FNO surrogate khả vi** làm lõi nhanh trong vòng điều khiển.

Bốn kết quả cốt lõi, mỗi cái đều đã kiểm chứng trên **vật lý FEM thật** (PhysX Deformable-Body trong Isaac Sim), không chỉ giải tích:

1. **Framing field→field là chìa khóa** (Phase 3): FNO thắng MLP per-point **2.24×** trên FEM thật (6.7× trên giải tích) — vì chuyển vị bề mặt là hàm **phi cục bộ** (Green's function), MLP cục bộ không nắm được.
2. **Khả vi → điều khiển hiệu quả** (Phase 4): học policy bằng backprop qua FNO đóng băng cần **~64× ít truy vấn / ~24× nhanh hơn** so với phương pháp gradient-free (Evolution Strategies), cùng chất lượng cuối.
3. **Cảm biến khả vi end-to-end** (Phase 5): trường dịch chuyển → ảnh marker-dot bằng torch → `render∘FNO` khả vi; khôi phục được lực kéo (sx,sy) **từ chính ảnh cảm biến** với sai số 2.1%.
4. **Framework hoàn chỉnh** (Phase 6): env khả vi gói FNO+sensor+reward (6a); đo sàn nhiễu camera (6b); chứng minh trạng thái cuối không phụ thuộc đường tải (6c); mở rộng hình học vật (6d).

**Tính trung thực xuyên suốt** (ghi rõ để không overclaim): (a) trần tiếp tuyến ~0.15 relL2 / ~14° là **giới hạn nội tại**, không phá được bằng mesh mịn hơn / model lớn hơn / lịch sử tải; (b) tốc độ thô **không** phải lợi thế bền (sim vật lý GPU như Taccel còn nhanh hơn về throughput) — lợi thế bền là **tổ hợp latency-thấp + khả-vi-rẻ + nhẹ-HW + mesh-free**; (c) so với mạng SOTA (U-Net) lợi thế FNO hẹp lại còn ~1.03×.

---

## 1. Bối cảnh & mục tiêu

### 1.1 Bài toán
Cảm biến VBTS (GelSight/DIGIT/marker-gel…) cho phép robot "nhìn" tiếp xúc qua biến dạng của một lớp gel mềm in chấm marker. Mô phỏng nó chính xác đòi hỏi giải tiếp xúc đàn hồi-ma sát — đắt, không khả vi, khó dùng trong vòng điều khiển/RL. Ý tưởng dự án: **học một toán tử** thay solver.

### 1.2 Ba lớp mô hình (phân vai rõ ràng)
| Lớp | Vai trò | Công cụ |
|---|---|---|
| Ground truth chính xác-chậm | sinh dữ liệu + làm thước đo | **PhysX FEM** (Isaac Sim, deformable body) |
| Validator giải tích | kiểm tra GT + baseline vật lý | **Hertz + Cattaneo–Mindlin** |
| Surrogate nhanh-khả-vi | đóng góp chính | **FNO** (Fourier Neural Operator) |

### 1.3 Hai nhánh đích
- **B (FNO):** đã chín — Gate 3 đóng (Phase 3), điều khiển khả vi (Phase 4).
- **A (môi trường cảm biến):** Phase 5 thêm mô hình cảm biến marker-dot; Phase 6 hoàn thiện thành framework dùng được. Còn lại: env Isaac-Sim thật với robot gắn cảm biến (lộ trình §9).

---

## 2. Nền tảng (trước Phase 3): ground truth & validator

- **Generator FEM** (`novbts.groundtruth.isaac_extract_{normal,shear}`): chạy standalone trong Docker image `isaac-lab-fem`. Gel 50×50×20 mm, indentor mặc định cầu r=0.02 m, lưới marker 32×32 = **1024 điểm probe**. Xuất `disp[N,M,3]` (ux,uy,uz) bề mặt gel.
  - Shear/slip qua **micro-steps kéo tiếp tuyến** (phá deadlock tiếp xúc bằng lún nông + nhiều bước nhỏ); ~3–4.6 s/frame ở res-24.
  - Normal indentation ~0.47 s/frame.
- **Validator Hertz–Mindlin** (`hertz_mindlin.py`): bán kính tiếp xúc lệch 1.3%, hội tụ 0.5% so với công thức giải tích → GT FEM đáng tin ở chế độ pháp tuyến.
- **Taxonomy 4 chế độ tiếp xúc** (`contact-mode-design`): no-contact / stick / partial-slip / full-slip theo ngưỡng kéo tiếp tuyến g; lưu ở `params[:,*]` + `mode`. Quyết định **không thêm chế độ** (spin/torsion chỉ làm nếu task RL có xoay).

---

## 3. Phase 3 — Neural operator (Gate 3)

**Mục tiêu:** trả lời RQ1 (độ chính xác), RQ2 (tổng quát hóa/OOD), RQ3 (tốc độ), và phân loại chế độ trượt (slip heads). Gate 3 = ngưỡng để tuyên bố operator "đủ tốt".

### 3.1 Khám phá then chốt: framing **field→field** (không phải param→field)
- **param→field** (mỗi điểm marker được mớm full tham số → MLP): MLP per-point **thắng** FNO (0.066 vs 0.079) vì bài toán trở thành cục bộ. Đây giờ chỉ là **ablation phản chứng**.
- **field→field** (đầu vào là *trường* profile lún + shear·mask; đầu ra là *trường* dịch chuyển): MLP per-point **sụp đổ** (0.743 relL2, hướng sai 63°) còn FNO giữ 0.111 → **FNO thắng 6.7×** (trên giải tích).
- **Luận điểm:** chuyển vị bề mặt là hàm **phi cục bộ** (đáp ứng đàn hồi kiểu Green's function — một điểm lún ảnh hưởng cả vùng). FNO học toán tử tích chập phổ toàn cục; MLP cục bộ không thể. → **Đây là đóng góp khoa học chính của dự án.**

### 3.2 Gate 3 đóng trên FEM thật (2000 frame swept, res-24, quét R/μ/E)
| Chỉ số | Kết quả |
|---|---|
| **RQ1** FNO overall relL2 | **0.144–0.146** (hướng tiếp tuyến 14.6–14.8°) |
| FNO thắng MLP per-point | **2.24×** (FNO 0.146 vs MLP 0.328; hướng 14.8° vs 35.7°) |
| **Slip-F1** head-a (multitask) | **0.904** (>0.75 ✓) |
| Slip-F1 head-b (riêng) | **0.753** sau khi cứu (xem 3.3) |
| **RQ2** ngoại suy đuôi R/μ/E | ~**1.3×** cả ba trục (mượt, trong-hộp) |
| RQ2 OOD hình học (flat punch) | 6.2× (mã hóa hình học vào trường lún tổng quát tốt hơn one-hot) |
| **RQ3** tốc độ | FNO ~7868 fps vs FEM-shear 0.341 fps = **~23.000×** |

> Luận điểm phi-cục-bộ **giữ vững trên vật lý thật** (giải tích 6.7× → FEM 2.24×, hẹp hơn vì FEM nhiễu/ít lý tưởng).

### 3.3 Cứu slip-classifier head-b
head-b suy biến (normal F1 = 0, macro 0.595) do **mất cân bằng lớp** (sweep `g~U(0,1.3)` → chỉ 3.1% frame normal). Khắc phục: thêm **400 frame normal thuần** (g=0) → tỉ lệ normal 3.1%→19.3%; **head-b normal F1 0→0.851, macro 0.595→0.753** (vượt ngưỡng). Regression accuracy không đổi → xác nhận lỗi do dữ liệu, không phải mô hình. Data: `shear_fine_swept_normaug.npz` (2400 frame) — **bộ chuẩn dùng cho mọi phase sau**.

### 3.4 Bake-off với các mô phỏng VBTS tiêu biểu (cô lập giá trị)
Cài lại **lõi chuyển động marker** của từng phương pháp, fit + đo trên *cùng* GT FEM (không trích số liệu paper vì không so trực tiếp được):

| Phương pháp | overall relL2 | hướng° | FNO hơn |
|---|---|---|---|
| TACTO-style (động học, no friction) | 0.504 | 65.8 | **3.51×** |
| Cattaneo–Mindlin giải tích (đã hiệu chỉnh affine) | 0.435 | 39.0 | **3.03×** |
| Taxim/FOTS-style (tuyến tính chồng chập) | 0.295 | 26.2 | **2.05×** |
| MLP per-point (cục bộ) | 0.321 | 33.6 | 2.24× |
| **FNO (ours)** | **0.144** | **14.6** | — |

**Đọc kết quả:** vật lý giải tích kinh điển (Cattaneo–Mindlin) dù hiệu chỉnh biên độ vẫn **thua mô hình tuyến tính fit-data** (affine không đổi được *hình dạng* profile; trường FEM lệch khỏi Hertz-Mindlin lý tưởng). Taxim/FOTS tuyến tính là baseline mạnh nhất nhưng FNO vẫn hơn 2.05× vì chuyển stick→partial→full slip là **phi tuyến**. → Đóng góp FNO = **phi tuyến + phi-cục-bộ cho trường slip**.

So với **mạng SOTA** (cùng nhóm dense/operator): DeepONet 0.210, Galerkin-Transformer 0.210, **U-Net 0.148 (≈ ngang FNO, ít hơn 5.7× params)**, FNO 0.144. → Trung thực: lợi thế thu hẹp còn 1.03–1.46×; thông điệp đúng là *"học dense/operator phi-cục-bộ thắng mô hình vật lý/tuyến tính"*, **không** phải "FNO độc tôn".

### 3.5 Sắc thái tốc độ (quan trọng, tránh ngộ nhận)
"~23.000×" chỉ so với **PhysX-FEM solver đơn-luồng** (0.34 fps). Sim vật lý GPU hiện đại (Taccel: 915 fps × 4096 env trên H100) có **throughput gộp vượt FNO**. → Lợi thế bền của FNO **không** phải fps thô mà là tổ hợp: (1) latency ~0.125 ms/frame; (2) khả vi rẻ qua autograd; (3) nhẹ phần cứng (RTX 2000 16 GB); (4) mesh-free, học từ bất kỳ GT. Các sim vật lý đầy đủ (DiffTactile/TacIPC/TacEx/Taccel) là **bổ trợ** (GT tốt hơn để train), không cạnh tranh surrogate.

---

## 4. Phase 4 — Điều khiển qua FNO khả vi

**Module:** `novbts.operator.diff_policy`. **Câu hỏi:** backprop qua FNO khả vi có hiệu quả mẫu hơn gradient-free coi FNO là hộp đen không? + kiểm tra phản đề Suh et al. 2022 ("differentiable simulators có cho gradient policy tốt hơn không?").

**Thiết kế:** FNO **đóng băng** (eval + `requires_grad_(False)`), gradient chỉ chảy tới *action*. Vì FNO là map tĩnh một-bước → đây là **policy ngữ-cảnh một-bước (amortized)**, không phải RL đa-bước. Task chính = **servo**: policy π(context μ,E,R,geom + tóm tắt trường target) → action (sx,sy); loss = ‖FNO(action) − y*‖².

**Kết quả (3 seed, `runs/phase4/policy_servo.json`):**
| | final loss | truy vấn (fwd) | wall |
|---|---|---|---|
| autograd | 8.02e-8 | **300** | **13 s** |
| ES (pop 32) | 8.36e-8 | 19 200 | 313 s |

- Cùng chất lượng cuối, autograd **~64× ít truy vấn / ~24× nhanh hơn**.
- Khoảng cách amortization: cả hai ~8e-8 vs per-instance oracle 1.6e-8 (một mạng cho mọi ngữ cảnh không bằng tối ưu từng-lần — đúng kỳ vọng).
- **Phản đề Suh KHÔNG cắn:** gradient autograd sạch ở cả stick lẫn slip. Lý do tinh tế: FNO **làm trơn** bước nhảy stick→slip (nhược điểm spectral low-pass) → gradient mượt — *dở cho độ chính xác forward nhưng tốt cho chất lượng gradient*. Câu chuyện nhất quán với trần tiếp tuyến (§7).
- **Task A (anti-slip) bỏ trung thực:** probe cho thấy `slip_proxy` *tăng* theo depth = hiệu ứng diện tích, không phải giảm trượt → tối ưu nó là bad science → không xây.

---

## 5. Phase 5 — Mô hình cảm biến marker-dot khả vi

**Subpackage:** `src/novbts/sensor/`. **Đòn bẩy:** generator đã xuất `disp[N,M,3]`; ảnh cảm biến = **chiếu + render chấm của trường có sẵn** (không cần giải FEM mới). Viết bằng torch → khả vi → `render∘FNO` khả vi end-to-end.

- **`markercam.py`:** `PinholeCamera.from_gel` (camera *dưới* màng nhìn lên +z, auto-fit nội tham số để chấm vừa khung), `deformed_marker_xyz`, `render_dots` (Gaussian-splat khả vi, hỗ trợ polarity tối/sáng + bão hòa), `sample_field_to_markers`, `track_flow_known/_image`.
- **`build_sensor_dataset.py`:** npz FEM → npz cảm biến (pix_rest/pix_def/pix_flow + cam config) + preview.

**Kết quả (`runs/phase5/`):**
| Chỉ số | Giá trị |
|---|---|
| Faithful: cos(flow, disp_xy) | **0.973** (pixel flow mã hóa trung thực dịch chuyển trong mặt phẳng) |
| Round-trip render→track (centroid) | 1.9 px @160 px |
| Compat FNO+renderer (marker-flow rel-L2) | 0.26 (khớp trần tiếp tuyến FNO — flow chỉ còn kênh tiếp tuyến khó) |
| **Inverse từ ẢNH** (sx,sy qua autograd render∘FNO) | **2.1% / 1.1°** (ngang 2.3% từ trường thô) |

→ Pipeline cảm biến khả vi end-to-end **chạy**: khôi phục được hành động từ chính ảnh marker, không chỉ từ trường thô.

---

## 6. Phase 6 — Hoàn thiện framework (4 hướng)

Thứ tự ưu tiên thấp-rủi-ro-trước. Plan: `docs/phase6_plan.md`.

### 6a — Env khả vi (framework core) [TRỌNG TÂM]
**Module:** `src/novbts/sensor/tactile_env.py`. Gói **FNO đóng băng (P3) + cảm biến marker-dot (P5) + reward** sau một API duy nhất.

- **Trung thực phạm vi:** FNO là map tĩnh một-bước → env là **single-step contextual** (goal-conditioned), không phải động lực nhiều bước. **Chỉ dùng sphere (geom=0)** theo yêu cầu (đơn giản nhất).
- `reset()` → lấy context sphere + render **target imprint** (mục tiêu xúc giác). `differentiable_step(action sx,sy)` → FNO → sample markers → render_dots → ảnh camera; **reward = −MSE(ảnh, target)** (mode image hoặc flow), khả vi. `step()` kiểu gym (detached) + thêm nhiễu camera lên observation. Adapter `gymnasium` nếu cài được.
- Tích hợp **PolicyMLP của Phase 4**, huấn luyện *qua* `env.differentiable_step`.

**Verify (`--demo`, `runs/phase6/env_demo.json`):**
| | mean reward (test, cao=tốt) |
|---|---|
| hành động ngẫu nhiên | −3.23e-3 |
| **Phase-4 policy** | **−6.52e-4** |
| hành động thật (oracle) | −2.72e-4 |

→ Policy đóng **87%** khoảng cách ngẫu-nhiên→oracle. Gradcheck finite-diff: rel_err 3.7% → gradient chảy tới action ✓. Preview `env_demo.png`: cột policy khớp target, random lệch hẳn.

### 6b — Realism + sim2real scaffold
**Module:** `sensor/realism.py`, `sensor/calibration.py`. Nhiễu camera Poisson-Gaussian + tracker centroid thật → đo **sàn nhiễu** (`runs/phase6/realism.json`).

- Thang đo chuẩn cảm biến = **EPE pixel**. GT-vs-FNO: EPE **0.54 px** (pitch marker 12 px).
- Quét read-noise 0→8%, tách per-marker: **jitter** (sàn nhiễu ngẫu nhiên) 0.055→0.107 px; **bias** (độ phân giải tracker) 0.31 px.
- **Kết quả (đảo kỳ vọng):** sai số FNO 0.54 px **> sàn nhiễu @2% (0.06 px)** và **> độ phân giải tracker (0.31 px)**. → **Nhiễu camera KHÔNG che được sai số FNO** (chấm to/sáng, centroid trung bình hóa nhiễu tốt). Muốn cải thiện: nâng FNO hoặc tracker sub-pixel, **đừng** đổ tiền vào camera ít nhiễu.
- `calibration.py`: chỉ schema `SensorCalib` (cầu nối sim↔phần cứng) + checklist; **chưa fit** (chờ hardware thật).

### 6c — Temporal + lịch sử tải
**Modules:** `sensor/temporal.py`, `operator/loading_history.py`; generator thêm `--save-trajectory` + 3 đường tải (linear/ortho/reverse) cùng endpoint.

- Render quỹ đạo kéo thành **video marker-dot** (progressive stick→slip) + đường cong slip-vs-tải. Đường tải reverse có hiện overshoot-rồi-về ở f≈0.71 (transient).
- **Test path-dependence (NULL):** FNO endpoint-only (3ch) 0.161 vs +load-mode one-hot (6ch) 0.171 (không giúp); model-free kNN distance-matched cross/same = **1.00**. → **Trạng thái cuối CHỈ phụ thuộc endpoint (sx,sy,depth), KHÔNG phụ thuộc đường đi.** Tiếp xúc nông + dịch nhỏ ≈ đàn hồi thuần, không trí nhớ ma sát.
- → Thành phần "bất khả giảm" của trần tiếp tuyến **không** phải lịch sử tải; là FNO làm trơn tiếp xúc sắc.

### 6d — Hình học vật (Isaac, rủi ro nhất)
Generator thêm `--indentor-geom {sphere,flat,cylinder,mesh}`. Render qua *cùng* cảm biến marker-dot (`sensor/object_geometry.py`, `runs/phase6/object_geometry.json`):

| Hình | mean_tang (m) | contact-area | mean marker flow |
|---|---|---|---|
| sphere | 0.0004 | 39% | **1.7 px** |
| cylinder | 0.0016 | 93% | **8.4 px** |
| flat punch | 0.0022 | 100% | **10.5 px** |

→ **Hình học định hình mạnh trường xúc giác** (flow khác nhau ~6×). mesh (UsdFileCfg) đã wire nhưng chưa test (cần file USD). cylinder chậm (~30 s/frame).

---

## 7. Phát hiện xuyên suốt: trần tiếp tuyến

Một sợi chỉ đỏ qua nhiều phase. Trần ≈ **0.15 relL2 / 14°** ở thành phần tiếp tuyến. Đã thử phá bằng nhiều đòn bẩy, **tất cả âm tính**:

| Đòn bẩy | Kết quả |
|---|---|
| Data-scaling (200→1600 frame) | bão hòa ~1600; gấp đôi chỉ mua ~0.01 |
| Model capacity (FNO 12/48→16/64, 3× params) | 0.146→0.151 (không giúp) |
| Lưới mịn hơn (res-24→res-32) | 0.146→0.158 (không hạ; res-32 còn khó fit hơn ở cùng modes) |
| Nhiều Fourier modes (12→16→20) | tệ hơn / lỗi (>Nyquist lưới) |
| Đổi biểu diễn đầu vào — opt1 (mớm trường Mindlin) | **NULL** (Mindlin là hàm tất định của input có sẵn) |
| opt2 (mask ranh giới sắc + tọa độ bán kính) | lợi nhỏ thực ~5% (vá phần "FNO làm trơn bước nhảy") |
| U-Net (conv cục bộ) | 13.4° ≈ opt2 → triangulation cùng một cơ chế |
| Lịch sử tải (Phase 6c) | **NULL** (trạng thái cuối chỉ phụ thuộc endpoint) |

**Kết luận:** trần phân rã 2 thành phần: (1) *sắc-nhưng-FNO-trơn* (~5%, vá được nhỏ); (2) **bất khả giảm từ tham số macro** (phần lớn — entropy cao ở mép tiếp xúc stick-slip, hoặc giới hạn biểu diễn marker 1024 điểm). **Không phải** do GT-fidelity res-24, **không phải** dung lượng model. Pháp tuyến (uz) đã hội tụ tốt.

> **Sắc thái cho sim2real:** res-24 *đủ* cho nghiên cứu operator (so apples-to-apples, trần độc lập lưới). Nhưng để làm **digital-twin calib với cảm biến thật**, tiếp tuyến res-24 *chưa hội tụ* (lệch res-32 ~78%) → cần convergence study. Tuy nhiên hội tụ lưới chỉ khử sai số rời rạc, không khử sai số mô-hình (gel siêu đàn hồi, luật ma sát). → Đường rẻ hơn: **pretrain FEM (res nào cũng được) rồi fine-tune thẳng trên dữ liệu cảm biến thật** — để thực tại sửa cả hai. Vì vậy chưa làm res cao bây giờ (vô nghĩa khi chưa có hardware).

---

## 8. Hạn chế & tính trung thực

1. **Env single-step:** FNO là map tĩnh → `tactile_env` chưa mô phỏng động lực nhiều bước. Multi-step cần FNO có trạng thái thời gian.
2. **Tốc độ không phải lợi thế bền** (§3.5).
3. **Lợi thế vs SOTA hẹp** (U-Net ≈ ngang FNO) (§3.4).
4. **Trần tiếp tuyến không phá được** bằng các đòn bẩy đã thử (§7).
5. **Chưa có hardware:** calibration mới là schema; sim2real chưa đóng vòng với cảm biến thật.
6. **Hình học mesh chưa test** (6d); cylinder chậm.
7. **Một số kết quả single-seed** (opt2 input-augment) cần đa-seed xác nhận.

---

## 9. Trạng thái framework & lộ trình tiếp

### 9.1 Framework hiện có (dùng được ngay, pure-Python + GPU)
```
reset() / step() / differentiable_step()      <- tactile_env.py (6a)
  |- frozen FNO surrogate                      <- field2field.py (P3)
  |- differentiable marker-dot sensor          <- markercam.py  (P5)
  |- camera noise + tracker                    <- realism.py    (6b)
  |- PolicyMLP                                 <- diff_policy.py(P4)
```
Một API duy nhất để huấn luyện/chạy policy điều khiển tới mục tiêu xúc giác, quan sát qua ảnh cảm biến (có nhiễu), khả vi toàn trình.

### 9.2 Lộ trình (theo mong muốn người dùng)
1. **Env Isaac-Sim thật với robot gắn cảm biến** — render môi trường 3D, tay máy mang gel VBTS, FNO làm lõi nhanh thay solver trong vòng điều khiển. (Mảnh lớn còn lại của Nhánh A.)
2. **Multi-step dynamics** — nâng env lên nhiều bước (liên quan FNO có thời gian).
3. **Sim2real** — dựng cảm biến DIY, fit `SensorCalib`, fine-tune FNO trên dữ liệu thật (đường khử cả sai số lưới lẫn mô-hình).
4. **Hình học mesh** vật thật (cần file USD).

---

## 10. Phụ lục

### 10.1 Bản đồ code (package `novbts`)
| Module | Vai trò |
|---|---|
| `models.py` | FNO / MLP / DeepONet / SpectralConv2d |
| `paths.py` | path config (ROOT/DATA/RUNS/DOCS) |
| `groundtruth/hertz_mindlin.py` | GT giải tích + validator |
| `groundtruth/isaac_extract_{normal,shear}.py` | GT FEM (Docker), + `--save-trajectory`, `--indentor-geom`, `--gmin/gmax` |
| `operator/field2field.py` | **HEADLINE** field→field FNO vs MLP |
| `operator/fem_benchmark.py` | RQ1–RQ3 trên FEM thật |
| `operator/diff_policy.py` | **Phase 4** policy autograd vs ES + PolicyMLP |
| `operator/vbts_baselines.py` | bake-off TACTO/Taxim/FOTS/Mindlin + SOTA |
| `operator/{input_augment,loading_history,transfer_benchmark}.py` | thí nghiệm trần tiếp tuyến |
| `sensor/markercam.py` | **Phase 5** camera + render chấm khả vi |
| `sensor/tactile_env.py` | **Phase 6a** env khả vi (framework core) |
| `sensor/{realism,calibration}.py` | **6b** nhiễu + calib schema |
| `sensor/temporal.py`, `operator/loading_history.py` | **6c** temporal + lịch sử tải |
| `sensor/object_geometry*.py` | **6d** hình học vật |

### 10.2 Dữ liệu (gitignored)
- **Chuẩn:** `data/fem/shear_fine_swept_normaug.npz` (2400 frame, res-24, side-32, +400 normal).
- Khác: `shear_fine_swept.npz` (2000), `shear_fine_swept_res32.npz` (2000, res-32), `shear_{fine,coarse}_paired.npz`, `traj_mix/`, `geom/{sphere,flat,cylinder}/`.

### 10.3 Kết quả (runs/)
`runs/phase3_fem/{benchmark,vbts_baselines,input_augment,inverse_demo}.json`, `runs/phase4/{policy_servo,probe}.json`, `runs/phase5/sensor_*.json`, `runs/phase6/{env_demo,realism,temporal_compare,object_geometry,loading_history}.json`.

### 10.4 Tái lập nhanh
```bash
.venv-gate2/bin/pip install -e .                              # install package
python -m novbts.operator.fem_benchmark --data data/fem/shear_fine_swept_normaug.npz   # Gate 3
python -m novbts.operator.diff_policy --train-policy --task servo --n-seeds 3           # Phase 4
python -m novbts.sensor.tactile_env --demo                    # Phase 6a env
```
GT FEM sinh trong Docker `isaac-lab-fem` (file ghi ra root-owned → chown qua container; xem `project-structure`).

---

*Báo cáo này tổng hợp từ trạng thái code + kết quả `runs/` + nhật ký nghiên cứu tại thời điểm 2026-06-17. Các báo cáo chi tiết theo phase: `docs/bao_cao_giai_doan3_rq_results.md` (+ PDF), `docs/bao_cao_giai_doan5_sensor.pdf`.*
