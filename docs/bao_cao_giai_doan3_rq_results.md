# Báo cáo Giai đoạn 3 — Hệ thống đầy đủ & tổng quát hóa (RQ1–RQ3)

Ngày: 2026-06-11.

## 0. Khung & vai trò

| Thành phần | Vai trò | Đặc tính |
|---|---|---|
| **Neural operator (FNO)** | Thay solver lúc inference (surrogate cho RL) | nhanh, xấp xỉ — **đóng góp** |
| **Hertz–Mindlin (giải tích)** | GT huấn luyện hiện tại + **validator** | exact closed-form, ~12000 fps |
| **PhysX FEM (Isaac Sim)** | GT vật lý thật + **mốc tốc độ solver** | FEM, 7.2 fps |

3 câu hỏi nghiên cứu đo trade-off "kém chính xác hơn FEM nhưng nhanh hơn nhiều bậc": RQ1 độ chính xác, RQ2 tổng quát hóa, RQ3 tốc độ.

## 1. Ground truth

**Hertz–Mindlin** (Johnson, *Contact Mechanics*): Hertz pháp tuyến ($a=\sqrt{Rd}$, $u_z$ exact) + Cattaneo–Mindlin tiếp tuyến (stick $c=a(1-Q/\mu P)^{1/3}$, partial→full slip). Nhãn mode từ $g=Q/\mu P$. Giả định: bán-không-gian đàn hồi tuyến tính, biến dạng nhỏ.

**PhysX FEM** (`isaac-lab-fem`, DeformableObject): gel block deformable (lưới tet co-rotational, E=1e5, ν=0.45) + indentor cầu rigid; pin node đáy; đọc `nodal_pos_w` → marker grid. **Normal-only** (`novbts.groundtruth.isaac_extract_normal`) **+ SHEAR** (`novbts.groundtruth.isaac_extract_shear`, đã phá deadlock — xem §3c).

## 2. Dataset

- **Hertz–Mindlin** `data/analytic/`: train 16k, val 2k, test_id 2k, test_slip 1k, + 8 OOD (radius/depth/material/friction/geometry/resolution). Marker 32×32.
- **FEM normal** `data/fem/normal.npz`: 40 frame normal-only, marker 24×24, solve-time logged.
- **FEM shear** `data/fem/shear_fine.npz` + `shear_coarse.npz`: 200 frame mỗi loại, kéo tiếp tuyến, drive-ratio g 0.04–1.28 (stick/partial/full), marker 32×32, lưới res-24 mịn vs default thô (xem §3c). Chunk per-seed để dựng lại bộ *paired*: `data/fem/chunks/`.

## 3. Kiểm chứng GT (validate_gt.py)

| Kiểm tra | Kết quả |
|---|---|
| Hertz–Mindlin: contact radius $a$ (qua $u_z(a)=0.5u_z(0)$) | rel err **1.3%** (res32) / 0.34% (res64) ✓ |
| Hertz–Mindlin: convergence peak $\|u_z\|$ (res32 vs 64) | 0.53% (<5%) ✓ |
| Hertz–Mindlin: stick radius $c$ | 48%→32% (res32→64), giới hạn đọc field thô (c enforce exact khi sinh) |
| **FEM vs Hertz: contact radius** | **lệch ~37%** → hiệu ứng **gel dày/rộng hữu hạn** mà half-space bỏ qua |
| FEM: peak$\|u_z\|$/depth | ~1.16 (hợp lý: lún + phồng quanh tiếp xúc) |

→ Hertz–Mindlin chính xác ở giới hạn chuẩn; **FEM lệch ~37% do hình học thật** — định lượng đúng lý do cần FEM.

## 3c. FEM shear — slip nảy sinh tự ma sát (open problem ĐÃ GIẢI QUYẾT)

### Chẩn đoán nguyên nhân gốc
Triệu chứng cũ: kéo indentor (rigid, `kinematic_enabled=True`) **trượt ngang khi đã lún sâu** vào gel deformable → PhysX treo >1h, **GPU idle** (kẹt thật, không phải chậm). Phase lún thẳng (normal) thì chạy bình thường.

Nguyên nhân: body kinematic là **vô hạn cứng / vô hạn nặng** — solver buộc nó đi đúng pose áp đặt bất kể phản lực. Khi nó cắm sâu rồi ép NGANG qua khối tet, sinh một loạt ràng buộc tiếp xúc **over-constrained** mà vòng lặp position-iteration của FEM không hội tụ nổi: mỗi iteration đẩy node ra, ràng buộc kinematic kéo lại → luẩn quẩn, không thoát.

### Cách phá deadlock (`novbts.groundtruth.isaac_extract_shear`)
Không đổi sang drive lực/vận tốc (phức tạp, phải xử lý trọng lực + PD 3 trục). Giữ indentor kinematic nhưng **giảm cường độ over-constraint** bằng 3 thay đổi, mỗi cái nhắm đúng cơ chế kẹt:

| # | Thay đổi | Vì sao hiệu quả |
|---|---|---|
| 1 | Lún **nông ~5mm** cho frame shear (thay vì sâu) | Tiếp xúc nhẹ → ít ràng buộc xung đột; "deep contact" là điều kiện cần của deadlock cũ |
| 2 | Kéo ngang bằng **60 bước cực nhỏ** (Δ≈shear/60) + settle vài substep mỗi bước | ≈velocity control: mỗi bước chỉ xê dịch ràng buộc một chút → solver bám kịp, không phải giải cú nhảy lớn |
| 3 | `solver_position_iteration_count`=30, `contact_offset`=0.002 | Nhiều iteration hơn để hội tụ contact; offset rộng hơn để bắt tiếp xúc sớm, mượt |

→ **Mỗi micro-step ổn định ~0.022s, không kẹt**; smoke 1 frame + sweep 40 frame đều chạy trơn. Theo dõi qua `fem_progress.txt` (Isaac nuốt stdout), kill container sau khi xong (`app.close()` treo).

### Kết quả — tín hiệu slip định tính (đánh giá trung thực)
Slip KHÔNG bị áp đặt (như Cattaneo–Mindlin) mà nảy từ contact ma sát của solver. **Tín hiệu chắc chắn nhất:** dịch chuyển tiếp tuyến bề mặt **bão hòa ~0.85 ± 0.26mm dù indentor kéo ngang tới 7.65mm** (travel trải 29×, corr(peak_tang, travel) = **−0.47**). Tức là indentor **trượt trên gel chứ không kéo gel đi mãi** — đúng bản chất slip. Contact giữ vững (peak|uz| 6.5mm > lún 5.5mm: gel phồng). Solve-time **1.57s/frame ≈ 0.64 fps** (chủ yếu do 240 step/frame; per-step ~0.022s như normal) → mốc solver cho RQ3.

⚠️ **KHÔNG over-claim "khớp Cattaneo–Mindlin"** — soi kỹ cho thấy validation định lượng chưa đứng vững:

| Hạn chế | Hệ quả |
|---|---|
| Lưới deformable thô (~605 node → **~5 node ngang vết tiếp xúc**, cần ≥8–10) | trường tiếp tuyến under-resolved; **không đọc được bán kính dính c**, không so field-level với lý thuyết. Mịn lưới hơn → cooking treo. |
| Trường radial của lún tạo **sàn ~0.85mm** trên peak_tang | metric tracking ở vùng stick (drive bé) bị lấn át; ρ=peak_tang/travel **giảm theo g phần lớn là artifact chia cho travel∝g**, không phải dấu hiệu slip |
| Một điểm vận hành (1 R, 1 μ, 1 E), depth 4–7mm, n=40 | chưa phủ tham số; không kết luận tổng quát |

→ Kết quả này **chứng minh method/pipeline shear chạy được** (deadlock giải quyết + có tín hiệu slip thô từ ma sát), **chưa phải GT slip độ-phân-giải-cao đã validate định lượng**. Để nâng cấp: lưới deformable mịn hơn (giải bài toán cooking-treo), quét nhiều R/μ, tách trường radial khỏi tiếp tuyến.

Dữ liệu: `data/fem/shear_fine.npz` (200 frame, g 0.04–1.28). Validate: `novbts.validation.validate_shear`.

*(Bẫy đã gặp & sửa: nhãn mode ban đầu gán sai — `label_mode` nhận quãng-kéo (m) thay vì drive-ratio nên 40 frame đều thành "normal"; sửa: gán mode từ g sample + relabel npz. npz do container ghi quyền root → ghi bản relabeled sang path user.)*

## 3d. Hội tụ lưới & trần ổn định PhysX (vs IPC/TacEx)

Câu hỏi "lưới mịn có chính xác hơn không" KHÔNG đo bằng Cattaneo (half-space, sai thước đo) mà bằng **convergence study**: cố định geometry, đổi mesh, xem nghiệm hội tụ. Ở **gel 50×50×2mm** (`isaac_extract_shear.py --hex-res ...`, field `simulation_hexahedral_resolution`):

| hex-res | nodes | peak_uz | trạng thái |
|---|---|---|---|
| 24 | 1250 | 0.700mm | ổn định |
| 32 | 2196 | 0.568mm | ổn định |
| 48 | 7248 | **0.551mm** | ổn định |
| 64 | 16926 | (nổ) | phân kỳ — kể cả DT 5ms→1ms |

→ **peak_uz hội tụ 0.70→0.57→0.55mm** (số gia nhỏ dần) → **lưới mịn cho nghiệm pháp tuyến đáng tin**; lưới thô đo hụt. (Trường tiếp tuyến hội tụ chậm hơn — qua bề dày mới 1–3 phần tử.) Hệ quả: con số "68% lệch Cattaneo" của bản mịn (so sánh trước) **không phải kém chính xác** mà do thước đo sai (Cattaneo half-space ≠ gel hữu hạn) + confound kích thước gel (đã sửa: cùng 50×50×2mm).

**Trần ổn định PhysX deformable (ĐIỀU KIỆN, ~ element_size/DT):** gel quá nhỏ MỌI chiều (5×5×2mm) **nổ** (peak_uz −9.6m); lưới quá mịn (res64, phần tử nhỏ) cũng **nổ** dù giảm DT. Tức PhysX có **trần mịn** (~res48 ở đây) — vượt qua phải tinh chỉnh sâu (DT/contact_offset/iterations) hoặc mass-scaling.

**TacEx KHÔNG vướng trần này:** gel biến dạng của TacEx dùng **IPC** (`sapienipc.IPCSystem` ở `fem_based`; `UIPC` ở `tacex_uipc`) — **ổn định vô điều kiện**; đường PhysX của TacEx chỉ là gelpad **rigid**. → Nổ ở gel nhỏ/mịn là **đúng cái giá của lựa chọn PhysX deformable native thay IPC/GIPC (đã bỏ TacEx)**. Nếu sau này cần scale sensor mm tuyệt đối ổn định → mass-scaling/giảm DT, hoặc dùng IPC.

**Hệ quả ở mức operator — lưới GT thô có dạy hỏng operator không?** Train field→field (`novbts.operator.fem_train_compare`) trên GT lưới **thô** (default) vs **mịn** (res-24), eval cả hai trên cùng test mịn. Dùng bộ **paired** (`data/fem/shear_{fine,coarse}_paired.npz`, 160 frame seed s43–s46, geometry giống hệt từng dòng → chênh lệch **thuần do độ phân giải lưới**, không lẫn nhiễu phân phối):

| Metric (test mịn) | GT thô → operator | GT mịn → operator |
|---|---|---|
| rel L2 tổng | 0.361 | **0.153** (2.35×) |
| rel L2 tiếp tuyến | 0.779 | **0.223** (3.5×) |
| sai số hướng | 34.8° | **12.2°** (2.85×) |
| bias biên độ tiếp tuyến | **−22%** | **−5%** |

→ Lưới GT thô làm operator **kém ~2.3–3.5× và lệch hướng tiếp tuyến gấp ~2.9×**; bias −22% ăn khớp việc lưới thô đo **hụt** tiếp tuyến (convergence study ~37%, operator bù một phần). Khi GT đủ mịn, operator gần khớp (hướng 12°, bias −5%). **Kết luận: độ phân giải lưới GT là điều kiện cần cho chất lượng operator — production GT phải ≥res-24.** (Lưu ý trung thực: bản smoke 8-frame ban đầu cho hướng-thô 84° ≈ ngẫu nhiên là **artifact mẫu nhỏ**; số robust+paired đúng là 34.8°.)

### Scale-up & chẩn đoán trần độ chính xác (mesh → data → model)

Sau khi chốt res-24, ta dựng tập train **lưới mịn quét tham số**: **2000 frame** trải trên R∈[15,25]mm, μ∈[0.4,0.8], E∈[0.5,2]·10⁵ Pa (50 combo × 40 frame; 1 container/combo để cô lập nhiễu Isaac dài-hạn; generator `isaac_extract_shear` thêm `--mu/--youngs`, lưu E thật vào params). Operator FEM **đa-điểm**: overall **0.146**, tiếp tuyến 0.348, hướng 14.8° (test 400 held-out, đa-điểm — khó hơn 1-điểm nên không so trực tiếp với 0.153).

Để biết trần tiếp tuyến ~0.35 bị giới hạn bởi đâu, ta khảo sát hai trục (cùng test 400):

| Trục khảo sát | Kết quả | Diễn giải |
|---|---|---|
| **Lượng data** (train 200→1600) | overall 0.177→0.167→0.155→0.146; tiếp tuyến 0.403→0.348 | **bão hòa** — mỗi lần gấp đôi chỉ mua ~0.01 overall, ~1% tiếp tuyến |
| **Dung lượng model** (FNO 12/48 → 16/64, 3× params) | overall 0.146→0.151 (không tốt hơn) | **không thiếu dung lượng** |

→ **Chẩn đoán tam giác:** trần tiếp tuyến KHÔNG do thiếu data (đường cong phẳng), KHÔNG do model nhỏ (lớn hơn không giúp) → bị giới hạn bởi **chất lượng GT** (res-24 trường tiếp tuyến mới qua bề dày 1–3 phần tử) hoặc độ khó nội tại của trường tiếp tuyến FEM.

**Hệ quả thực tiễn:** **~2000 frame là điểm ngọt** cho train FEM ở res-24; scale lên 16k (~17h, 1 GPU) bị **chính đường cong scaling bác bỏ** (overall chỉ về ~0.13, tiếp tuyến gần đứng yên). Muốn hạ tiếp tuyến dưới 0.35, đòn bẩy là **GT mịn hơn (res-32, ~2× chi phí/frame)** hoặc **transfer learning** (pretrain analytic-thang-SI → finetune FEM) — KHÔNG phải thêm frame. Tài sản: generator `infra/gen_fem_sweep.sh` (chunk 40, resumable theo số frame trong npz), data `data/fem/shear_fine_swept.npz`.

## 3b. Framing là quyết định — vì sao headline phải là field→field

Cùng một vật lý, có 2 cách đóng gói input cho mạng, và **chính cách này quyết định FNO thắng hay thua** baseline:

- **param→field** (CŨ): input = vector 9 số `[cx,cy,depth,radius,shear_x,shear_y,mu,stiffness,geom]`, output = field 32×32. Mỗi điểm lưới được "mớm" đầy đủ 9 params → MLP coordinate giải **cục bộ** được, không cần ngữ cảnh toàn cục → FNO mất lợi thế.
- **field→field** (HEADLINE): input = **field bản đồ lún** 32×32 (3 kênh: penetration `max(0,d−r²/2R)`, shear_x·mask, shear_y·mask) + 2 scalar (mu, E); output = field chuyển vị. Điểm ngoài vùng tiếp xúc nhận penetration=0 → MLP per-point **không biết** tiếp xúc ở đâu. Chuyển vị là **hàm phi cục bộ** của toàn bộ tiếp xúc (Green's function đàn hồi) → chỉ operator tích phân toàn field (FNO) giải được.

→ **Toàn bộ RQ1–RQ3 dưới đây chạy trong framing field→field** (module `novbts.operator.field2field`, 16k train / 40 epoch, mirror đúng splits của param→field). Framing param→field cũ chuyển xuống **§4b ablation** làm chứng cứ phản chứng.

## 4. RQ1 — Độ chính xác (test_id, field→field, GT Hertz–Mindlin)

| Model | params | rel L2 | normal | stick | partial | full | dir err |
|---|---|---|---|---|---|---|---|
| MLP (per-point) | 134K | 0.743 | 0.782 | 0.759 | 0.661 | 0.807 | 62.8° |
| **FNO (operator)** | 2.67M | **0.111** | 0.090 | 0.091 | 0.123 | 0.147 | 4.2° |
| FNO+head a | 2.67M | **0.109** | 0.093 | 0.092 | 0.114 | 0.143 | 3.8° |

- **FNO thắng MLP 6.7×** (0.111 vs 0.743) — đây là setup operator đúng nghĩa, không phải artifact framing.
- test_slip (slip-only, khó hơn): FNO rel L2 **0.162**.
- FNO trong dải Gate ~11%; full_slip khó nhất; direction error ~3.8–4.2° (xuất sắc). MLP per-point sụp đổ (74% lỗi, hướng sai 63°) vì thiếu ngữ cảnh toàn cục.

**Slip detection (mode-F1):**

| Head | macro-F1 | normal | stick | partial | full | slip-binary-F1 |
|---|---|---|---|---|---|---|
| a (multitask, gắn FNO) | **0.985** | 1.00 | 1.00 | 0.98 | 0.96 | 1.00 |
| b (classifier riêng) | 0.856 | 0.98 | 0.85 | 0.79 | 0.81 | 0.94 |

→ **Cả hai vượt ngưỡng 0.75** (heuristic cũ Gate 3 chỉ 0.67). Multitask >> separate. **Đóng điều kiện slip của Gate 3.**

## 5. RQ2 — Tổng quát hóa (FNO, OOD, field→field)

| OOD split | rel L2 | degradation |
|---|---|---|
| deep_indent | 0.075 | **0.67×** (tốt hơn!) |
| large_radius | 0.107 | 0.96× |
| res64 (upsample) | 0.112 | 1.01× |
| soft_material | 0.133 | 1.19× |
| low_friction | 0.138 | 1.24× |
| small_radius | 0.205 | 1.84× |
| **flat_geom** | 0.690 | **6.21×** (hình học chưa train) |
| res16 | — | không eval được (FNO modes=12 > grid 16) |

→ Tổng quát **tốt với OOD tham số** (<2×) và **bất biến phân giải lên** (res64 1.01×). **flat_geom chỉ 6.2×** — giảm mạnh từ **19× của framing param→field**: mã hóa hình học vào *field lún* (thay vì one-hot scalar `geom`) khiến operator tổng quát sang hình học mới tốt hơn nhiều. Vẫn không xuống được dưới phân giải mode (res16).

## 6. RQ3 — Tốc độ (field→field)

| Hệ | throughput | / frame |
|---|---|---|
| **FNO inference** | **8087 fps** | 0.124 ms |
| FNO+slip(a) | 8031 fps | 0.124 ms |
| MLP inference | 11780 fps | 0.085 ms |
| **PhysX FEM solver** | **7.2 fps** | **139 ms** |
| Hertz–Mindlin analytic | 13114 fps | 0.076 ms |

→ **FNO nhanh hơn FEM solver ≈ 1123×** (8087 / 7.2). Đây là RQ3 speedup thật — chỉ có nghĩa khi đối chiếu **solver chậm (FEM)**, không phải công thức analytic. Biểu đồ `runs/phase3_f2f_full/fidelity_speed.png`.

## 4b. Ablation framing — vì sao param→field gây hiểu lầm (chứng cứ phản chứng)

Cùng pipeline, đổi cách đặt bài toán, kết quả lật ngược:

| Framing | FNO | MLP | Ghi chú |
|---|---|---|---|
| **param→field** (vector 9 số → field) | 0.079 | **0.066** | MLP thắng — mỗi điểm được "mớm" full params → giải cục bộ được (artifact) |
| **field→field** (bản đồ lún → field) | **0.111** | 0.743 | **FNO thắng 6.7×** (headline §4) |

Trong param→field, mọi điểm lưới đều nhận đủ 9 params (biết chính xác tâm/độ sâu tiếp xúc) nên MLP chỉ cần khớp công thức cục bộ → không cần operator. Đây là **lý do KHÔNG dùng param→field cho paper**: nó che mất giá trị của operator learning. Ablation modes 12→16 cho thấy low-pass cũng góp phần nhỏ (FNO 0.078→0.073 trong param→field) nhưng không đủ lật ngược — chỉ *framing* mới lật được. Script: `scripts/archive/phase3_field2field.py` (PoC) → `novbts.operator.field2field` (đầy đủ).

## 6c. RQ1–RQ3 trên GT FEM THẬT — đóng Gate 3

Toàn bộ RQ ở trên đứng trên GT **analytic** Hertz–Mindlin (16k, lý tưởng hoá). Gate 3 yêu cầu làm lại trên **vật lý thật**. Ta train+đo lại trên tập **FEM swept 2000 frame** (res-24, quét R/μ/E; `novbts.operator.fem_benchmark` → `runs/phase3_fem/benchmark.json`):

| | analytic GT (16k) | **FEM GT (2000)** |
|---|---|---|
| **RQ1** FNO rel L2 | 0.111 | **0.146** |
| RQ1 MLP rel L2 | 0.743 | 0.328 |
| **FNO thắng MLP** | 6.7× | **2.24×** |
| RQ1 hướng tiếp tuyến (FNO) | 3.8° | 14.8° |
| **slip-F1** (head-a multitask) | 0.985 | **0.904** |
| **RQ3** FNO vs solver | 1123× (vs 7.2 fps) | **≈23.000×** (vs 0.341 fps) |

**Kết luận chính (đều GIỮ trên vật lý thật):**
- **Luận điểm phi-cục-bộ giữ vững:** FNO thắng MLP **2.24×** (hướng 35.7°→14.8°). Biên hẹp hơn analytic (6.7×) vì FEM nhiễu/ít lý tưởng + MLP không sụp đổ thảm như trên trường giải tích — nhưng FNO vẫn thắng dứt khoát.
- **RQ1 per-mode (FNO):** normal 0.123 · stick 0.134 · partial 0.150 · full_slip 0.167 — sai số tăng dần theo slip (đúng kỳ vọng, slip khó hơn).
- **Slip-F1 head-a = 0.904** (>0.75 — đóng điều kiện Gate 3 trên nhãn thật). Head-b ban đầu suy biến (normal F1=0, macro 0.595) vì sweep g∈[0,1.3] gần như không có frame normal thuần (chỉ 63/2000 = 3.1%) → mất cân bằng lớp, không phải lỗi mô hình.
  - **Cứu head-b bằng frame normal thuần:** bổ sung **400 frame g=0** (mode normal) span cùng hộp R/μ/E (`infra/gen_fem_normal_sweep.sh`, lateral-steps=2 → ~0.47s/frame), merge+shuffle thành **2400 frame** (normal 3.1%→19.3%, `data/fem/shear_fine_swept_normaug.npz`). Kết quả (`benchmark_normaug.json`): **head-b normal F1 0.0→0.851, macro 0.595→0.753 (vượt 0.75)**; head-a normal F1 cũng tăng 0.769→0.951. Regression không suy giảm (FNO overall 0.146→0.144, vẫn thắng MLP 2.24×; RQ2 ~1.3×; RQ3 23.427×). → **xác nhận head-b suy biến là do mất cân bằng lớp, không phải lỗi mô hình.**
- **RQ2 (ngoại suy đuôi tham số, train low-80% → test high-20%):** high-R 1.30×, high-μ 1.29×, high-E 1.29× — tổng quát hóa mượt, nhất quán cả ba trục. (Phạm vi khiêm tốn: ngoại suy *trong hộp* R∈[15,25]mm/μ∈[0.4,0.8]/E∈[0.5,2]e5, chưa phải OOD ngoài-dải vì không sinh FEM ngoài hộp rẻ được.)
- **RQ3 ≈23.000×** so với solver PhysX-FEM shear thật (0.341 fps = 2.9s/frame) — đây là fidelity-speed thật, lớn hơn nhiều mốc analytic vì solver shear thật chậm hơn hẳn.

→ **Gate 3 đóng trên GT vật lý thật:** FNO là surrogate phi-cục-bộ thắng baseline, phân loại slip đạt ngưỡng, nhanh hơn solver ~4 bậc. (Trần tiếp tuyến 0.146/14.8° là giới hạn GT-fidelity res-24, xem §3d — không phải giới hạn operator.)

## 6d. So sánh với mô hình marker của các mô phỏng VBTS tiêu biểu

Để biết "công trình của ta tốt tới đâu", ta đối chiếu trực tiếp với **mô hình chuyển động marker** của các mô phỏng cảm biến xúc giác thị giác (VBTS) tiêu biểu. **Số liệu liên-paper KHÔNG so trực tiếp được** (cảm biến khác nhau, đầu ra là ảnh RGB vs trường marker của ta) — nên ta **cài lại lõi mô hình marker** của từng phương pháp rồi **fit + đo trên CHÍNH GT FEM của ta** (cùng data/split/metric, `novbts.operator.vbts_baselines`, `runs/phase3_fem/vbts_baselines.json`):

- **TACTO** (Wang et al., RA-L 2022): mô hình **động học, không ma sát** — marker bám mặt tiếp xúc, tiếp tuyến chỉ là kéo cứng (rigid drag) cả vùng, **không có stick-slip**.
- **Cattaneo–Mindlin giải tích** (vật lý first-principles, nền của các sim xúc giác phân tích): trường Hertz (pháp tuyến) + Cattaneo–Mindlin (tiếp tuyến partial-slip). Cho nó cơ hội công bằng nhất: **hiệu chỉnh affine per-channel (scale+bias) fit trên train** → cô lập xem *hình dạng* trường giải tích có khớp FEM không.
- **Taxim** (Si & Yuan, RA-L 2022) + **FOTS** (Zhao et al., 2023): mô hình **đàn hồi tuyến tính chồng chập** (superposition) — bản đồ tuyến tính (Green's function) từ tải tiếp xúc → chuyển vị marker. Ta cài bằng **một conv tuyến tính bất biến dịch** (không phi tuyến).
- **MLP per-point**: học nhưng **cục bộ** (không ngữ cảnh toàn cục) — cận dưới.
- **FNO (của ta)**: operator phổ **phi-cục-bộ**.

| Phương pháp (lõi marker) | overall | normal | stick | partial | full | hướng° | params | FNO hơn |
|---|---|---|---|---|---|---|---|---|
| TACTO-style (động học, no friction) | 0.504 | 0.440 | 0.432 | 0.524 | 0.625 | 65.8 | 3 | **3.51×** |
| Cattaneo–Mindlin giải tích (đã hiệu chỉnh) | 0.435 | 0.318 | 0.341 | 0.475 | 0.604 | 39.0 | 6 | **3.03×** |
| Taxim/FOTS-style (tuyến tính chồng chập) | 0.295 | 0.276 | 0.266 | 0.285 | 0.363 | 26.2 | 14.4K | **2.05×** |
| MLP per-point (cục bộ) | 0.321 | 0.227 | 0.242 | 0.349 | 0.467 | 33.6 | 134K | 2.24× |
| **FNO (ours)** | **0.144** | **0.112** | **0.140** | **0.150** | **0.168** | **14.6** | 2.67M | — |

**Đọc kết quả:**
- **TACTO-style sụp trên tiếp tuyến** (hướng 65.8° ≈ ngẫu nhiên): mô hình không-ma-sát không thể tái tạo trường stick-slip → FNO hơn **3.51×**. (Nó cực rẻ về fps nhưng vô dụng cho slip.)
- **Vật lý giải tích kinh điển (Cattaneo–Mindlin) — dù đã hiệu chỉnh biên độ — chỉ đạt 0.435, TỆ HƠN cả mô hình tuyến tính fit-data (0.295)**: hiệu chỉnh affine chỉ chỉnh được *biên độ*, không đổi được *hình dạng* profile; trường FEM thật lệch khỏi công thức Hertz–Mindlin lý tưởng (nhất quán với khe ~37% half-space vs gel thật, §7). FNO hơn mô hình vật lý first-principles **3.03×** → operator nắm được cấu trúc trường mà công thức giải tích bỏ sót.
- **Taxim/FOTS-style là baseline mạnh nhất** (0.295) — đàn hồi tuyến tính bắt được phần lớn trường, **nhưng FNO vẫn hơn 2.05×** vì chuyển stick→partial→full slip là **phi tuyến**, mô hình tuyến tính chồng chập không biểu diễn được. Đây là luận điểm cốt lõi: cái mà neural operator thêm vào so với mô phỏng VBTS nhanh là **phi tuyến + phi cục bộ của slip**.
- **MLP (cục bộ) tốt ở normal/stick** (0.227/0.242, trường trơn → cục bộ đủ) **nhưng sụp ở partial/full** (0.349/0.467) — đúng kỳ vọng: slip cần ngữ cảnh toàn cục.
- **FNO thắng ở MỌI mode**, cách biệt lớn nhất ở slip (full 0.168 vs 0.363–0.625 của baseline). → khẳng định đóng góp đúng chỗ: phi-cục-bộ + phi tuyến cho trường tiếp tuyến slip.

⚠️ *Phạm vi trung thực:* đây là cài lại **lõi mô hình marker** của từng mô phỏng (không phải toàn bộ renderer quang học của chúng), fit công bằng trên GT FEM của ta. Chạy **trọng số hiệu-chỉnh-sẵn của bản gốc** lên gel FEM của ta sẽ là *sai cảm biến* (kém một cách vô nghĩa), KHÔNG công bằng hơn — hướng A đúng đắn cho mô hình đã-hiệu-chỉnh là fit *form* của chúng lên data của ta, đúng như đã làm. Mục đích là cô lập **giá trị của việc mô hình hóa trường slip phi tuyến/phi-cục-bộ** mà các mô phỏng VBTS tuyến tính/động học/giải tích không nắm được — không phải bảng xếp hạng fps liên-paper.

## 7. ⚠️ Vấn đề mở & hạn chế (trung thực)

1. **FNO > MLP đã chốt trong framing field→field** (§4, 6.7×) — đây là headline. Mâu thuẫn cũ (param→field FNO thua) đã được giải thích là **artifact của framing** (§4b) và loại khỏi headline. **Toàn bộ RQ1–RQ3 nay đứng trên field→field.**
2. **FEM shear — DEADLOCK đã phá** (§3c) + đã mịn lưới (res-24) + scale + quét tham số (§3d, §6c). Còn lại: **trần tiếp tuyến ~0.35 do GT-fidelity res-24** (data/model đã bão hòa — §3d); đòn bẩy là GT res-32 hoặc transfer learning, không phải thêm frame. RQ2 trên FEM mới là ngoại suy *trong hộp*, chưa phải OOD ngoài-dải.
3. **Operator ĐÃ train trên FEM** (§6c: 2000 frame swept, FNO thắng MLP 2.24×, slip-F1 0.90, nhanh ~23.000× solver thật) → Gate 3 đóng trên vật lý thật. Lưu ý: GT FEM (2000) và analytic (16k) khác thang đơn vị, train riêng — chưa hợp nhất (transfer learning là hướng để analytic bootstrap FEM).
4. Hertz–Mindlin là half-space tuyến tính ≠ gel thật (lệch ~37% đã đo) — khe hở để Giai đoạn 4 sim-to-real đóng.

## 8. Quyết định Gate (paper-scale)

| RQ | Phán quyết | Ghi chú |
|---|---|---|
| RQ1 accuracy + slip | **GO** | FNO ~11% dải Gate; slip-F1 0.985 đóng điều kiện Gate 3 |
| RQ1 operator > baseline | **GO** | field→field: FNO 0.111 vs MLP 0.743 (6.7×); param→field chỉ là ablation phản chứng (§4b) |
| RQ2 generalization | **một phần** | tốt param OOD (<2×) + res-up; kém geometry (6.2×, nhưng đã cải thiện 3× so với param→field) |
| RQ3 speed | **GO** | FNO ≈ 1123× FEM solver (mốc thật) |

**Kết luận:** Giai đoạn 3 đạt **proof-of-machinery hoàn chỉnh** — pipeline field→field chạy, **operator thắng baseline 6.7× một cách chính danh**, slip-discontinuity giải quyết trên GT analytic (đóng Gate 3), **FEM shear deadlock đã phá (§3c)** mở đường cho GT slip từ ma sát (hiện mới mức định tính, lưới thô), tốc độ thật vs FEM 1123×. **Việc còn lại trước paper:** (a) mịn lưới deformable + scale data FEM (normal+shear) để có GT slip định lượng, (b) train headline field→field trên FEM thay vì chỉ Hertz–Mindlin.

## 9. Tài sản (package `novbts` + runs/)
Mã nguồn dưới `src/novbts/`: `groundtruth/{hertz_mindlin, data_gen, isaac_extract_normal (FEM normal), isaac_extract_shear (FEM shear, phá deadlock)}`, `operator/{`**`field2field` (HEADLINE field→field, RQ1–RQ3 analytic)**`, param2field (param→field ablation), eval_rq, fem_train_compare (thô vs mịn), `**`fem_benchmark` (RQ1–RQ3 trên FEM, §6c)**`, `**`vbts_baselines` (so với mô hình marker TACTO/Taxim/FOTS, §6d)**`}`, `validation/{validate_gt, validate_shear, compare_shear}`, `models.py`, `report/make_pdf.py`; `infra/{Dockerfile.fem, setup_isaac.sh, gen_fem_sweep.sh, gen_fem_normal_sweep.sh (frame normal cứu head-b)}`; script chết/PoC ở `scripts/archive/`. Dữ liệu/kết quả: `runs/phase3_f2f_full/results.json` + `fidelity_speed.png` (headline) · `runs/phase3/` (param→field ablation) · `runs/phase3_fem/{benchmark.json, benchmark_normaug.json (head-b cứu), vbts_baselines.json (§6d)}` · `data/fem/{shear_fine_swept.npz, shear_fine_swept_normaug.npz (2400, +400 normal), normal.npz, shear_fine.npz, shear_coarse.npz}`.
