# Báo cáo phản biện — KSE 2026

**Paper:** *A Differentiable Neural-Operator Surrogate for Vision-Based Tactile Sensing and Control*
**Venue:** KSE 2026, special session "Fusion of Embodied AI and Soft Robotics" (single-blind, 6 trang IEEE, deadline 15/07/2026)
**Nguồn:** tổng hợp từ workflow phản biện 6 góc nhìn (novelty / rigor / soundness / validity / reproducibility / presentation), mỗi finding được kiểm chứng đối kháng trên paper + dữ liệu `runs/` + source `src/novbts/`. 35 finding → 19 major, 14 minor, 1 bị bác, 0 critical.

---

## 0. Phán quyết tổng thể

| | |
|---|---|
| **Khuyến nghị (bản hiện tại, *as submitted*)** | **Reject / Major revision** — KHÔNG phải vì khoa học, mà vì PDF còn dấu hiệu "chưa hoàn thiện" (TODO đỏ, hình lỗi, tác giả placeholder, bib `Anonymous`) đủ để một referee gạch ngay. |
| **Khuyến nghị (sau khi sửa Tier A + B)** | **Weak Accept** — đóng góp thật, trung thực bất thường, grounded trên FEM thật; phù hợp một venue khu vực vững như KSE. |
| **Độ tự tin** | Cao về dữ liệu (mọi số đều đối chiếu được với `runs/`), trung bình về novelty (xem §3). |

**Một câu cho tác giả:** *Phần khoa học đủ để được nhận ở KSE; cái đang chắn giữa bạn và "accept" gần như toàn bộ là việc dọn dẹp bản nháp (Tier A) và hạ giọng vài tuyên bố quá lời (Tier B) — đều sửa được trước deadline mà không cần chạy lại thí nghiệm.*

---

## 1. Điểm mạnh (cần giữ và làm nổi)

1. **Trung thực bất thường.** Có hẳn §VII "Limitations and Honest Scope" thừa nhận trần tiếp tuyến, biên hẹp so với U-Net, tốc độ không phải lợi thế throughput. Đây là tài sản ở venue single-blind — đừng bỏ.
2. **Grounded trên vật lý thật.** Mọi số headline khớp với file trong `runs/` (đã kiểm: Bảng I khớp `vbts_baselines.json`; Bảng II khớp `policy_servo.json`). Baseline được fit trên *cùng* FEM GT.
3. **Góc differentiability + control là điểm bán thật.** `render∘FNO` khả vi end-to-end, gradcheck pass (rel-err 3.7%, `env_demo.json`), khôi phục được action từ ảnh.
4. **Eq.1 và cài đặt FNO đúng chuẩn.** `SpectralConv2d` trong `models.py` là Fourier layer 2-D chuẩn (rfft2 → nhân trọng số mode thấp → irfft2), khớp Li et al. 2021. Tham số 2.67M tái dựng được từ `FNOField` (width=48, modes=12, 4 layer).

---

## 2. Điểm yếu — xếp theo mức đe doạ chấp nhận

### TIER A — BẮT BUỘC sửa trước khi nộp (dấu hiệu "chưa hoàn thiện", nhưng sửa trong vài phút)

**A1. `\todo` đỏ còn render trong PDF + tác giả placeholder.** *(major, nhiều lens xác nhận — ứng viên "chí mạng nhất")*
- `main.tex:19` định nghĩa `\todo` → chữ đỏ `[TODO:...]`. Còn **2 chỗ render**: footnote RQ5 (`:379-381`) thừa nhận số image-inverse "still being finalised", và Acknowledgment (`:431`).
- Tác giả là `Author One/Author Two` (`:27-30`); Acknowledgment trống.
- **Vì sao nguy:** referee đọc thẳng trong PDF dòng tác giả tự nhận "số này chưa kiểm chứng" → tín hiệu tự hủy. Single-blind cho phép để tên thật.
- **Sửa:** điền `\author{}` + Acknowledgment; xử lý footnote (xem A4); **xoá macro `\todo` ở dòng 19** rồi `grep -n "todo\|TODO" main.tex` xác nhận sạch để không lọt chữ đỏ khi recompile.

**A2. Hình 2 (`fidelity_speed.png`) là hình CŨ, mâu thuẫn với Bảng I.** *(major/valid)*
- Hình hiện tại: trục y "relative L2" = 0.066–0.082, chỉ 3 điểm, **MLP là điểm CHÍNH XÁC NHẤT** — ngược hẳn Bảng I (MLP 0.321, tệ hơn FNO 2.24×). Tiêu đề còn ghi "Phase 3 fidelity-speed trade-off". Trục x chỉ 8e3–1e4 fps → không thấy FEM 0.34 fps lẫn "23,000×".
- **Gốc rễ:** hình bị copy từ pha analytic-GT cũ (`eval_rq.py:195` lưu vào `runs/phase3/`); các số là `rq_results.json` cũ, không phải bake-off FEM.
- **Vì sao nguy:** referee mở hình → tưởng số bị tráo/đảo trên chính tuyên bố độ chính xác trung tâm = nghi vấn data-integrity.
- **Sửa (chọn 1):** (a) **Xoá hình**, thay bằng 1 câu trong RQ3 (7917 fps / 0.125 ms vs FEM 0.34 fps); hoặc (b) vẽ lại đúng từ số FEM (y = rel-L2 0.14–0.50 với tất cả baseline Bảng I, x = log latency đủ rộng để thấy cả FEM 0.34 fps và operator ~8000 fps), bỏ tiêu đề "Phase 3". **Tuyệt đối không nộp PNG hiện tại.**

**A3. Bibliography lỗi: Taccel = `Anonymous and others`.** *(major/valid)*
- `references.bib:124-130` Taccel có `author={Anonymous and others}`, `journal={arXiv preprint}` → render ra chuỗi "Anonymous" trong bài single-blind. FOTS (`zhao2023fots`) thiếu tác giả/volume/pages, key ghi 2023 nhưng `year=2024`. DiffTactile thì ĐÚNG (ICLR 2024) — chỉ cần bỏ `% VERIFY`.
- **Sửa:** điền tác giả/venue thật cho Taccel + FOTS (xác minh arXiv id với nguồn gốc, đừng tin số tôi đoán); bỏ 3 comment `% VERIFY` và header note. Dòng `main.tex:123` "TacIPC and Taccel" gộp 2 công trình khác nhau dưới 1 cite — tách hoặc bỏ "TacIPC and".

**A4. Số image-inverse: 2.1%/1.1° không có trên đĩa; chỉ 1 frame 10.6%/1.5°.** *(major, 3 lens)*
- `sensor_inverse.json` = **1 frame** (2031, full_slip): magnitude **10.6%**, direction 1.52°. Số 2.1%/1.1° chỉ nằm trong report tiếng Việt, không file nào tái dựng được. Số 2.3% là **raw-field** (`inverse_demo.json`, task dễ hơn — không qua renderer), không phải image-inverse.
- **Sửa:** bỏ footnote `\todo`. Hoặc chạy sweep nhiều frame rồi báo mean±std (magnitude + direction, cho cả raw-field và image); hoặc gắn nhãn rõ "1 frame full_slip đại diện", báo magnitude ảnh **10.6%** trung thực, **tách bạch raw-field 2.3% vs image 10.6%** ở cả abstract lẫn RQ5. Đừng để 2.1% trong body.

---

### TIER B — overclaim một referee kỹ tính sẽ bắt (sửa bằng câu chữ, không cần chạy lại)

**B1. "Irreducible tangential ceiling" — quá lời trong chính mục tên là "Honest Scope".** *(major, 4 lens — đây là điểm khoa học đáng lưu ý nhất)*
- Sắc thái quan trọng (đã tự kiểm chứng): bằng chứng *self-consistent* res-24→res-32 (overall **0.146→0.158**, hơi tệ hơn) **ỦNG HỘ** câu "làm mịn lưới không hạ được trần" — số này phòng thủ được.
- NHƯNG (a) từ "irreducible" + "finer FEM meshes ... all negative" là quá mạnh, vì GT tiếp tuyến res-24 **chưa hội tụ** (~78% lệch vs res-32) — điều báo cáo nội bộ của bạn thừa nhận nhưng **paper giấu**; (b) `compare_paired.json` (tang 0.779 coarse vs 0.223 fine) là **ablation chất lượng GT huấn luyện** (train coarse vs train fine, cùng test trên fine GT), **KHÔNG phải** sweep hội tụ lưới — đừng dùng nó làm bằng chứng cho cả hai chiều; (c) nhập nhằng số: tiếp tuyến riêng ≈ 0.22, còn 0.144/14.6° là **overall**.
- **Sửa:** thay "irreducible" → "một trần tiếp tuyến *robust với mịn hoá lưới tới res-32 ở chế độ fit self-consistent, nhưng mức tuyệt đối bị giới hạn bởi độ trung thực của ground-truth*"; **công bố** caveat ~78% non-convergence; phân biệt rõ hai thí nghiệm (self-consistent benchmark vs paired training-quality). Tuỳ chọn mạnh nhất: chạy 1 benchmark self-consistent res-40+ để biến "robust tới res-32" thành "đã hội tụ".

**B2. "Statistically indistinguishable" thực ra phân biệt được về mặt thống kê.** *(major/valid)*
- `policy_servo.json`: autograd 8.018e-8 (std 9.1e-10, n=3) vs ES 8.358e-8 (std 7.86e-10). t≈4.9, **p<0.01** → autograd *đáng tin là tốt hơn nhẹ*, không "indistinguishable". Và **không số regression nào có variance** (single seed `torch.manual_seed(0)`).
- **Sửa:** Bảng II thêm mean±std mọi số; đổi câu thành "autograd đạt loss bằng-hoặc-thấp-hơn nhẹ (8.02 vs 8.36e-8; chênh nhỏ nhưng nhất quán) với 64× ít query và 24× nhanh hơn"; caption đổi "Equal accuracy" → "comparable accuracy".

**B3. U-Net hoà FNO → "global spectral support là quyết định" bị chính dữ liệu của bạn bác.** *(major, validity+novelty)*
- U-Net (0.148, **0.47M param, dir-err 13.4° còn THẤP hơn** FNO 14.6°) hoà FNO (0.144, 2.67M). Vậy cái quyết định là **field→field + receptive field dày**, KHÔNG phải tính toàn cục của Fourier. Body đã thừa nhận (`:270-274`) nhưng title/abstract/contribution-1 vẫn đẩy FNO/spectral.
- **Sửa:** đổi tên contribution 1 → *"một kết quả thực nghiệm về framing field→field: mô hình dày, phi-cục-bộ thắng per-point/analytic/linear trên FEM thật"*; hạ "global support is the crux" (`:217`). **Giữ** số FNO-vs-MLP 2.24× và 2.05–3.51× (thật và lớn) — chỉ hạ diễn giải "spectral globality là cơ chế".
- *Phụ:* con số "MLP collapse 0.74/63°" (`:220`) đến từ `runs/phase3_f2f_full/results.json` — pha analytic-GT 16k frame, KHÁC dataset của Bảng I; phải gắn nhãn là ablation trên GT analytic, hoặc chạy lại trên cùng 2000-frame FEM.

**B4. Bake-off single-seed (n=1) cho toàn bộ regression.** *(major)*
- Thứ hạng FNO 0.144 vs U-Net 0.148 đặt trên n=1. Không thể nói "FNO holds".
- **Sửa:** chạy ≥3 seed cho top contenders (FNO/U-Net/DeepONet/Galerkin/MLP) — hạ tầng đã có sẵn từ thí nghiệm control; hoặc tối thiểu ghi rõ n=1 trong Setup + caption và nói FNO/U-Net không phân biệt được ở khe này.

**B5. "Differentiable control" bán quá mức một bài toán nghịch đảo tĩnh 2-D.** *(major, soundness)*
- Code `diff_policy.py`: action [B,2], vào FNO tuyến tính qua `action*mask`, loss `||FNO(action)-y*||²`, **1 forward/step, không rollout, không feedback**. Autograd thắng ES 64× trên mục tiêu trơn 2-D là kết quả *textbook*, không phải phát hiện mạnh. Suh-2022 "pathology không cắn" gần như **tautology**: bạn vi phân một surrogate *trơn* (chính bạn nói nó low-pass cạnh stick→slip), nên đương nhiên gradient sạch — đây là *né* Suh, không phải *bác* Suh. Probe `grad_variance_probe` lấy autograd LÀM tham chiếu (es_bias_rel ~250%) → không chứng minh được autograd "sạch".
- **Sửa:** đổi "Differentiable control"/RQ4 → "amortized action inversion through a frozen operator"; nói rõ action 2-D, vào input tuyến tính, kết quả là *nghịch đảo nhanh hàm trơn thấp chiều*; dời Limitation-4 lên trước số 64×; viết lại câu Suh thành "né bằng thiết kế, không bác"; **bỏ "we test this directly"**. Nếu muốn thật sự test Suh: so autograd-qua-FNO với finite-difference-qua-FEM thật.

**B6. Related Work thiếu làn sóng tactile-sim khả vi/GPU 2024-2025 — đối thủ thật.** *(major/valid)*
- Thiếu **TacSL** (NVIDIA 2024, IEEE), **TacEx** (2025, *cùng nền Isaac Sim của bạn*, đề xuất FEM+marker khả vi cho policy-gradient), **MIT Xu/Kim** "Efficient Tactile Simulation with Differentiability". Chúng chia sẻ đúng điểm bán của bạn (diff + control + GPU). Referee special-session sẽ biết TacEx.
- **Sửa:** thêm 2-3 câu ở §II; **sắc hoá khác biệt**: chúng lấy gradient qua physics-engine/force model và *vẫn giữ solver trong vòng lặp*, còn bạn đóng băng **một operator học sẵn LÀ bản đồ khả vi** (mesh-free, sub-ms, không solver). Xác minh bib trước khi thêm.

**B7. Fit special-session chỉ tuyên bố 1 câu, chưa "earned".** *(major/valid)*
- Đọc như paper PDE-surrogate/ML: phần embodied/soft bị dồn cuối (RQ4/RQ5), câu fit duy nhất `:86` là khẳng định suông; payload robot yếu theo chính limitation (single-step, no hardware, one-shot).
- **Sửa (không chạy lại):** mở abstract bằng *câu chuyện cảm-biến-mềm-khả-vi-trong-vòng-lặp* (khôi phục shear từ ảnh, env đóng 87% gap), đặt kết quả operator làm *bộ kích hoạt*; đảo contribution để diff-sensor + control lên đầu; thay câu `:86` bằng 1 use-case manipulation cụ thể per keyword (vd shear-from-image cho grasping nhận biết trượt). Đừng overcorrect thành "closed-loop manipulation" mà bạn chưa có.

---

### TIER C — Reproducibility (major, nhưng chuẩn mực)

**C1. Không có Data/Code Availability, không hyperparameter, không spec compute.** *(major/valid)*
- Thiếu: link repo; FNO modes/width/layers, LR/optimizer/epochs, trọng số loss multitask, ES sigma (`policy_servo.json` có sigma=0.02/pop=32); GPU model+VRAM ("single RTX-class GPU" không audit được 8000 fps/23,000×).
- **Sửa:** thêm "Data and Code Availability" (link repo ẩn-danh-cho-review — `runs/` + `src/novbts/` đã có sẵn, công sức thấp/uy tín cao) + đoạn "Implementation Details" rút từ JSON/code. Gắn 1 qualifier vào "2.05–3.51×" ở abstract (`:46`)/contribution (`:95`): "over our refits of each method's marker-motion core".

---

### TIER D & E — Polish (minor)

- **D1.** "23,000×" = throughput GPU batched ÷ latency CPU đơn luồng (không apples-to-apples). Nhưng paper đã hedge ở 4 chỗ và không load-bearing. Sửa: báo latency batch=1, trình bày dạng "~10³× thấp hơn về latency"; sửa caption Fig.2 "four orders of magnitude lower latency".
- **E1.** Kiến trúc chưa nêu cụ thể (Eq.1 & cài đặt đúng, nhưng layers=4/width=48/modes=12 + kênh scalar μ,E không ghi). Thêm 1 câu "Implementation details" để 2.67M tái dựng được.
- **E2.** RQ2 "generalization" thực ra là upper-tail trong-bao (in-envelope), không phải OOD thật; câu flat-punch OOD (`:308-310`) **không có số** nào trên đĩa. Đổi nhãn thành "sensitivity to held-out upper quantiles"; thêm số FNO trên flat-punch hoặc hạ thành quan sát định tính.
- **E3.** README number→source map trỏ **sai file** cho dòng sensor faithful/round-trip (phải là `sensor_build.json`, không phải `sensor_{compare,compat}.json`). "1.9px" là **slip-subset**; overall là **1.7px** (`overall_px=1.712`). cos 0.973 → in đúng 0.975.
- **E4.** Fig.5 lưới 8×4 với tiêu đề "MSE=" không đọc được ở 0.95 cột → gom còn 2×3 (1 cặp GT-vs-FNO mỗi mode), bỏ tiêu đề per-cell, đưa MSE vào caption.
- **E5.** "(Table is implicit)" (`:146`) là chữ nháp sót → xoá. Cột "adv. ×" (`:287`) chưa giải nghĩa → gloss trong caption.
- **E6.** Abstract over-hedge (mở/đóng đều bằng limitation); "honest scope" lặp 3 lần. Đổi contribution-4 từ "An honest scope" → trình bày **kết quả dương**: *"một cơ chế duy nhất vừa giúp vừa giới hạn"* (chính spectral smoothing cho gradient sạch cũng là cái chặn độ chính xác tiếp tuyến) — nâng insight bị chôn ở §VII lên thành đóng góp.
- **E7.** Báo `flow_rel_l2 ≈ 0.25` cạnh cos 0.973 (cosine bỏ qua biên độ) để không bị reviewer mở `sensor_compat.json` bắt.

---

## 3. Novelty — phán quyết

**Lõi novelty ĐỨNG VỮNG.** Đã tìm: neural-operator cho cơ học vật rắn tồn tại (Geo-FNO 2207.05209; FNO mã hoá vật lý cho trường ứng suất 2408.15408), nhưng **chưa ai** dùng cho trường marker-displacement của VBTS; ngược lại tactile-sim (DiffTactile/FOTS/Taccel/TacEx) đều physics-engine hoặc example/MLP, **không phải operator-learning**. Tổ hợp "học operator nghiệm FEM thành FNO đóng băng + khả vi làm surrogate VBTS" là **chưa được claim tới 2026**. Đây là đóng góp thật, vừa tầm KSE.

**Nhưng** "non-locality argument" (contribution 1) phần lớn là *phát biểu lại* lý do quen thuộc operator > MLP, không phải insight mới riêng cho tactile (xem B3). Và gap với prior art mới chỉ *khẳng định*, chưa *chứng minh bằng citation* (xem B6). Cần chuyển từ asserted-gap → demonstrated-gap.

---

## 4. Câu hỏi cho tác giả (chuẩn bị rebuttal)

1. Trần tiếp tuyến: bạn có số benchmark self-consistent res-40+ không? Nếu chưa, trên cơ sở nào khẳng định nó "irreducible" thay vì "GT-fidelity-limited ở res-24/32"?
2. Variance: tất cả số regression là single-seed phải không? Cho biết mean±std (≥3 seed) của FNO vs U-Net để chứng minh thứ hạng.
3. U-Net hoà FNO với 5.7× ít param và dir-err thấp hơn — vậy "global spectral support" đóng góp gì *vượt trên* một receptive field dày?
4. "Control": có thí nghiệm nào multi-step/closed-loop không, hay toàn bộ là nghịch đảo 1-bước? Nếu chỉ 1-bước, kết quả 64× nói gì *riêng về control* mà không phải về "gradient của hàm trơn"?
5. Image-inverse: số aggregate đa-frame thật là bao nhiêu (magnitude + direction)? Frame đơn cho 10.6% magnitude.
6. Vì sao Fig.2 mâu thuẫn Bảng I (MLP là điểm chính xác nhất)?

---

## 5. Điểm "chí mạng" nhất & devil's advocate

**Chí mạng nhất (đe doạ reject thực tế):** không phải một lỗi khoa học, mà là **tổ hợp tín hiệu "bản nháp chưa xong"** trong PDF nộp — `\todo` đỏ tự nhận số chưa kiểm chứng (A1) + Fig.2 cũ mâu thuẫn tuyên bố trung tâm, tạo ảo giác data-integrity (A2) + bib `Anonymous` (A3). Một referee gạch bài *trước khi* kịp đánh giá khoa học. **May mắn: cả ba sửa trong < 1 giờ.**

**Đe doạ khoa học lớn nhất:** B3 (U-Net hoà FNO) + B4 (single-seed) cộng hưởng — chúng làm rỗng cái khung "FNO/spectral là quyết định" mà title đang đứng trên đó.

**Devil's advocate (lời bào chữa mạnh nhất, trung thực, cho tác giả):** *"Title chúng tôi nói 'neural-operator surrogate', không nói 'FNO là duy nhất tốt nhất'; body (§V-A) đã tự nói thông điệp phòng thủ là 'dense non-local operator learning thắng physics/linear models'. U-Net cũng là một operator phi-cục-bộ học field→field — nó *củng cố* luận điểm field→field chứ không bác. Đóng góp thật là **framing field→field + differentiability**, không phải kiến trúc cụ thể."* — Nếu tác giả viết đúng giọng này vào contribution 1 + abstract, B3 chuyển từ điểm yếu thành điểm mạnh.

---

## 6. Checklist trước 15/07 (xếp theo impact/effort)

| Ưu tiên | Việc | Impact | Effort |
|---|---|---|---|
| 1 | A1 — điền tác giả/ack, xoá macro `\todo`, grep sạch | Cao | Thấp |
| 2 | A2 — xoá/vẽ lại Fig.2 | Cao | Thấp |
| 3 | A3 + A4 — sửa bib Taccel/FOTS; xử lý số image-inverse | Cao | Thấp |
| 4 | B1 — hạ giọng "irreducible", công bố non-convergence 78% | Cao | Thấp |
| 5 | B3 + B5 + B7 — đổi tên contribution 1 & 4, hạ giọng "control", reframe abstract về embodied/soft | Cao | Trung |
| 6 | B6 — thêm TacSL/TacEx/Xu-Kim + sắc hoá khác biệt | Cao | Thấp |
| 7 | C1 — Data/Code Availability + Implementation Details | Trung-Cao | Thấp |
| 8 | B2 + B4 — thêm mean±std; chạy ≥3 seed bake-off (hoặc ghi rõ n=1) | Trung | Trung |
| 9 | D1, E1–E7 — polish (latency framing, kiến trúc, RQ2 nhãn, README map, Fig.5, draft-smell, abstract) | Thấp-Trung | Thấp |

---

## Phụ lục — finding bị BÁC (để bạn yên tâm)

- *"cos 0.973 và round-trip 1.9px không có file backing"* → **SAI**. Cả hai nằm trong `runs/phase5/sensor_build.json` (`flow_disp_cos_mean=0.9749`, `round_trip.overall_px=1.712`). Chỉ có 2 vấn đề nhỏ còn lại: README trỏ sai file (E3), và 1.9px là slip-subset chứ overall là 1.7px.
