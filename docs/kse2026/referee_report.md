# Báo cáo phản biện — KSE 2026, sau realistic reground

**Paper:** *A Differentiable Neural-Operator Surrogate for Vision-Based Tactile Sensing and Control*  
**Ngày cập nhật:** 2026-06-30  
**Trạng thái:** Bản phản biện cũ dựa trên draft PhysX/analytic đã bị thay thế. Review này đối chiếu với `docs/kse2026/main.tex` và outputs realistic hiện hành.

---

## 0. Phán quyết tổng thể

| | |
|---|---|
| Khuyến nghị nội bộ | **Weak Accept nếu giữ giọng trung thực hiện tại** |
| Rủi ro chính | GT vẫn synthetic sphere-only; sensor inverse normal-only yếu; DeepONet gần FNO |
| Điểm mạnh chính | realistic IPC/UIPC reground đồng bộ, FNO field-to-field mạnh, differentiable policy/sensor/env có artifact tái dựng được |

Paper hiện đã qua các lỗi nặng của draft cũ: headline metrics không còn dựa trên PhysX/Hertz tiền-reground, RQ5 không còn số một-frame khó truy xuất, figures/paper được rebuild, và acceptance script bắt stale literals trong `main.tex`.

---

## 1. Điểm mạnh

1. **Ground truth được reground rõ ràng.** Dataset final `data/uipc/shear_res24_avg_swept_REALISTIC.npz` có N=2520, split 2120/400, mode split công khai, K=3 averaging, và solver/contact parameters ghi được.
2. **RQ1 mạnh và sạch hơn bản cũ.** FNO đạt relL2 **0.041**, hướng **1.6 deg**, thắng MLP **12.14x**; slip macro-F1 **0.940** và binary slip-F1 **0.980**.
3. **RQ3 giờ cùng thước đo với GT hiện hành.** FNO **7839 fps** vs IPC/UIPC single solve **0.094 fps**, speedup **83204x**; không còn trộn PhysX speed cũ.
4. **Differentiable story có artifact đủ.** Phase 4 policy có autograd vs ES, Phase 5 có renderer/inverse multi-frame, Phase 6 có env demo và finite-difference diagnostic.
5. **Limitations đã đáng tin hơn.** Paper không còn claim flat-punch OOD, không gọi sensor pipeline là physical calibration, và ghi rõ normal-only inverse yếu.

---

## 2. Điểm yếu còn lại

### A. Scope GT vẫn hẹp

Production dataset là sphere-only trong envelope depth/radius/friction/material. RQ2 chỉ nên đọc là upper-tail in-envelope, không phải geometric OOD.

**Khuyến nghị:** giữ wording hiện tại kiểu “in-envelope upper-tail stress test”; nếu còn chỗ, thêm một câu trong limitation rằng indentor geometry transfer chưa được chứng minh.

### B. K=3 không xoá hết nhiễu contact

Nonnormal tangential noise mean **2.378%** là ổn, nhưng p95 **9.56%** và max **45.35%** vẫn cao. Đây là risk nếu referee hỏi “ground truth precision”.

**Khuyến nghị:** trình bày K=3 như noise reduction/target stabilization, không phải proof of noise-free GT.

### C. DeepONet gần FNO

DeepONet relL2 **0.046** và direction **1.9 deg** khá sát FNO **0.041 / 1.6 deg**. Vì vậy novelty architecture-only không phải điểm bán mạnh nhất.

**Khuyến nghị:** nhấn contribution là field-to-field operator framing + differentiable control/sensor pipeline trên realistic GT; FNO là lựa chọn tốt, không phải bằng chứng mọi spectral operator sẽ thống trị.

### D. Sensor inverse không đều theo mode

Overall multi-frame inverse là **15.51%** magnitude và **3.79 deg**, nhưng normal mode tệ hơn đáng kể: **30.09%** và **10.17 deg**. Normal image có cosine alignment âm trong sensor build.

**Khuyến nghị:** giữ RQ5 ở mức “image-space loop works best in shear-rich regimes”; không dùng nó như claim force reconstruction tổng quát.

### E. Env gradient diagnostic chưa pass formal

Phase 6 reward gap closed **99.93%**, nhưng finite-difference relative error **5.21%** và `passed=false`.

**Khuyến nghị:** claim “gradients flow and support optimization demo”, không claim formal gradient exactness.

---

## 3. Checklist trước khi nộp

- Chạy `infra/verify_realistic_reground.py` trước bản nộp cuối.
- Nếu rebuild paper, kiểm tra figures `fidelity_speed.png`, `policy_servo_curve.png`, `sensor_gt_vs_fno.png` vẫn được sinh từ runs realistic.
- Grep `main.tex` cho stale headline metrics tiền-reground.
- Đảm bảo generated PDFs/PPTX báo cáo nội bộ cũ không nằm cạnh `.md` current nếu chưa rebuild.
- Caption RQ2/RQ5/RQ6 phải giữ caveat như trong `main.tex`.

---

## 4. Kết luận

Sau realistic reground, đề tài đã qua ngưỡng “paper có thể bảo vệ được” cho KSE nếu giữ đúng phạm vi. Điểm cần giữ vững là sự trung thực: surrogate rất mạnh trong setting hiện tại, nhưng evidence chưa phủ physical sensor calibration, geometry OOD, hay formal gradient verification toàn diện.
