# KSE 2026 submission — Differentiable Neural-Operator Tactile Surrogate

Bản nháp paper cho **KSE 2026**, special session *"Fusion of Embodied AI and Soft Robotics"*.

## Mốc thời gian (xác nhận từ CFP chính thức)
| Mốc | Ngày |
|---|---|
| **Nộp bài (full paper)** | **15/07/2026** |
| Thông báo chấp nhận | 31/08/2026 |
| Camera-ready | 10/09/2026 |
| Hội nghị | 11–14/11/2026, Kanazawa, Nhật |

- Định dạng: **IEEE conference (IEEEtran), tối đa 6 trang, tiếng Anh**.
- Nộp qua CMT: https://cmt3.research.microsoft.com/KSE2026
- CFP: https://kse2026.kse-conferences.org/call-for-papers/

## File
```
main.tex          # paper (IEEEtran conference)
references.bib    # tài liệu tham khảo
figs/             # hình (đã copy từ runs/)
  fidelity_speed.png      # Fig 2 — fidelity vs speed (RQ3)
  policy_servo_curve.png  # Fig 4 — control autograd vs ES (RQ4)
  sensor_gt_vs_fno.png    # Fig 5 — marker-dot image GT vs FNO (RQ5)
  env_demo.png            # (dự phòng) policy vs target imprint
```
Fig 1 (sơ đồ pipeline) vẽ bằng TikZ ngay trong `main.tex`, không cần file ảnh.

## Build
**Cách nhanh nhất — Overleaf** (IEEEtran có sẵn): tạo project mới, upload cả thư mục `kse2026/`, đặt `main.tex` làm main document, compiler = pdfLaTeX. Xong.

**Local** (cần texlive: `texlive-latex-recommended texlive-publishers texlive-pictures texlive-fonts-recommended`):
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
> Máy hiện tại CHƯA cài texlive nên chưa compile thử được — đã lint cấu trúc (braces/env/cite/ref/figs đều OK).

## Việc cần làm trước khi nộp (đánh dấu `[TODO]`/đỏ trong `main.tex`)
1. **Tác giả + cơ quan** — điền `\author{...}` (đang là placeholder).
2. **Acknowledgment** — điền nguồn tài trợ/lab.
3. **Số inverse-từ-ảnh trung bình** — báo cáo ghi 2.1%/1.1° nhưng file trên đĩa
   (`runs/phase5/sensor_inverse.json`) chỉ có 1 frame (10.6%/1.5°). Draft đang dùng
   số xác minh được (raw-field 2.3%, hướng-ảnh ~1.5°). → Nên sinh lại số trung bình
   nhiều frame rồi cập nhật câu + bỏ footnote TODO.
4. **Kiểm tra page-count** — 4 hình + 2 bảng khá nhiều cho 6 trang. Nếu tràn:
   bỏ Fig 2 (fidelity_speed) chuyển thành câu trong text, hoặc thu nhỏ Fig 5.
5. **Verify 3 trích dẫn** đánh dấu `% VERIFY` trong `references.bib`
   (FOTS, DiffTactile, Taccel — venue/năm/tác giả).
6. **Anonymize?** KSE thường review **không** ẩn danh (single-blind) → giữ tên tác
   giả. Kiểm tra lại yêu cầu của special session phòng khi double-blind.

## Map số liệu → nguồn (để tự kiểm)
| Số trong paper | File |
|---|---|
| Bake-off (Bảng I) | `runs/phase3_fem/vbts_baselines.json` |
| RQ1/RQ2/RQ3 | `runs/phase3_fem/benchmark.json` |
| Control autograd vs ES (Bảng II) | `runs/phase4/policy_servo.json` |
| Sensor faithful/round-trip | `runs/phase5/sensor_{compare,compat}.json` |
| Inverse raw-field 2.3% | `runs/phase3_fem/inverse_demo.json` |
| Env gap-closed 87% / gradcheck | `runs/phase6/env_demo.json` |
