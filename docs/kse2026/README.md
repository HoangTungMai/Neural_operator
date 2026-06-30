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
Máy hiện tại có TeX Live; bản cuối phải chạy đủ chuỗi trên và kiểm tra
`main.log` không còn undefined citation/reference.

## Việc cần làm trước khi nộp
1. **Tác giả + cơ quan** — điền `\author{...}` (đang là placeholder).
2. **Acknowledgment** — điền nguồn tài trợ/lab.
3. **Kiểm tra page-count** — 4 hình + 2 bảng khá nhiều cho 6 trang. Nếu tràn:
   bỏ Fig 2 (fidelity_speed) chuyển thành câu trong text, hoặc thu nhỏ Fig 5.
4. **Verify 3 trích dẫn** đánh dấu `% VERIFY` trong `references.bib`
   (FOTS, DiffTactile, Taccel — venue/năm/tác giả).
5. **Anonymize?** KSE thường review **không** ẩn danh (single-blind) → giữ tên tác
   giả. Kiểm tra lại yêu cầu của special session phòng khi double-blind.

## Map số liệu → nguồn (để tự kiểm)
| Số trong paper | File |
|---|---|
| Bake-off (Bảng I) | `runs/phase3_fem/vbts_baselines.json` |
| RQ1/RQ2/RQ3 | `runs/phase3_fem/benchmark.json` |
| Control autograd vs ES (Bảng II) | `runs/phase4/policy_servo.json` |
| Sensor cosine by regime / round-trip | `runs/phase5/sensor_build.json` |
| Sensor inversion by regime | `runs/phase5/sensor_inverse_multiframe.json` |
| Env gap-closed / gradcheck | `runs/phase6/env_demo.json` |

Current realistic-geometry inverse-from-image numbers are multi-frame:
`runs/phase5/sensor_inverse_multiframe.json` reports 20 frames total, with
overall magnitude error `15.51%` and direction error `3.79°`.
