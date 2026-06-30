# Báo cáo Giai đoạn 3 — RQ1-RQ3 sau realistic reground

**Ngày cập nhật:** 2026-06-30  
**Trạng thái:** Bản cũ dùng Hertz/PhysX FEM đã bị thay thế cho paper hiện tại. Báo cáo này chỉ dùng số từ realistic IPC/UIPC reground.

---

## 0. Kết luận ngắn

Giai đoạn 3 đã được reground trên bộ ground truth đại diện VBTS mỏng thực tế hơn: gel `20 x 20 x 3 mm`, structured tetrahedral mesh, marker grid `32 x 32`, field res `24`, lấy mẫu bề mặt bằng bilinear interpolation. Bộ chuẩn hiện hành là:

`data/uipc/shear_res24_avg_swept_REALISTIC.npz`

Các claim Phase 3 hiện dùng:

| Câu hỏi | Kết quả hiện hành |
|---|---:|
| RQ1 FNO relL2 toàn cục | **0.041** |
| RQ1 FNO hướng tiếp tuyến | **1.6 deg** |
| MLP relL2 / hướng | **0.501 / 83.9 deg** |
| FNO thắng MLP | **12.14x** |
| Slip macro-F1 / binary slip-F1 | **0.940 / 0.980** |
| RQ2 high-radius degradation | **1.80x** |
| RQ2 high-friction degradation | **2.36x** |
| RQ2 high-modulus degradation | **1.11x** |
| RQ3 FNO throughput | **7839 fps** |
| RQ3 GT solver throughput, single solve | **0.094 fps** |
| RQ3 GT target throughput, K=3 | **0.031 fps** |
| RQ3 FNO speedup vs single solve | **83204x** |

Những headline tiền-reground về PhysX/analytic, speedup cũ, và flat-punch OOD không còn dùng làm evidence cho bản paper realistic.

---

## 1. Ground truth hiện hành

Pipeline hiện dùng IPC/UIPC thay cho PhysX FEM cũ. Mỗi sample được chạy `K=3` replicate và lấy trung bình để giảm nhiễu contact.

| Mục | Giá trị |
|---|---|
| File dữ liệu | `data/uipc/shear_res24_avg_swept_REALISTIC.npz` |
| Tổng mẫu | **2520** |
| Split | **2120 train / 400 test** |
| Test seed | **2026** |
| Mode counts, all | normal 425 / stick 712 / partial 895 / full 488 |
| Mode counts, test | normal 68 / stick 113 / partial 142 / full 77 |
| Geometry | gel `20 x 20 x 3 mm`, sphere-only |
| Contact eps velocity | `2.5e-5` |
| `d_hat` | `1e-4` |
| Contact resistance | `1e9` |
| Velocity tolerance | `1e-3` |
| Solver time, single solve | **10.614 s/sample** |
| Solver time, K=3 target | **31.843 s/sample** |
| Nonnormal tangential noise, mean / p95 / max | **2.378% / 9.56% / 45.35%** |

Production sweep:

| Parameter | Range |
|---|---|
| Depth | `0.15-0.75 mm` |
| Radius | `2-6 mm` |
| Shear drive | `g * mu * 0.001 m` |
| `g` | `0-1.3` |
| `mu` | `0.4-0.8` |
| `E` | `0.5e5-2e5 Pa` |

---

## 2. RQ1 — FNO accuracy

FNO giữ lợi thế rõ ràng khi bài toán được đặt đúng dạng field-to-field. Kết quả chính lấy từ `runs/phase3_fem/benchmark.json`.

| Model | Overall relL2 | Direction error |
|---|---:|---:|
| **FNO** | **0.041** | **1.6 deg** |
| MLP per-point | 0.501 | 83.9 deg |

Theo mode:

| Mode | FNO relL2 |
|---|---:|
| normal | 0.033 |
| stick | 0.030 |
| partial | 0.042 |
| full | 0.065 |

Slip heads:

| Head | Score |
|---|---:|
| Macro-F1, 4 mode | **0.940** |
| Binary slip-F1 | **0.980** |

Kết luận: claim khoa học vẫn là nonlocal operator framing. Nhưng evidence hiện tại phải nói trên realistic IPC/UIPC thin-gel GT, không phải trên bộ PhysX/analytic cũ.

---

## 3. RQ2 — Generalization trong envelope hiện hành

RQ2 hiện được diễn giải là kiểm tra upper-tail trong cùng family sphere-contact, không còn claim flat-punch OOD.

| Axis | Degradation |
|---|---:|
| High radius | **1.80x** |
| High friction `mu` | **2.36x** |
| High modulus `E` | **1.11x** |

Giới hạn diễn giải: đây là stress test ở đuôi phân phối trong sweep realistic, chưa phải bằng chứng tổng quát hóa sang hình học indentor khác.

---

## 4. RQ3 — Speed

FNO inference nhanh hơn solver IPC/UIPC nhiều bậc, nhưng so sánh phải ghi rõ đơn vị solver:

| Metric | Value |
|---|---:|
| FNO throughput | **7839 fps** |
| GT single-solve throughput | **0.094 fps** |
| GT K=3 target throughput | **0.031 fps** |
| Speedup vs single solve | **83204x** |

Trong paper nên tránh mọi câu tốc độ dựa trên bộ PhysX cũ, vì không cùng GT hiện hành.

---

## 5. Bakeoff VBTS/surrogate

Nguồn: `runs/phase3_fem/vbts_baselines.json`.

| Baseline | relL2 | Direction | FNO advantage |
|---|---:|---:|---:|
| TACTO-style | 0.521 | 79.7 deg | 12.63x |
| Cattaneo-Mindlin | 0.491 | 81.0 deg | 11.89x |
| Taxim/FOTS-style | 0.229 | 5.1 deg | 5.54x |
| MLP | 0.501 | 83.9 deg | 12.14x |
| DeepONet | 0.046 | 1.9 deg | 1.12x |
| U-Net | 0.058 | 2.7 deg | 1.42x |
| Galerkin | 0.091 | 2.4 deg | 2.19x |
| **FNO** | **0.041** | **1.6 deg** | **1.00x** |

Kết luận trung thực: FNO thắng mạnh các baseline cục bộ/giải tích sai hình học; lợi thế so với DeepONet hẹp, nên claim chính nên nhấn vào operator framing + differentiable control pipeline, không phóng đại architecture-only novelty.

---

## 6. Nguồn kiểm chứng

- Dataset: `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- Benchmark: `runs/phase3_fem/benchmark.json`
- Baseline bakeoff: `runs/phase3_fem/vbts_baselines.json`
- Gate checker: `infra/verify_realistic_reground.py`
- Paper sync: `docs/kse2026/main.tex`
