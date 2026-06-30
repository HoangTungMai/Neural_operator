# Báo cáo tổng kết Phase 3-6 — realistic IPC/UIPC reground

**Ngày cập nhật:** 2026-06-30  
**Phạm vi:** Phase 3-6 sau khi reground toàn bộ paper trên realistic thin-gel IPC/UIPC GT  
**Trạng thái:** Bản tổng kết cũ ngày 2026-06-17 dùng PhysX/Hertz và nhiều số tiền-reground; không còn là nguồn evidence cho paper hiện tại.

---

## 0. Executive summary hiện hành

Dự án hiện chứng minh một pipeline VBTS surrogate gồm:

1. **Ground truth realistic hơn:** IPC/UIPC thin-gel contact, gel `20 x 20 x 3 mm`, marker grid `32 x 32`, field res `24`, sphere-contact sweep có friction/material/radius/depth.
2. **FNO field-to-field:** FNO đạt relL2 **0.041**, hướng **1.6 deg**, thắng MLP **12.14x** trên realistic GT.
3. **Differentiable policy:** backprop qua FNO đóng băng đạt chất lượng tương đương ES/oracle nhưng dùng ít forward eval hơn nhiều.
4. **Sensor pipeline:** image-space marker renderer nối được với FNO; inverse từ ảnh multi-frame hoạt động, nhưng normal-only còn yếu.
5. **Differentiable env:** reward gap closed **99.93%**; finite-difference diagnostic trên noisy image reward có relative error **5.21%** và **không pass formal gradcheck**, nên chỉ dùng như sanity check gradient-flow cục bộ cho one-step image-reward env.

Các headline tiền-reground về geometry dày, PhysX speed, force inverse một-frame, và env demo cũ đã bị supersede.

---

## 1. Ground truth và dataset

| Mục | Giá trị hiện hành |
|---|---|
| Dataset | `data/uipc/shear_res24_avg_swept_REALISTIC.npz` |
| Samples | **2520** |
| Split | **2120 train / 400 test** |
| Modes, all | normal 425 / stick 712 / partial 895 / full 488 |
| Modes, test | normal 68 / stick 113 / partial 142 / full 77 |
| Replicates | **K=3**, averaged |
| Mean single solve | **10.614 s** |
| Mean K=3 target | **31.843 s** |
| Mean nonnormal tangential noise | **2.378%** |
| Nonnormal noise p95 / max | **9.56% / 45.35%** |

Production envelope:

| Parameter | Range |
|---|---|
| Depth | `0.15-0.75 mm` |
| Radius | `2-6 mm` |
| Shear drive | `g * mu * 0.001 m` |
| `g` | `0-1.3` |
| `mu` | `0.4-0.8` |
| `E` | `0.5e5-2e5 Pa` |

Contact numerics:

| Parameter | Value |
|---|---|
| `eps_velocity` | `2.5e-5` |
| `d_hat` | `1e-4` |
| `contact_resistance` | `1e9` |
| `velocity_tol` | `1e-3` |

---

## 2. Phase 3 — Neural operator RQ1-RQ3

Nguồn chính: `runs/phase3_fem/benchmark.json` và `runs/phase3_fem/vbts_baselines.json`.

| Metric | Value |
|---|---:|
| FNO overall relL2 | **0.041** |
| FNO direction error | **1.6 deg** |
| MLP relL2 / direction | **0.501 / 83.9 deg** |
| FNO advantage over MLP | **12.14x** |
| Slip macro-F1 / binary slip-F1 | **0.940 / 0.980** |

Per-mode FNO relL2:

| normal | stick | partial | full |
|---:|---:|---:|---:|
| 0.033 | 0.030 | 0.042 | 0.065 |

RQ2:

| Tail axis | Degradation |
|---|---:|
| High radius | **1.80x** |
| High friction `mu` | **2.36x** |
| High modulus `E` | **1.11x** |

RQ3:

| Metric | Value |
|---|---:|
| FNO throughput | **7803 fps** |
| GT single-solve throughput | **0.094 fps** |
| GT K=3 target throughput | **0.031 fps** |
| Speedup vs single solve | **82827x** |

Architecture/baseline comparison:

| Baseline | relL2 | Direction | FNO advantage |
|---|---:|---:|---:|
| TACTO-style | 0.521 | 79.7 deg | 12.63x |
| Cattaneo-Mindlin | 0.491 | 81.0 deg | 11.89x |
| Taxim/FOTS-style | 0.229 | 5.1 deg | 5.54x |
| MLP | 0.501 | 83.9 deg | 12.14x |
| DeepONet | 0.046 | 1.9 deg | 1.12x |
| U-Net | 0.067 | 3.5 deg | 1.63x |
| Galerkin | 0.091 | 2.4 deg | 2.19x |
| **FNO** | **0.041** | **1.6 deg** | **1.00x** |

Claim nên dùng: FNO là lựa chọn mạnh trong framing field-to-field và gọn cho differentiable control. Không nên nói FNO thắng “áp đảo” mọi neural architecture, vì DeepONet khá sát.

---

## 3. Phase 4 — Differentiable policy

Nguồn: `runs/phase4/policy_servo.json`.

| Metric | Autograd policy | ES baseline | Oracle/reference |
|---|---:|---:|---:|
| Final loss | **5.70e-10** | 6.21e-10 | 3.04e-11 |
| Forward evaluations | **300** | 19200 | - |
| Wall time | **13.21 s** | 317.75 s | - |
| Reaches target | **21** | 1771 | - |

Diễn giải:

- Autograd dùng **64x** ít forward eval hơn ES.
- Đạt target với **84x** fewer iterations/eval checkpoints theo metric hiện hành.
- Wall-clock nhanh hơn **24.1x** trong cấu hình demo.

---

## 4. Phase 5 — Sensor renderer và inverse từ ảnh

Nguồn: `runs/phase5/sensor_build.json` và `runs/phase5/sensor_inverse_multiframe.json`.

Sensor build, cosine agreement:

| Mode | Cosine |
|---|---:|
| normal | -0.396 |
| stick | 0.559 |
| partial | 0.918 |
| full | 0.967 |

Round-trip marker pixel error:

| Mode | px error |
|---|---:|
| normal | 0.232 |
| stick | 0.394 |
| partial | 0.868 |
| full | 1.500 |

Multi-frame inverse from rendered sensor images:

| Metric | Value |
|---|---:|
| Overall magnitude error | **15.51% +/- 16.76%** |
| Overall direction error | **3.79 +/- 4.84 deg** |

By mode:

| Mode | Magnitude error | Direction error |
|---|---:|---:|
| normal | 30.09% | 10.17 deg |
| stick | 7.35% | 2.05 deg |
| partial | 11.02% | 1.67 deg |
| full | 13.60% | 1.28 deg |

Diễn giải trung thực: image-space loop khả dụng cho shear-rich regimes; normal-only inverse vẫn yếu và không nên dùng các claim force-recovery một-frame từ báo cáo cũ.

---

## 5. Phase 6 — Differentiable tactile environment

Nguồn: `runs/phase6/env_demo.json`.

| Metric | Value |
|---|---:|
| Reward gap closed | **99.93%** |
| Finite-difference relative error | **5.21%** |
| Formal gradcheck passed | **false** |

Claim nên dùng: env nối FNO + sensor + reward có gradient flow và policy demo hoạt động; finite-difference diagnostic trên noisy image-reward env hiện tại là sanity check, không phải formal proof. Không nên mở rộng claim này sang multi-step contact dynamics.

---

## 6. Limitations còn phải nói rõ

- GT hiện là sphere-only trong production envelope, chưa chứng minh OOD hình học khác indentor.
- K=3 averaging giảm noise nhưng vẫn còn tail tangential noise lớn: p95 **9.56%**, max **45.35%**.
- Sensor renderer là differentiable synthetic marker image, chưa phải calibration với camera/gel thật.
- Normal-only image inverse yếu hơn shear modes.
- DeepONet gần FNO về relL2; contribution nên đặt vào toàn pipeline và field-to-field framing.

---

## 7. Source map

- Paper: `docs/kse2026/main.tex`
- Dataset: `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- Phase 3 metrics: `runs/phase3_fem/benchmark.json`
- Baseline bakeoff: `runs/phase3_fem/vbts_baselines.json`
- Phase 4 policy: `runs/phase4/policy_servo.json`
- Phase 5 sensor: `runs/phase5/sensor_build.json`, `runs/phase5/sensor_inverse_multiframe.json`
- Phase 6 env: `runs/phase6/env_demo.json`
- Acceptance check: `infra/verify_realistic_reground.py`
