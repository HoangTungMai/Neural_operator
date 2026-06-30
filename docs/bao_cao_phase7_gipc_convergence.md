# Báo cáo Phase 7 — realistic IPC/UIPC ground-truth reground

**Ngày cập nhật:** 2026-06-30  
**Trạng thái:** Bản hội tụ GIPC cũ đã hoàn thành vai trò exploratory. Paper hiện dùng cấu hình realistic IPC/UIPC final dưới đây.

---

## 0. Kết luận

Phase 7 đã chuyển dự án khỏi GT PhysX/analytic cũ sang bộ IPC/UIPC representative cho VBTS thin gel. Cấu hình được chọn vì đạt cân bằng giữa:

- hình học mỏng thực tế hơn (`20 x 20 x 3 mm`),
- ổn định contact đủ cho shear sweep,
- chi phí production chấp nhận được,
- và nhiễu replicate được định lượng bằng K=3 averaging.

Dataset final:

`data/uipc/shear_res24_avg_swept_REALISTIC.npz`

---

## 1. Cấu hình final

| Mục | Giá trị |
|---|---|
| Gel | `20 x 20 x 3 mm` |
| Mesh | structured tetrahedral |
| Marker grid | `32 x 32` |
| Field resolution | `24` |
| Surface sampling | bilinear top sampling |
| Indentor | sphere-only |
| Replicates | `K=3`, averaged |
| `eps_velocity` | `2.5e-5` |
| `d_hat` | `1e-4` |
| `contact_resistance` | `1e9` |
| `velocity_tol` | `1e-3` |

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

## 2. Hội tụ và lý do chọn tham số

Exploratory Phase 7 cũ dùng nhiều experiment để chọn solver setting. Các kết luận còn giữ:

| Check | Kết quả dùng để ra quyết định |
|---|---|
| `eps_velocity` axis | giảm xuống `2.5e-5` cho nghiệm ổn hơn |
| `5e-5 -> 2.5e-5`, tangential delta | khoảng **1.54%** |
| `5e-5 -> 2.5e-5`, normal delta | khoảng **0.42%** |
| res24 -> res28, K=3 averaged tangential | khoảng **0.95%** |
| res24 -> res28, K=3 averaged normal | khoảng **1.18%** |
| res20 -> res24 | chưa sạch để dùng làm final convergence claim |

Diễn giải: final report/paper chỉ nên nói “chosen by convergence/stability sweep and K=3 replicate averaging”, không nên overclaim mesh convergence tuyệt đối trên mọi trục.

---

## 3. Dataset final

| Metric | Value |
|---|---:|
| Samples | **2520** |
| Train / test | **2120 / 400** |
| Test seed | **2026** |
| Mode counts, all | normal 425 / stick 712 / partial 895 / full 488 |
| Mode counts, test | normal 68 / stick 113 / partial 142 / full 77 |
| Mean single solve | **10.614 s** |
| Mean K=3 target | **31.843 s** |
| Nonnormal tangential noise mean | **2.378%** |
| Nonnormal tangential noise p95 | **9.56%** |
| Nonnormal tangential noise max | **45.35%** |

Nhiễu tail vẫn là limitation phải giữ trong paper. K=3 không biến GT thành “noise-free”; nó chỉ làm target ổn hơn đủ để train/evaluate surrogate.

---

## 4. Những claim cũ bị thay thế

Không dùng các claim sau như evidence hiện hành:

| Claim cũ | Trạng thái |
|---|---|
| PhysX FEM là GT chính | Thay bằng IPC/UIPC realistic GT |
| Thick-gel geometry cũ | Thay bằng `20 x 20 x 3 mm` |
| Hertz/half-space là validator chính cho paper | Chỉ còn là historical context/baseline |
| PhysX speed/speedup cũ | Thay bằng IPC/UIPC `0.094 fps` single solve và `83204x` speedup |
| Flat-punch OOD | Không còn claim trong paper current |
| Force inverse một-frame cũ | Thay bằng multi-frame inverse `15.51% / 3.79 deg` |

---

## 5. Acceptance

Acceptance hiện được kiểm tra bằng:

`infra/verify_realistic_reground.py`

Checklist:

- dataset final tồn tại và có đúng N/mode split,
- metrics Phase 3-6 đồng bộ với paper,
- stale headline literals trong `docs/kse2026/main.tex` không còn,
- các file figure/paper được rebuild từ kết quả realistic.
