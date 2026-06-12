# Bao cao Gate 3 - Tien de 3: operator hoc displacement field va slip

Ngay kiem tra: 2026-06-05

## Ket luan ngan

Trang thai tien de 3: **GO CO DIEU KIEN**.

Noi dung da xac minh:
- Neural operator dang FNO hoc duoc displacement field tren physics-proxy marker dataset.
- Ket qua tot ca tren partial slip va full slip: relative L2 khoang 5-7%.
- DeepONet nho/dua vua underfit ro ret trong bai nay, nen khong nen chon DeepONet lam kien truc chinh neu marker nam tren grid.

Dieu kien:
- Ket qua nay dung synthetic/physics-proxy field, chua phai TacEx/UIPC ground-truth.
- Slip event detection bang heuristic tu field chua tot: F1 chi khoang 0.67. Neu paper claim slip event/mode, can them slip head hoac contact-mode gating.

## Dataset PoC

Script:
```text
scripts/gate3_operator_slip_check.py
```

Dataset:
- Marker grid: 32x32 = 1024 markers.
- Output: displacement `(u_x, u_y, u_z)`.
- Train: 16,000 frames.
- Test: 4,000 frames.
- Modes can bang:
  - normal
  - stick
  - partial_slip
  - full_slip

Field generator:
- Contact patch Gaussian.
- Normal indentation + lateral marker displacement.
- Stick: tangential displacement decays smoothly.
- Partial slip: sharp annular transition.
- Full slip: tangential motion gan rigid translation trong contact patch.

## Ket qua

### MLP coordinate baseline

Lenh:
```bash
.venv-gate2/bin/python scripts/gate3_operator_slip_check.py --train-per-mode 4000 --test-per-mode 1000 --marker-side 32 --batch-size 128 --epochs 25 --models mlp deeponet
```

Ket qua MLP field:
```text
overall relative L2: 0.130
normal:       0.112
stick:        0.183
partial slip: 0.122
full slip:    0.102
train time: 66.6 s
peak VRAM: 1.07 GiB
```

Dien giai:
- MLP hoc duoc field o muc tam chap nhan, nhung stick mode con kho.
- Day la baseline neural thuong can giu trong paper.

### DeepONet

DeepONet nho ban dau:
```text
overall relative L2: 0.454
partial slip: 0.429
full slip:    0.437
```

DeepONet lon hon, 50 epochs:
```text
overall relative L2: 0.355
normal:       0.352
stick:        0.395
partial slip: 0.334
full slip:    0.340
train time: 186.4 s
peak VRAM: 1.18 GiB
```

Dien giai:
- DeepONet underfit ro trong bai toan marker field co contact patch di dong va slip transition.
- Khong nen lay DeepONet lam main architecture neu khong co cai tien lon: Fourier features, local basis, gating, hoac decomposition theo contact patch.

### FNO

Lenh:
```bash
.venv-gate2/bin/python scripts/gate3_operator_slip_check.py --train-per-mode 4000 --test-per-mode 1000 --marker-side 32 --batch-size 128 --epochs 35 --models fno
```

Ket qua:
```text
overall relative L2: 0.0665
overall RMSE: 0.00290

normal:       0.0518
stick:        0.0947
partial slip: 0.0687
full slip:    0.0507

train time: 199.7 s
peak VRAM: 0.88 GiB
parameters: 2.67M
```

Slip metric:
```text
slip score MAE: 0.104
fixed-threshold slip F1: 0.667
best-threshold slip F1: 0.673
```

Dien giai:
- Field learning qua nguong PoC, ke ca partial/full slip.
- Slip event classification bang heuristic score tu field chua dat nguong 0.75.
- Neu can slip-mode output, nen them supervised slip head hoac contact-mode gating thay vi suy slip bang heuristic hau xu ly.

## Quyet dinh Gate 3

**GO CO DIEU KIEN.**

GO cho claim hep:
```text
Neural operator dang FNO co the hoc marker displacement field, ke ca vung partial/full slip, tren physics-proxy PoC.
```

PIVOT kien truc:
```text
DeepONet -> FNO cho marker grid.
Neu marker khong phai grid deu: can Geo-FNO, PointNet-style operator, hoac coordinate model co local Fourier features.
```

PIVOT slip:
```text
Khong claim slip event detection neu chi dung heuristic tu field.
Neu can slip mode cho RL/reviewer, them contact-mode classifier/gating:
- head phan loai normal/stick/partial/full slip
- expert rieng cho slip/non-slip
- loss phu cho tangential direction va slip boundary
```

## Dieu kien truoc Giai doan 3/paper-scale

1. Lap lai benchmark voi TacEx/UIPC data extractor that.
2. Giu FNO lam baseline main neu marker duoc noi suy thanh grid.
3. Giu MLP coordinate regressor lam baseline bat buoc.
4. Khong dung DeepONet lam main model tru khi co cai tien ro.
5. Them metric slip co label that: mode F1, event timing, tangential direction error, boundary localization.
