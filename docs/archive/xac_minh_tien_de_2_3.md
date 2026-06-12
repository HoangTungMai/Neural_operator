# Xac minh tien de 2 va 3

Trang thai hien tai:
- Tien de 1 da qua Gate 1: chua thay cong trinh lap day tron ven neural operator cho marker displacement field cua VBTS.
- Tien de 2 qua muc PoC: PyTorch CUDA chay tren RTX 2000 Ada 16GB, DeepONet nho train duoc synthetic marker field 32x32-64x64, va TacEx/UIPC logs cho thay chi phi sinh contact frames vat ly kha thi trong vai gio cho 20k-100k frames. Con thieu extractor TacEx/UIPC dung format de tai.
- Tien de 3 qua muc physics-proxy PoC: FNO hoc duoc displacement field 32x32, ke ca partial/full slip, voi relative L2 khoang 5-7%. DeepONet underfit; slip event detection bang heuristic chua tot.

Thu tu dung:
1. Giai doan 1/P0 -> xac minh tien de 2 -> Gate 2.
2. Neu Gate 2 GO/PIVOT kha thi -> Giai doan 2/PoC -> xac minh tien de 3 -> Gate 3.

## Tien de 2 - Kha thi tren 1 GPU

Trang thai: **GO CO DIEU KIEN, NGHIENG VE GO CHO POC**. Xem `bao_cao_gate2_kha_thi_1gpu.md` va `bao_cao_gate2_chi_phi_groundtruth.md`.

Cau hoi can tra loi:
- Sinh duoc bao nhieu frame/cap ground-truth moi gio?
- Can bao nhieu frame de train PoC toi thieu?
- Model nho co train duoc trong VRAM 12-24GB khong?
- Dataset + training co nam trong quy thoi gian thuc te khong?

P0 toi thieu:
- Tao mot dataset nho normal-only: 500-2000 frames.
- Train MLP coordinate baseline.
- Train DeepONet nho.
- Do wall-clock time, VRAM peak, throughput inference.

Metric can ghi:
```text
data_source
num_frames
markers_per_frame
channels_per_marker
data_generation_frames_per_hour
train_time_per_epoch
total_train_time
peak_vram_gb
inference_frames_per_second
```

Gate 2:
- GO: sinh data va train model nho trong 1 GPU, co duong mo rong len 10k-50k frames.
- PIVOT: data dat nhung van co the thu hep scope, dung physics-informed loss, hoac dung simulator/data cong khai.
- NO-GO: khong sinh noi data hoac model khong vua VRAM ngay ca voi PoC nho.

Deliverable:
- Bao cao P0 ngan 1-2 trang.
- Bang chi phi data/train/VRAM.
- Quyet dinh nguon ground-truth cho Giai doan 2.

## Tien de 3 - Neural operator hoc duoc field va slip

Trang thai: **GO CO DIEU KIEN**. Xem `bao_cao_gate3_tien_de3_operator_slip.md`.

Da xac minh tren physics-proxy PoC:
- FNO hoc duoc field normal/stick/partial slip/full slip.
- Partial slip relative L2: ~0.069.
- Full slip relative L2: ~0.051.
- Peak VRAM: < 1 GiB trong benchmark 32x32.

Can pivot:
- DeepONet underfit trong bai toan nay; khong nen lam main architecture ban dau.
- Neu marker nam tren grid, dung FNO lam main.
- Neu marker la point cloud/irregular, dung Geo-FNO/PointNet-style operator/coordinate model co Fourier features.

Chua duoc claim:
- Slip event detection/mode classification bang heuristic tu field chua tot, F1 ~0.67.
- Neu can slip mode cho RL, them slip head hoac contact-mode gating.

Cau hoi can tra loi:
- Operator co hoc duoc displacement field normal/shear trong che do tron khong?
- Khi co partial slip/full slip, loi co tang vo ly khong?
- Operator co hon baseline neural thuong khong?

PoC toi thieu:
- Dataset normal-only de dam bao model hoc duoc field co ban.
- Them shear stick.
- Them press -> shear -> partial slip/full slip.

Baseline:
- MLP coordinate regressor.
- DeepONet nho.
- FNO/U-Net neu marker nam tren grid deu.
- FOTS neu tai lap duoc marker motion baseline.

Metric:
- Relative L2, MAE, RMSE toan field.
- Error theo contact mode: normal, stick, partial slip, full slip.
- Slip F1 hoac event timing error neu co label/sequence.
- Direction error cua tangential displacement.
- Inference speed va VRAM.

Gate 3:
- GO: model hoc tot field va slip chap nhan duoc; tot hon baseline neural thuong o OOD/data efficiency/toc do.
- PIVOT: field tron tot nhung slip hong -> contact-mode gating hoac thu hep claim sang quasi-static/stick.
- NO-GO: khong hoc duoc field co y nghia hoac khong hon baseline tam thuong.

Deliverable:
- Notebook/script train-evaluate tai lap.
- Bang MLP vs DeepONet vs optional FNO/FOTS.
- Plot error quanh vung slip.
- Quyet dinh Gate 3.
