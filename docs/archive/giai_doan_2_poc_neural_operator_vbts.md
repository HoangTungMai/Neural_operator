# Giai doan 2 - Proof-of-Concept loi

Trang thai: KICH HOAT CO DIEU KIEN.

Ly do: 3 tien de ban dau da duoc xac minh o muc PoC:
- Tien de 1 qua Gate 1: chua thay cong trinh lap day tron ven neural operator cho marker displacement field cua VBTS.
- Tien de 2 qua Gate 2 co dieu kien: 1 GPU RTX 2000 Ada 16GB train duoc model nho, TacEx/UIPC logs cho thay chi phi ground-truth kha thi.
- Tien de 3 qua Gate 3 co dieu kien: FNO hoc duoc field physics-proxy 32x32 ke ca partial/full slip; DeepONet underfit; slip event detection bang heuristic chua tot.

Buoc tiep theo de kich hoat Giai doan 2 that: tao TacEx/UIPC extractor dung format `params, coords, disp, mode`.

Muc tieu: chung minh neural operator hoc duoc truong dich chuyen marker VBTS trong kich ban tiep xuc don gian, va kiem tra som rui ro lon nhat: slip / partial slip.

Quyet dinh dau vao:
- Gate 1 da qua: chua thay cong trinh lap day tron ven "operator learning cho marker displacement field cua VBTS".
- Gate 2 da qua co dieu kien: compute/training va chi phi sinh physical contact frames kha thi.
- Gate 3 da qua co dieu kien tren physics-proxy: dung FNO lam main architecture neu marker nam tren grid.
- Giai doan nay chi hop le voi data that khi co it nhat mot nguon ground-truth nho: FEM/GIPC/MPM/TacEx/FOTS-derived baseline.

## 1. Pham vi PoC

PoC toi thieu:
- Sensor surface: mat phang 2D voi grid marker co dinh.
- Contact primitive: normal indentation bang sphere/flat punch.
- Input: tham so tiep xuc va vat lieu toi thieu, vi du `(x, y, depth, radius, shear_x, shear_y, mu, E)`.
- Output: truong dich chuyen marker `u(x, y) = (u_x, u_y, u_z)` hoac toi thieu `(u_x, u_y)`.

PoC co slip:
- Contact sequence: press -> shear -> partial slip/full slip.
- Label phu neu co: contact mode `{stick, partial_slip, slip}`.
- Neu khong co label mode, suy ra bang threshold tren tangential displacement gradient hoac residual so voi rigid shear.

## 2. Dataset nho cho Giai doan 2

Muc tieu ban dau:
- Train: 8k-20k frames.
- Validation: 1k-2k frames.
- Test in-distribution: 1k-2k frames.
- Test out-of-distribution: 1k-2k frames voi depth/radius/shear/material nam ngoai khoang train.

Split bat buoc:
- Split theo trajectory/contact case, khong split ngau nhien tung frame neu cac frame lien tiep gan nhau.
- Tach rieng slip test set de khong bi RMSE trung binh che lap loi vung slip.

Dinh dang du lieu de xuat:
```text
data/
  phase2_poc/
    train.npz
    val.npz
    test_id.npz
    test_ood.npz
    test_slip.npz
```

Moi file nen co:
```text
params: [N, P]              # contact/material parameters
coords: [M, 2] or [M, 3]    # marker coordinates
disp:   [N, M, C]           # marker displacement field
mode:   [N] optional        # stick / partial_slip / slip
meta:   solver, units, sensor geometry, material range
```

## 3. Baseline bat buoc

Baseline de reviewer kho bat be:
- MLP coordinate regressor: `(params, coord) -> displacement`.
- FNO nho: main architecture neu marker nam tren grid deu.
- DeepONet nho: chi giu lam ablation/negative baseline, vi Gate 3 proxy cho thay underfit.
- U-Net hoac coordinate model co Fourier features: baseline bo sung neu can.
- FOTS marker motion neu tai lap duoc: baseline tactile-specific.

Khong nen chi so DeepONet voi solver. Cau hoi chinh la: operator co hon surrogate neural thuong khong?

## 4. Metric

Metric toan truong:
- MAE marker displacement.
- RMSE marker displacement.
- Relative L2 error.
- Max error / 95th percentile error.

Metric theo che do tiep xuc:
- Error rieng cho normal-only.
- Error rieng cho shear/stick.
- Error rieng cho partial slip.
- Error rieng cho full slip.

Metric slip:
- Slip event timing error, neu co sequence.
- Slip classification F1, neu co mode label.
- Tangential direction error: angle giua vector shear du doan va ground truth.
- Boundary/localization error cua vung partial slip, neu co mask.

Metric toc do:
- Inference frames/s tren 1 GPU.
- Latency/frame voi batch size 1 va batch size RL.
- VRAM peak.
- Speedup so voi solver ground-truth.

## 5. Tieu chi Gate 3

GO:
- Relative L2 tren test ID <= 5-10%.
- Slip/partial slip khong bi vo hoan toan: F1 >= 0.75 hoac metric tuong duong.
- Toc do inference nhanh hon solver it nhat 50x, muc tieu 100x+.
- Ket qua tot hon MLP coordinate baseline ro rang tren OOD hoac data efficiency.

PIVOT:
- Vung tron hoc tot nhung slip hong.
- Chuyen sang contact-mode gating: model phan loai mode truoc, moi mode mot expert.
- Hoac thu hep claim: quasi-static / stick-contact marker displacement surrogate.

NO-GO:
- Model khong hoc duoc field co y nghia, loi gan bang baseline tam thuong.
- Data ground-truth qua nhieu noise/khong nhat quan khien khong danh gia duoc.
- Chi phi sinh data khong dap ung duoc quy mo PoC.

## 6. Thu tu thuc hien

1. Chon nguon ground-truth: simulator that neu co; neu chua co thi dung synthetic toy data de smoke-test architecture.
2. Co dinh representation: marker grid hay point cloud.
3. Tao dataset nho normal-only.
4. Train MLP coordinate baseline.
5. Train DeepONet nho.
6. Them shear va slip sequence.
7. Bao cao Gate 3 voi bang metric theo contact mode.

## 7. Ket qua can co de ket thuc Giai doan 2

- Script/notebook train va evaluate tai lap duoc.
- Bang ket qua MLP vs DeepONet vs optional FNO/FOTS.
- Bieu do field error tren normal, shear, slip.
- Quyet dinh Gate 3: GO / PIVOT / NO-GO.
