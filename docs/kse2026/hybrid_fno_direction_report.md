# Huong di moi: Local-refined FNO va Full U-FNO

**Ngay:** 2026-07-02  
**Trang thai:** proposal/probe noi bo, chua phai artifact paper chinh thuc; U-FNO khong phai kien truc moi trong literature  
**Lien quan:** RQ1 surrogate fidelity, architecture ablation, huong nang weak accept len strong accept

## 1. Ly do mo huong nay

Sau corrected-BC reground, ket qua paper hien tai da sach hon ve ground truth: gel bottom duoc fixed, shear khong con bi rigid drift, va FNO baseline dat rel-L2 `0.0741`, direction error `1.61 deg` tren seed0. Tuy nhien review noi bo chi ra mot diem yeu: FNO hien tai chua thang ap dao U-Net. FNO tot hon U-Net, nhung margin chua du manh de claim kien truc Fourier la dong gop architecture thuyet phuc.

Do do huong moi la nang cap FNO theo cach giu uu diem global/non-local cua Fourier layers, nhung bo sung local convolutional refinement de bat chi tiet contact-local deformation tot hon. Diem moi hop le cua paper khong nen la "de xuat U-FNO", ma la "ap dung va kiem chung global-local neural operator cho corrected-BC VBTS tactile deformation fields".

## 1.1. Chinh sua novelty: U-FNO da ton tai

U-FNO khong nen duoc claim la kien truc moi. Literature da co U-FNO/U-shaped neural operator:

- Wen et al., **U-FNO -- An enhanced Fourier neural operator-based deep-learning model for multiphase flow**, arXiv 2021 / Advances in Water Resources 2022: https://arxiv.org/abs/2109.03697
- U-FNO code/repo cua tac gia mo ta viec them mini U-Net path vao Fourier layer: https://github.com/gegewen/ufno
- U-NO, **U-shaped Neural Operators**, OpenReview 2022: https://openreview.net/forum?id=j3oQF9coJd

Vi vay claim dung nen la:

> We adapt and evaluate a lightweight global-local FNO variant for corrected-BC vision-based tactile deformation fields.

Khong nen viet:

> We propose a new U-FNO architecture.

## 2. Ba muc architecture

| Model | Y tuong | Params | Trang thai |
|---|---:|---:|---|
| FNO baseline | Fourier/global operator trunk | `2.67M` | paper baseline hien tai |
| Local-refined FNO | FNO trunk + CNN local refinement nho o cuoi | `2.76M` | da co 3-seed probe |
| Full U-FNO | U-Net encoder-decoder + skip connections + Fourier/global block | `5.26M` | existing architecture family; moi co seed0 probe |

### FNO baseline

FNO baseline hoc field-to-field map bang spectral/Fourier convolution. Diem manh la bat quan he global, phu hop voi tactile elastic field vi contact mot vung co anh huong phi cuc bo len ca gel. Diem yeu la chi tiet cuc bo gan contact co the bi lam muot.

### Local-refined FNO

Local-refined FNO giu FNO lam trunk chinh, sau do them mot module CNN nho de refine output theo lan can cuc bo. No gan nhu la "FNO + local correction head". Vi chi tang tu `2.67M` len `2.76M` params, day la huong co ti le loi ich/chi phi rat tot.

### Full U-FNO

Full U-FNO lai sau hon: dung encoder-decoder kieu U-Net, skip connections de giu chi tiet multi-scale, va Fourier/global reasoning de giu nang luc non-local. Tuy nhien day la mot architecture family da co trong literature, nen trong paper chi nen dung nhu baseline/ablation hybrid co tham chieu citation. Ket qua seed0 tot nhat, nhung params gan gap doi FNO va chua co multi-seed.

## 3. Ket qua probe hien co

Nguon so lieu:

- `runs/phase3_fem/local_refined_fno_multiseed.json`
- `runs/phase3_fem/ufno_seed0.json`

### Multi-seed: FNO vs Local-refined FNO

| Model | rel-L2 mean | rel-L2 std | Direction mean | Direction std | Params |
|---|---:|---:|---:|---:|---:|
| FNO | `0.0790` | `0.0118` | `1.84 deg` | `0.38 deg` | `2.67M` |
| Local-refined FNO | `0.0595` | `0.0024` | `1.51 deg` | `0.12 deg` | `2.76M` |

Nhan xet:

- Local-refined FNO giam rel-L2 trung binh khoang `24.7%` so voi FNO multi-seed.
- Variance giam manh: rel-L2 std tu `0.0118` xuong `0.0024`.
- Direction error giam tu `1.84 deg` xuong `1.51 deg`.
- Params chi tang khoang `3.3%`.

Day la ket qua dang gia nhat hien tai vi no vua cai thien accuracy, vua lam mo hinh on dinh hon, trong khi chi phi architecture gan nhu khong tang dang ke.

### Seed0: Full U-FNO

| Model | rel-L2 overall | normal | stick | partial | full-slip | Direction | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| FNO seed0 | `0.0741` | `0.0691` | `0.0704` | `0.0694` | `0.0928` | `1.61 deg` | `2.67M` |
| Local-refined FNO seed0 | `0.0602` | `0.0448` | `0.0514` | `0.0604` | `0.0861` | `1.46 deg` | `2.76M` |
| Full U-FNO seed0 | `0.0552` | `0.0372` | `0.0442` | `0.0552` | `0.0874` | `1.32 deg` | `5.26M` |

Nhan xet:

- Full U-FNO seed0 la ket qua accuracy tot nhat hien tai.
- Loi ich lon nhat nam o normal/stick/partial; full-slip khong thang ro Local-refined FNO.
- Params tang gan `97%` so voi FNO baseline.
- Moi co seed0, nen chua du de claim chinh trong paper.

## 4. Dien giai khoa hoc

Ket qua probe ung ho gia thuyet:

> Fourier/global operator can learn the non-local elastic response, while lightweight local convolutional refinement improves contact-local details that are smoothed by a pure FNO trunk.

Day la cau chuyen khoa hoc manh hon viec chi noi "FNO tot hon U-Net". Neu chung minh day du, paper co the chuyen tu narrative:

> FNO is our best surrogate.

sang:

> A lightweight global-local FNO adaptation is better suited for corrected-BC tactile deformation fields because it combines non-local elastic coupling with local contact-detail refinement.

Cau chuyen nay co kha nang day paper tu weak accept len strong accept, vi no bien mot ket qua benchmark thanh mot insight co ly do vat ly. Tuy nhien contribution phai duoc viet la domain-specific adaptation/validation, khong phai architecture invention.

## 5. Nen dua model nao lam ung vien chinh?

### Khuyen nghi hien tai

Dung **Local-refined FNO** lam ung vien chinh de nang cap paper, neu cac benchmark chinh thuc tiep theo tiep tuc khop voi probe.

Ly do:

- Cai thien rel-L2 ro: `0.0790 -> 0.0595` multi-seed.
- On dinh hon FNO: std nho hon nhieu.
- Params chi tang nhe: `2.67M -> 2.76M`.
- De justify hon trong paper vi trade-off accuracy/complexity rat dep.

Dung **Full U-FNO** nhu cited architecture ablation hoac promising upper-bound, chua nen lam model chinh ngay.

Ly do:

- Seed0 tot nhat: `0.0552`.
- Nhung params gan gap doi.
- Chua co multi-seed va throughput.
- Khoang cach voi Local-refined FNO khong lon bang khoang cach Local-refined FNO voi FNO.

## 6. Viec can lam truoc khi dua vao paper

### Bat buoc

1. Tich hop Local-refined FNO vao source chinh, khong chi nam trong script probe.
2. Chay lai benchmark chinh thuc tren corrected-BC dataset voi cung split/budget:
   - FNO
   - Local-refined FNO
   - Full U-FNO neu kip
   - U-Net
   - DeepONet
   - MLP
3. Report multi-seed mean/std, toi thieu 3 seed.
4. Report per-mode rel-L2: normal, stick, partial, full-slip.
5. Do throughput/FPS, latency, params, GPU memory.
6. Cap nhat verifier/provenance neu bat ky so nao duoc dua vao paper.

### Nen lam

1. Ve mot figure nho: architecture schematic FNO vs Local-refined FNO.
2. Them ablation:
   - FNO trunk only
   - local CNN only/U-Net
   - FNO + local refinement
   - Full U-FNO
3. Kiem tra loi ich co truyen xuong downstream:
   - policy servo
   - sensor inverse
   - differentiable tactile env

### Chua nen lam

1. Thay paper claim sang Full U-FNO khi moi co seed0.
2. Noi Full U-FNO la best architecture neu chua do variance va speed.
3. Noi U-FNO/Full U-FNO la kien truc moi.
4. Xoa FNO baseline cu; can giu no de reviewer thay contribution tang dan va co comparison cong bang.

## 7. Rui ro va cach viet trung thuc

Rui ro lon nhat la overclaim. Probe hien tai rat hua hen, nhung chua phai artifact chinh thuc. Cach viet an toan:

- "We further observe that adding a lightweight local refinement head improves stability and field accuracy."
- "Full U-FNO is included as a cited higher-capacity hybrid baseline."
- "The main reported model is selected by the best accuracy/complexity trade-off, not by seed0 peak performance."

Nen tranh:

- "Full U-FNO solves the problem."
- "U-FNO is conclusively superior."
- "FNO architecture alone is the main reason for all gains."
- "We introduce U-FNO as a new architecture."

## 8. Ket luan

Huong hybrid FNO la mot huong rat dang theo duoi, nhung novelty phai dat dung cho. Ket qua manh nhat hien tai khong phai la Full U-FNO, ma la **Local-refined FNO**: no cai thien accuracy va variance mot cach ro rang voi chi phi tham so rat nho. Full U-FNO co the la cited upper-bound accuracy, nhung can multi-seed va throughput truoc khi co the dung lam dong gop chinh.

De nang paper len muc strong accept, huong tot nhat la bien Local-refined FNO thanh mot contribution co kiem chung day du: corrected-BC GT + lightweight global-local FNO adaptation cho tactile fields + differentiable downstream demos.
