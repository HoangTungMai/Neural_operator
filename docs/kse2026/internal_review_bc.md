# Review de tai Neural Operator VBTS

**Reviewer:** kho tinh, uu tien tinh khoa hoc, do trung thuc va kha nang tai lap  
**Pham vi review:** paper KSE draft, README, codebase `src/`, scripts `infra/`, outputs `runs/`, dataset/provenance trong `data/`  
**Ket luan ngan:** **Weak Accept co dieu kien** neu giu dung scope hien tai va don sach artifact cu.

---

## 1. Phan quyet tong the

De tai co loi khoa hoc dang bao ve:

- hoc surrogate field-to-field cho vision-based tactile sensor (VBTS);
- corrected-BC IPC/UIPC ground truth co provenance va verifier;
- FNO surrogate duoc gan vao differentiable control va differentiable marker renderer;
- paper hien tai da noi kha ro gioi han: synthetic GT, sphere-centric, one-step quasi-static.

Nhung de tai chua du bang chung de claim thanh:

- physical tactile sensor model da duoc calibration voi hardware that;
- tactile simulator tong quat cho nhieu hinh hoc indentor;
- formal differentiable simulator cho multi-step contact dynamics;
- FNO architecture vuot troi ap dao moi neural architecture.

Danh gia cuoi: **co the nop neu sua nhung diem ve trung thuc artifact/provenance truoc deadline**. Neu de nguyen repo voi nhieu bao cao va checker mau thuan, reviewer noi bo kho tinh se xem day la rui ro reproducibility nghiem trong.

---

## 2. Diem manh

1. **Huong khoa hoc hop ly.** Field-to-field framing dung voi ban chat elastic response phi cuc bo: dau vao la contact/penetration field, dau ra la marker displacement field. Day la luan diem tot hon so voi chi noi "dung FNO".

2. **Paper chinh hien da trung thuc hon ban cu.** `docs/kse2026/main.tex` dung bo corrected-BC `REALISTIC_BC`, bao cao FNO relL2 `0.074`, MLP advantage `7.35x`, speedup `41,253x`, va noi ro sphere-centric/single-step/simulation GT.

3. **Co acceptance checker dung cho bo final.** `infra/verify_bc_reground.py` check dataset final, fixed bottom BC, indentor strength, adaptive K, stale literals trong paper, downstream JSON va figures.

4. **Pipeline co artifact tai lap.** Main outputs nam trong:
   - `runs/phase3_fem/benchmark.json`
   - `runs/phase3_fem/vbts_baselines.json`
   - `runs/phase4/policy_servo.json`
   - `runs/phase5/sensor_build.json`
   - `runs/phase5/sensor_inverse_multiframe.json`
   - `runs/phase6/env_demo.json`

5. **Code compile duoc va checker moi pass.** Da kiem:
   - `.venv-gate2/bin/python -m compileall -q src tests infra` pass
   - `.venv-gate2/bin/python infra/verify_bc_reground.py` pass
   - `.venv-gate2/bin/python tests/test_aggregate_uipc_replicates.py` pass
   - `.venv-gate2/bin/python -m pip check` pass

---

## 3. Findings nghiem trong

### F1. Nhieu "su that cuoi cung" mau thuan trong repo

Muc do: **High**

Paper/README hien hanh dung:

- dataset: `data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz`
- FNO relL2: `0.074`
- MLP advantage: `7.35x`
- speedup: `41,253x`
- robust target: `K<=5`

Nhung nhieu report noi bo van ghi:

- dataset: `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- FNO relL2: `0.041`
- MLP advantage: `12.14x`
- speedup: `83k`
- replicate: `K=3`

Vi du cac file co so cu:

- `docs/bao_cao_giai_doan3_rq_results.md`
- `docs/bao_cao_tong_ket_phase3-6.md`
- `docs/bao_cao_phase7_gipc_convergence.md`
- `docs/kse2026/referee_report.md`
- `infra/verify_realistic_reground.py`

Checker cu `infra/verify_realistic_reground.py` da fail voi output:

```text
FAIL: runs/phase3_fem/benchmark.json gt='shear_res24_avg_swept_REALISTIC_BC.npz'
```

Day la bang chung ro rang rang repo dang co hai lineage artifact cung song song. Day la rui ro lon nhat ve do trung thuc va reproducibility.

Khuyen nghi:

- doi ten cac report cu thanh `*.superseded.md` hoac chuyen vao `docs/archive/`;
- them banner dau file: "Superseded by REALISTIC_BC";
- chi giu `infra/verify_bc_reground.py` la final acceptance checker;
- README can ghi mot bang "canonical source of truth".

---

### F2. Scope khoa hoc con hep

Muc do: **High**

De tai dang dung:

- synthetic IPC/UIPC simulation ground truth;
- sphere-only production geometry;
- one-step quasi-static map;
- synthetic marker-dot renderer;
- no real hardware calibration.

Paper co noi dieu nay trong limitations, nhung khi thuyet trinh/bao cao can cuc ky can than. Claim hop le la:

> FNO surrogate reproduces corrected-BC IPC/UIPC synthetic tactile fields and enables differentiable one-step control/image demos.

Claim khong nen dung:

> We solve real VBTS simulation/control in general.

Rui ro reviewer:

- "FEM-fidelity" chi co nghia fidelity voi solver/dataset nay, khong phai fidelity voi real GelSight/DIGIT.
- "differentiable sensor pipeline" chua co optics/camera/material calibration that.
- "generalization" chi la upper-tail in-envelope, chua phai geometry OOD.

Khuyen nghi:

- giu tu "sphere-centric", "synthetic", "one-step", "quasi-static" trong abstract/limitations;
- neu con cho, them cau: "No physical sensor calibration or non-spherical object transfer is claimed."

---

### F3. FNO khong thang ap dao cac neural architecture

Muc do: **Medium-High**

Ket qua final trong `runs/phase3_fem/vbts_baselines.json`:

| Model | relL2 | Direction |
|---|---:|---:|
| FNO | 0.074 | 1.6 deg |
| U-Net | 0.085 | 2.2 deg |
| DeepONet | 0.115 | 3.2 deg |
| Galerkin Transformer | 0.151 | 4.1 deg |

FNO hon U-Net chi `1.15x`, DeepONet `1.55x`. Trong khi do RQ1/bakeoff co ve single-seed (`torch.manual_seed(0)`) cho cac model chinh.

Ket luan dung:

> Dense/non-local learned models beat local/analytic/linear marker-motion baselines; FNO is strongest in this bakeoff.

Ket luan khong nen dung:

> FNO architecture is uniquely or overwhelmingly superior.

Khuyen nghi:

- chay 3-5 seeds cho FNO, U-Net, DeepONet, MLP;
- report mean/std;
- neu khong kip, ha giong claim architecture-only.

---

### F4. Slip-F1 la regime classification, khong phai physical slip-state identification tong quat

Muc do: **Medium**

Mode label den tu threshold drive ratio trong aggregation:

- `G_STICK = 0.04`
- `G_PARTIAL = 0.48`
- `G_FULL = 1.0`

Paper co noi "labelled by tangential-drive thresholds", nen khong gian doi. Nhung neu goi la "slip detection" qua rong thi reviewer co the bat:

- model dang hoc nhan regime do pipeline gan;
- chua chung minh detect stick/slip tu contact state vat ly doc lap;
- nhan gan theo input drive co the de hon nhan vat ly.

Khuyen nghi:

- dung "contact-mode/regime F1";
- tranh "physical slip detector" tru khi co nhan tu solver contact state.

---

### F5. Baseline VBTS la reimplementation core, khong phai full simulator comparison

Muc do: **Medium**

`vbts_baselines.py` noi ro:

- chi reimplement marker-motion core;
- khong chay full TACTO/Taxim/FOTS optical/rendering stack;
- cross-paper numbers khong comparable.

Day la cach fair hon cho metric marker field, nhung reviewer co the hoi:

- reimplementation co dung voi original khong?
- hyperparameter/calibration co cong bang khong?
- "beats TACTO" co phai system-level claim khong?

Khuyen nghi:

- trong paper giu "TACTO-style", "Taxim/FOTS-style", "marker-motion core";
- khong viet "we outperform TACTO/Taxim/FOTS" theo nghia full simulator.

---

## 4. Findings codebase / reproducibility

### C1. Default data cua nhieu module van tro ve `data/fem/...`

Muc do: **Medium**

Nhieu entrypoint live default sang legacy PhysX/FEM data:

- `src/novbts/research/fno/fem_benchmark.py`
- `src/novbts/research/fno/vbts_baselines.py`
- `src/novbts/research/fno/diff_policy.py`
- `src/novbts/simulation/sensor/build_sensor_dataset.py`
- `src/novbts/simulation/sensor/sensor_inverse_demo.py`
- `src/novbts/simulation/sensor/tactile_env.py`

Pipeline dung final thi co truyen `--data data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz` trong `infra/run_bc_downstream.sh`, nhung nguoi doc chay module khong tham so se sinh ket qua khac.

Khuyen nghi:

- doi default data cua cac script paper-facing sang `REALISTIC_BC`;
- hoac bat buoc `--data` required cho script paper-facing;
- de legacy defaults trong scripts archive thoi.

---

### C2. Test story chua khep kin

Muc do: **Medium**

`tests/test_aggregate_uipc_replicates.py` chay truc tiep pass, nhung:

```bash
.venv-gate2/bin/python -m pytest tests
```

fail vi `pytest` khong co trong venv. `pyproject.toml` khong khai bao optional test deps.

Khuyen nghi:

- them `[project.optional-dependencies] test = ["pytest"]`;
- them command chinh thuc vao README;
- neu khong muon dung pytest, doi tests thanh scripts va document ro.

---

### C3. Naming/provenance con legacy PhysX/FEM

Muc do: **Medium-Low**

Mot so docstring/alias van dung "FEM", "PhysX" cho pipeline hien dung IPC/UIPC. Vi du:

- `fem_benchmark.py` docstring noi PhysX/FEM cu;
- `paths.py` comment noi `data/fem`;
- `gt_solver_k3_averaged` con la compatibility alias trong JSON.

Khuyen nghi:

- doi language sang "IPC/UIPC GT" trong file paper-facing;
- neu giu alias, ghi ro "compatibility alias, not current semantics".

---

### C4. Paper con placeholder truoc khi nop

Muc do: **Low**

Trong `docs/kse2026/main.tex`:

- `\author{\IEEEauthorblockN{Anonymous Authors}}`
- `\todo{funding / grant numbers / hardware support...}`

Khuyen nghi:

- xac nhan KSE single-blind/double-blind;
- xoa TODO truoc khi submit;
- rebuild LaTeX full chain.

---

## 5. Do trung thuc cua de tai

Danh gia: **kha tot trong paper chinh, yeu trong repo artifact hygiene**.

Khong thay bang chung ve viec co y bop meo ket qua trong `main.tex`. Cac so paper hien tai khop voi JSON final:

- FNO `0.074`
- MLP `0.545`
- advantage `7.35x`
- speedup `41,253x`
- image inverse `12.6% / 3.4 deg`
- env grad diagnostic rel error `7.31%`, `passed=false`

Tuy nhien repo de lai qua nhieu tai lieu cu chua archive, lam nguoi doc co the nham sang bo so tot hon. Voi reviewer kho tinh, day la "honest presentation risk", khong phai fraud, nhung can sua truoc nop.

---

## 6. Checklist sua truoc khi nop

Bat buoc:

- [ ] Archive hoac gan banner superseded cho tat ca docs con so `0.041`, `12.14x`, `83k`, `K=3`, `REALISTIC.npz`.
- [ ] Chi dinh `infra/verify_bc_reground.py` la final checker; doi ten hoac archive `verify_realistic_reground.py`.
- [ ] Doi defaults hoac require `--data` cho cac script paper-facing.
- [ ] Xoa TODO/placeholder author trong `docs/kse2026/main.tex`.
- [ ] Them test dependency hoac command test chinh thuc.

Nen lam neu con thoi gian:

- [ ] Multi-seed RQ1 cho FNO/U-Net/DeepONet/MLP.
- [ ] Report mean/std cho bakeoff neural architectures.
- [ ] Them ablation ve adaptive replicate filtering K<=5.
- [ ] Them mot table nho "claim vs evidence vs limitation" vao README.

Khong nen lam truoc deadline neu lam roi:

- [ ] Claim real hardware calibration.
- [ ] Claim geometry OOD neu chua co corrected-BC non-sphere data.
- [ ] Claim formal gradient verification cho env hien tai.
- [ ] Claim FNO la architecture duy nhat vuot troi.

---

## 7. Ket luan cuoi

De tai co the bao ve o muc **weak accept** neu tac gia giu giong trung thuc:

> Mot FNO field-to-field surrogate hoc corrected-BC IPC/UIPC synthetic tactile fields, chay rat nhanh, co the dat vao differentiable one-step control va differentiable marker-image demo.

Diem can giu vung:

- surrogate manh trong setting hien tai;
- GT la averaged synthetic IPC/UIPC, khong phai real sensor;
- geometry con sphere-centric;
- env la one-step quasi-static;
- FNO hon U-Net/DeepONet nhung margin voi neural baselines khong lon.

Neu don sach provenance va artifact drift, day la mot de tai co cau chuyen khoa hoc sach. Neu khong don, reviewer se co ly do chinh dang de nghi reject vi khong ro ket qua nao moi la ket qua final.
