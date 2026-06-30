# PROGRESS: realistic VBTS gel geometry reground

## Latest checkpoint (2026-06-30 00:20 +07)

Realistic geometry reground da pass end-to-end acceptance.

Post-review correction (2026-06-30 00:27 +07):

- Fixed stale old-run headline numbers that remained in `docs/kse2026/main.tex`
  abstract, contribution list, and RQ4 text:
  - FNO vs MLP now `12.14x`
  - classical/analytic VBTS baseline range now `5.54--12.63x`
  - control target-query advantage now `84x`
  - control wall speedup now `24.5x`
  - RQ4 losses/walls/oracle now match `runs/phase4/policy_servo.json`
- Rephrased the broad "whole sensor differentiable end-to-end" language to an
  autograd-connected image-space loop, consistent with the current env
  finite-difference diagnostic.
- Extended `infra/verify_realistic_reground.py` to fail on these stale headline
  numbers and require the refreshed ones.
- Rebuilt `docs/kse2026/main.pdf`; verifier passed
  `REALISTIC_REGROUND_ACCEPTANCE_OK` again.

Refactor/cleanup pass:

- Replaced remaining "end-to-end differentiable sensor" wording in
  `docs/kse2026/main.tex` with "autograd-connected sensor pipeline" wording.
- Updated `docs/kse2026/README.md` so the submission checklist no longer cites
  stale one-frame inversion numbers; it now points to
  `runs/phase5/sensor_inverse_multiframe.json`.
- Removed transient root progress files (`fem_progress*.txt`, `convergence.txt`)
  and generated LaTeX intermediates (`main.aux`, `main.blg`, `main.out`).
- Removed source/script `__pycache__` directories; left virtualenv caches alone.
- Kept datasets, runs, logs, `*.PRE_REALISTIC.*` backups, `main.pdf`, and
  `main.log` because they are provenance or verifier inputs.
- Rebuilt paper and reran verifier; acceptance still passes.

Runtime/state:

- `uipc-sweep-realistic-main.service`: `inactive` / unit not found; khong restart
- Docker containers: `0`
- downstream runner: exited `REALISTIC_DOWNSTREAM_DONE`
- final verifier: `REALISTIC_REGROUND_ACCEPTANCE_OK`
- legacy dataset unchanged:
  `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`

Final dataset:

- `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- N=`2520`, train/test=`2120/400`
- modes all normal/stick/partial/full = `425/712/895/488`
- test modes = `68/113/142/77`
- K=`3`, gel_res=`24`, bilinear marker sampling
- mean K3 solve time `31.843 s`; fair single-solve `10.614 s`
- mean non-normal tangential replicate noise `2.378%`
- split metadata present: `split_test_size=400`, `split_shuffle_seed=2026`

Fresh downstream outputs:

- `runs/phase3_fem/benchmark.json`
  - FNO rel-L2 overall `0.041`, direction `1.6 deg`
  - MLP rel-L2 overall `0.501`, direction `83.9 deg`
  - FNO/MLP rel-L2 advantage `12.14x`
  - RQ3 FNO `7839 fps`, fair GT solver `0.094 fps`, K3 target `0.031 fps`
  - speedup vs fair single solve `83204x`
- `runs/phase3_fem/vbts_baselines.json`
  - FNO advantages:
    - TACTO-style `12.63x`
    - Cattaneo--Mindlin `11.89x`
    - Taxim/FOTS linear `5.54x`
    - DeepONet `1.12x`
    - U-Net `1.42x`
    - Galerkin Transformer `2.19x`
- `runs/phase4/policy_servo.json`
  - autograd final `5.70e-10`, ES final `6.21e-10`
  - autograd reaches target in `21` fwd queries vs ES `1771` (`84x` fewer)
  - wall `13 s` vs `324 s`
- `runs/phase5/sensor_build.json`
  - cosine by mode normal/stick/partial/full =
    `-0.396/0.559/0.918/0.967`
  - round-trip px by mode normal/stick/partial/full =
    `0.232/0.394/0.868/1.500`
- `runs/phase5/sensor_inverse_multiframe.json`
  - overall magnitude error `15.51%`, direction `3.79 deg`
  - by-mode magnitude normal/stick/partial/full =
    `30.09%/7.35%/11.02%/13.60%`
- `runs/phase6/env_demo.json`
  - policy closes `99.93%` random-to-oracle reward gap
  - gradients flow to action
  - finite-difference diagnostic rel error `5.21%`, `passed=false`; paper now
    phrases this honestly as a sanity check, not a formal pass

Figures regenerated and accepted:

- `docs/kse2026/figs/fidelity_speed.png`
- `docs/kse2026/figs/policy_servo_curve.png`
- `docs/kse2026/figs/sensor_gt_vs_fno.png`

Paper:

- `docs/kse2026/main.tex` updated for realistic `20 x 20 x 3 mm` geometry,
  new counts, downstream metrics, solver speed, no stale flat-punch/OOD claim
- rebuilt `docs/kse2026/main.pdf` with
  `pdflatex -> bibtex -> pdflatex -> pdflatex`
- verifier confirmed no undefined refs/citations and semantic stale-claim gate
  passed

Acceptance command that passed:

```bash
rtk proxy .venv-gate2/bin/python infra/verify_realistic_reground.py
```

## Latest checkpoint (2026-06-29 17:20 +07)

Production GT van dang chay va la uu tien chinh:

- systemd user unit: `uipc-sweep-realistic-main.service`
- status: `active`
- current container: `uipcsweep_45`
- complete combos: `46 / 63`
- averaged frames: `1840 / 2520` (`73.02%`)
- raw replicates implied/present: `5520 / 7560` (`73.02%`)
- final realistic NPZ:
  `data/uipc/shear_res24_avg_swept_REALISTIC.npz` chua ton tai, dung nhu
  expected truoc khi full sweep xong

Pre-Phase-2 preparation da lam trong luc cho GT:

- `py_compile` pass:
  - `infra/verify_realistic_reground.py`
  - `src/novbts/groundtruth/aggregate_uipc_replicates.py`
- `bash -n` pass:
  - `infra/run_realistic_downstream.sh`
  - `infra/gen_uipc_sweep.sh`
- `infra/verify_realistic_reground.py` da chay thu va fail dung tai:
  missing `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- verifier xac nhan legacy dataset van unchanged:
  `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`

Khong nen launch them rehearsal/training nang. Cho production finish, sau do lam
theo thu tu:

1. Aggregate/gate final NPZ. Neu final NPZ thieu split metadata, rerun host
   aggregator voi `--test-size 400 --shuffle-seed 2026`.
2. Chay `infra/verify_realistic_reground.py`; dataset gate phai pass truoc Phase
   2.
3. Chay `infra/run_realistic_downstream.sh` voi realistic NPZ.
4. Cap nhat paper va rebuild sau khi co final downstream metrics.

## CPU partial-data rehearsal (2026-06-29)

Theo yeu cau tiep tuc thu downstream trong khi GPU production chay, da tao
snapshot rieng, khong overwrite active/final artifacts:

- snapshot: `data/uipc/rehearsal/shear_realistic_partial.npz`
- N=`964`, split stratified `764/200`, seed `2026`
- modes all=`30/310/402/222`
- modes test=`6/64/84/46`
- K=3, finite, non-normal tangential noise `2.178%`

Da them path overrides voi defaults khong doi trong `src/novbts/paths.py`:

- `NOVBTS_RUNS_DIR`
- `NOVBTS_FEM_DIR`
- `NOVBTS_DOCS_DIR`

Rehearsal chay CPU-only (`CUDA_VISIBLE_DEVICES=''`, nice priority, 4 threads),
epochs/steps rut gon de gate entrypoint/schema, khong dung metrics nay cho paper:

- benchmark -> `runs/rehearsal_realistic_partial/phase3_fem/benchmark.json`
- VBTS bake-off -> `runs/rehearsal_realistic_partial/phase3_fem/vbts_baselines.json`
- policy -> `runs/rehearsal_realistic_partial/phase4/policy_servo.json`
- sensor build -> `runs/rehearsal_realistic_partial/phase5/sensor_build.json`
- inversion -> `runs/rehearsal_realistic_partial/phase5/sensor_inverse_multiframe.json`
- env/gradcheck -> `runs/rehearsal_realistic_partial/phase6/env_demo.json`
- 3 figures -> `docs/rehearsal_realistic_partial/kse2026/figs/`

Ket qua pipeline:

- tat ca entrypoints exit 0
- env finite-difference gradcheck pass, rel error `1.49%`
- sensor JSON co cosine/round-trip per mode
- RQ3 JSON tach single solve `0.093 fps` va K3 target `0.031 fps`
- 9 rehearsal JSON co dung `gt`/`gt_path`

Rehearsal phat hien va da sua:

- concurrent snapshot aggregator khong duoc ghi avg vao root-owned active combo:
  them `--no-write-frame-averages`
- `sensor_inverse.json` bo sot `gt`/`gt_path` trong nested output: da them va
  rerun verify

Luu y: model final van phai retrain tren full N=2520. Snapshot rehearsal chi
chung minh commands/schema/output flow san sang; khong the chi append GT vao
weights rehearsal va coi la final scientific result.

### Full 80-epoch partial rehearsal (stopped de uu tien production)

Theo yeu cau user, da launch full-settings rehearsal tren cung frozen N=964
snapshot:

- initial 4-thread unit `realistic-partial-80ep.service` da stop sau 8 phut theo
  yeu cau user; no van o benchmark va chua sinh benchmark JSON
- current systemd unit: `realistic-partial-80ep-12t.service`
- CPU-only: `CUDA_VISIBLE_DEVICES=''`
- `OMP_NUM_THREADS=12`, `MKL_NUM_THREADS=12`, nice level `5`
- script: `infra/run_realistic_partial_80ep.sh`
- log: `logs/realistic_partial_80ep_12t.log`
- runs: `runs/rehearsal_realistic_partial_80ep/`
- FEM/sensor data: `data/fem/rehearsal_realistic_partial_80ep/`
- figures: `docs/rehearsal_realistic_partial_80ep/kse2026/figs/`

Sequence dung full settings:

- benchmark 80 epochs + classifier 40
- VBTS bake-off 80 epochs
- policy FNO 80 epochs + 300-step, 3-seed autograd/ES
- sensor build
- sensor inversion FNO 80 epochs + 400 steps, 5 frames/mode, 8 restarts
- env FNO 80 epochs + 300-step policy + gradcheck
- 3 figures

12-thread relaunch gate pass: N=964, train/test=764/200. Python dang dung
~`1147%` CPU (xap xi 12 cores) o `fem_benchmark`. Production service
`uipc-sweep-realistic-main` van active, khong restart.

Status full-80 rehearsal luc `2026-06-29 09:52 +07`:

- `fem_benchmark` da xong, JSON da ghi
- partial-snapshot RQ1 (rehearsal, khong phai final paper):
  - MLP overall `0.482`, direction `88.1 deg`
  - FNO overall `0.056`, direction `2.4 deg`
  - FNO/MLP rel-L2 advantage `8.55x`
  - FNO per-mode normal/stick/partial/full =
    `0.031/0.049/0.058/0.068`
  - FNO throughput `625 fps`
  - GT single solve `0.093 fps`, speedup `6729x`
- current step: `vbts_baselines` 80 epochs
- unit `realistic-partial-80ep-12t.service` active

Status luc `11:39 +07`:

- benchmark va VBTS baselines da xong
- current step: `diff_policy` / policy servo full settings
- policy command dung 300 steps, ES population 32, 3 seeds; day la CPU-heavy
  stage va chua co `policy_servo.json`
- unit active, process khong bi treo
- 12-thread CPU load lam production solve time tang tu `~10.7 s` len
  `~16--17 s`; GT van tien trien nhung cham hon

**Stopped theo yeu cau user:** luc sau `11:39 +07`, user quyet dinh rehearsal
da du va uu tien production GT. Da stop rieng
`realistic-partial-80ep-12t.service`:

- rehearsal unit: `inactive`
- production `uipc-sweep-realistic-main`: van `active`, khong restart
- benchmark + VBTS baseline JSON duoc giu
- policy servo dang chay bi dung, khong co `policy_servo.json`
- sensor/inversion/env full-80 chua chay
- reduced smoke rehearsal truoc do van day du cho tat ca entrypoints
- production solve time sau khi tra CPU ve khoang `12--12.5 s`

## Production status moi nhat (2026-06-29)

Kiem tra tu account/session moi:

- `uipc-sweep-realistic-main.service`: `active`
- khong restart service
- current container: combo `025`
- replicate NPZ: `4568 / 7560` (`60.4%`) at `2026-06-29 14:43 +07`
- completed averaged frames: `1520 / 2520` (`60.3%`, combos `000..037`)
- combo 38 dang chay (`8/120`)
- final realistic NPZ chua ton tai, dung nhu expected truoc khi full sweep xong

Audit tat ca 920 averaged frames:

- bad/corrupt/schema mismatch: `0`
- current mode counts: normal/stick/partial/full = `27/297/384/212`
- mean K=3 target solve time: `32.066 s`
- mean single solve time: `10.689 s`
- mean non-normal tangential replicate noise: `2.201%`
- p95: `8.842%`; co outlier cao nhung mean van duoi acceptance gate `3%`
- ETA theo measured throughput: khoang `14--15 h` con lai

## Phase 1 dang chay (2026-06-29 00:03 +07)

User da tiep tuc goal sau checkpoint Phase 0. Production campaign da launch:

- systemd user unit: `uipc-sweep-realistic.service`
- main PID sau relaunch: `2593317` (PID co the doi neu restart; unit name moi la
  identifier on dinh)
- sweep root: `data/uipc/sweep_realistic`
- target: `63 combos x 40 frames = 2520 frames`
- replicates: `K=3`
- final target: `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
- log: `logs/uipc_sweep_realistic.log`
- progress file: `fem_progress_uipc_0_62.txt`

Campaign dung transient user systemd service thay vi `nohup`. Lan launch `nohup`
dau tien lam wrapper bash mat trong khi container combo 0 bi orphan, nen campaign
se khong chuyen combo. Container orphan da duoc stop va campaign duoc relaunch
bang systemd; sweep resumable nen rep da xong khong bi tinh lai.

Theo doi:

```bash
rtk proxy systemctl --user status uipc-sweep-realistic --no-pager
rtk docker ps
rtk proxy tail -50 fem_progress_uipc_0_62.txt
rtk proxy tail -50 logs/uipc_sweep_realistic.log
```

Neu service bi fail:

```bash
rtk proxy systemctl --user reset-failed uipc-sweep-realistic
```

Sau do relaunch cung command/cau hinh; script se skip moi rep NPZ da ton tai va
hop le. Khong chay them UIPC workload GPU song song voi service nay.

Ngay sau launch, audit rows phat hien 63 combos ban dau deu la random, trai voi
yeu cau normal-contact top-up. Campaign da duoc dung khi moi co 15 rep, cac
artifact sai da bi xoa, va sampler da duoc sua thanh campaign design:

- combos `000..049`: random
- combos `050..058`: pure normal, grid `R={2,4,6} mm x E={0.5,1,2}e5 Pa`
- combos `059..062`: extra random

Class counts du kien truc tiep tu rows (va se duoc verify lai tren final NPZ):

- normal: `425`
- stick: `712`
- partial: `895`
- full: `488`
- total: `2520`

Service sach da relaunch luc `2026-06-29 00:06:37 +07`.

Luc `00:12 +07`, campaign duoc chuyen sang hai shard GPU song song theo chinh
pattern duoc document trong `gen_uipc_sweep.sh` (~1.5x throughput):

- `uipc-sweep-realistic-a.service`: combos `000..031`
  - progress `fem_progress_uipc_0_31.txt`
  - log `logs/uipc_sweep_realistic_a.log`
  - temporary aggregate `data/uipc/shear_res24_avg_swept_REALISTIC_shard_a.npz`
- `uipc-sweep-realistic-b.service`: combos `032..062`
  - progress `fem_progress_uipc_32_62.txt`
  - log `logs/uipc_sweep_realistic_b.log`
  - temporary aggregate `data/uipc/shear_res24_avg_swept_REALISTIC_shard_b.npz`

Hai shard chung resumable sweep root, khong overlap combo/container name. Khong
shard nao ghi vao final filename. Sau khi ca hai complete, phai aggregate lai
toan bo sweep root vao dung final path roi moi chay Phase 2 gate.

**Superseded luc 00:15 +07:** thu nghiem hai shard cho thay moi solve tang tu
`~11 s` len `~60 s` voi contact config `eps=2.5e-5` (GPU contention), nen tong
throughput cham gan 3x. Ca hai shard da duoc stop, orphan container da remove.
Campaign hien tai quay lai mot service:

- `uipc-sweep-realistic-main.service`
- log: `logs/uipc_sweep_realistic_main.log`
- progress: `fem_progress_uipc_0_62.txt`
- output final truc tiep:
  `data/uipc/shear_res24_avg_swept_REALISTIC.npz`

Sau khi quay lai mot container, solve time da tro ve `11.2--11.3 s`. Mot rep cua
combo 32 tu shard test duoc giu; no cung config va se duoc resumable batch skip
khi main service toi combo 32.

User systemd ban dau co `Linger=no`, nen service co nguy co dung neu tat het
session khi chuyen account. Da chay `loginctl enable-linger tungmai`; hien
`Linger=yes`, service se tiep tuc qua full logout/relogin.

Phase 2 da duoc chuan bi trong:

- `infra/run_realistic_downstream.sh`

Script nay:

- gate final NPZ: exactly 2520 frames, modes 0..3, K=3, res24, chosen contact
  params va bilinear marker sampling
- backup artifact cu thanh `*.PRE_REALISTIC.*` mot lan truoc khi overwrite
- chay benchmark, VBTS baselines, policy servo, sensor build, sensor inversion,
  env demo/gradcheck, figures theo thu tu dependency
- copy fresh `runs/phase5/gt_vs_fno_samples.png` vao paper figures

Tat ca JSON output Phase 2 da duoc bo sung provenance top-level:

- `gt = shear_res24_avg_swept_REALISTIC.npz`
- `gt_path = data/uipc/shear_res24_avg_swept_REALISTIC.npz`

Final UIPC aggregator cung da duoc sua de concatenate `velocity_tol`; schema nay
da test bang Phase-0 K=3 smoke artifact.

Acceptance verifier da duoc them:

- `infra/verify_realistic_reground.py`

Verifier gate dataset/schema/parameter box, JSON freshness + provenance, figures,
stale paper claims va LaTeX undefined refs. Hien tai no fail dung o missing final
NPZ vi Phase 1 chua xong.

Hai schema/metric gap duoc bat khi audit truoc Phase 2:

- `sensor_build.json` bay gio ghi `flow_disp_cos_by_mode` va
  `round_trip.by_mode_px` cho normal/stick/partial/full, thay vi chi blanket
  cosine + stick/slip.
- RQ3 benchmark bay gio tach:
  - `gt_solver`: fair single-solve FPS = `solve_time_s / n_replicates`
  - `gt_solver_k3_averaged`: production target FPS tinh ca K=3 calls
  Figure dung single-solve line; JSON note noi ro chi phi K=3.

Verifier da gate ca hai invariant nay va gate FNO rel-L2 < MLP rel-L2.

Split audit phat hien dataset IPC cu duoc aggregate theo combo order, lam last-400
test set bi top-up skew (`247 normal`, chi `34 full`). Final realistic aggregator
da duoc them deterministic stratified train/test ordering:

- `split_test_size=400`
- `split_shuffle_seed=2026`
- train roi test van nam theo convention first 2120 / last 400
- voi row counts du kien, test modes se xap xi `68/113/142/77`
- `source_frame_dir` duoc luu de trace moi shuffled frame ve artifact goc

Do main service da launch truoc edit nay, running bash co the van giu old script
inode. Sau Phase 1 phai gate split metadata; neu thieu, rerun host aggregator:

```bash
rtk proxy .venv-gate2/bin/python -m novbts.groundtruth.aggregate_uipc_replicates \
  --sweep-dir data/uipc/sweep_realistic \
  --out data/uipc/shear_res24_avg_swept_REALISTIC.npz \
  --mode-shear-scale 0.001 --expect-reps 3 \
  --test-size 400 --shuffle-seed 2026
```

Downstream runner va final verifier deu tu choi dataset neu split metadata/test
mode coverage khong dung.

Early production integrity check (combo 0, 20 complete frames):

- rows/depth/shear/mode label match: `62/62` rep artifacts checked, `0` errors
- no K replicate pair bitwise-identical
- mean raw tangential replicate noise:
  - stick `1.81%`
  - partial `2.71%`
  - full `2.87%`
- co mot so outlier `~7--8%`, nen final verifier gate mean non-normal
  tangential replicate noise `<=3%` truoc khi Phase 2 duoc phep chay.

Host-side K=3 aggregation da duoc test tren production frame 000:

- `solve_time_s=35.418 s` bang tong ba solve
- mean single solve `11.806 s`
- tangential replicate noise `1.862%`
- schema/provenance va shapes deu dung

Legacy dataset no-overwrite invariant da duoc khoa trong final verifier:

- path: `data/uipc/shear_res24_avg_swept.npz`
- size: `15,296,208 bytes`
- SHA-256:
  `c19338b94ac8e9cded746ac689ba543fb0b24abcc03036b48453376e57f81f91`

Verifier fail neu legacy file bi sua/xoa.

Acceptance grep scope note:

- raw grep toan bo `src/runs` khong the zero-match hop ly:
  - analytic baseline source co support `flat-punch` that su
  - archived/PRE_TOPUP JSON co old numbers va phai duoc giu
  - `0.975`/`0.341` cung co the xuat hien nhu mot metric khac do trung chuoi
- final verifier gate semantic target:
  - `docs/kse2026/main.tex` khong con stale claims/numbers
  - 6 active downstream JSON phai moi hon realistic NPZ va co exact realistic
    `gt`/`gt_path`
  - figures phai moi hon JSON dependency
- archived/PRE artifacts khong duoc dung lam evidence, nhung khong xoa vi task
  yeu cau backup.

Static traceability fix da ap dung trong `docs/kse2026/README.md`:

- sensor cosine/round-trip -> `runs/phase5/sensor_build.json`
- inversion by regime -> `runs/phase5/sensor_inverse_multiframe.json`
- env source giu `runs/phase6/env_demo.json` nhung bo stale hard-coded 87%

Physical BC da duoc audit truc tiep trong
`src/novbts/groundtruth/tacex_uipc_extract_shear.py`:

- structured gel z spans `0..gel_z`
- chi bottom mask (`z=min`) duoc set `is_constrained=1` va aim ve rest position
- top va sides khong co constraint
- toan bo rigid/stiff indenter duoc prescribed theo displacement schedule
- UIPC ground half-plane dat tai `-gel_z`, khong support gel

Vi vay paper wording "bottom bonded/pinned Dirichlet, top contact surface, free
sides, rigid displacement-controlled indenter" dung voi implementation. Khong
them provenance key moi vao driver khi production dang chay, de tranh combo 0 va
cac combo sau khac schema.

## Cap nhat checkpoint Phase 0 (2026-06-28 17:01 +07)

Phase 0 da hoan tat. Cau hinh duoc chon:

- geometry: `20 x 20 x 3 mm`
- structured tet, `gel_res=24`, marker grid `32 x 32`
- marker sampling: `bilinear`
- `eps_velocity=0.000025`
- `d_hat=0.0001`
- `contact_resistance=1.0e9`
- `velocity_tol=0.001`
- production depth: `U(0.15, 0.75) mm`
- production radius: `U(2, 6) mm`
- production shear: `g * mu * 0.001 m`, `g in [0, 1.3]`
- `K=3`

Eps-axis tai res24:

- `0.001 -> 0.0005`: tangential `29.21%`
- `0.0005 -> 0.00025`: tangential `19.14%`
- `0.00025 -> 0.0001`: tangential `10.27%`
- `0.0001 -> 0.00005`: tangential `4.97%`
- `0.00005 -> 0.000025`: tangential `1.54%`, normal `0.42%`

Report:
`data/uipc/conv_realistic_eps_axis_vtol001/convergence_report.json`.

Do do, lua chon cu `eps_velocity=0.00025` da bi loai. Default trong hai
infra scripts da duoc doi sang contact config moi.

Production-shaped smoke:

- artifact:
  `data/uipc/sweep_realistic_phase0_ev000025/combo_000/frame_000/uipc_gt_shear_avg.npz`
- one realistic random frame, depth `0.614 mm`, mode partial
- `K=3`
- replicate noise: tangential `1.02%`, normal `0.10%`
- mean single-solve time `11.79 s`; K=3 sum `35.38 s`
- provenance co `gel_res`, `eps_velocity`, `velocity_tol`, `d_hat`,
  `contact_resistance`, `marker_sampling`, `n_replicates`

Regime check tai chinh chosen config:

- artifact root: `data/uipc/realistic_phase0/chosen_eps_regimes`
- normal: mode 0, shear `0 mm`
- stick: mode 1, shear `0.12 mm`
- partial: mode 2, shear `0.42 mm`
- full: mode 3, shear `0.69 mm`

Mesh gate truoc do van la:

- K=3 averaged res24 -> res28: tangential `0.95%`, normal `1.18%`
- K=3 res24 replicate noise: tangential `1.56%`
- K=3 res28 replicate noise: tangential `2.18%`
- res20 -> res24 khong sach (`~12%`), nen res20 qua coarse cho thin gel;
  res24 -> res28 moi la plateau co y nghia.

`infra/gen_uipc_sweep.sh` bay gio co `OUT_DATA` override de Phase-0 smoke
khong ghi nham vao final production filename. File final 1-frame tao tam trong
smoke da bi xoa; `data/uipc/shear_res24_avg_swept_REALISTIC.npz` hien khong ton
tai, dung nhu trang thai truoc Phase 1.

Checkpoint tiep theo: user review Phase 0 truoc khi launch detached Phase 1.
Lenh production de xuat:

```bash
rtk proxy nohup env \
  SWEEP_DIR=data/uipc/sweep_realistic \
  OUT_DATA=data/uipc/shear_res24_avg_swept_REALISTIC.npz \
  GEL_RES=24 EPS_VELOCITY=0.000025 D_HAT=0.0001 \
  CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.001 \
  bash infra/gen_uipc_sweep.sh 63 40 3 \
  > logs/uipc_sweep_realistic.log 2>&1 &
```

## Goal ban dau

Doc va thuc hien `codex/TASK_realistic_geometry_reground.md`.

Muc tieu lon:
- Regenerate IPC/UIPC ground truth tren hinh hoc gel realistic cho VBTS:
  - gel footprint `20 x 20 mm`
  - gel thickness `3 mm`
  - representative, sensor-agnostic, khong gan voi DIGIT/GelSight cu the
- Phase 0 phai validate stability truoc khi chay full production sweep.
- Phase 1 tao final npz moi:
  - `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  - khong overwrite `data/uipc/shear_res24_avg_swept.npz`
- Phase 2 rerun all downstream voi npz realistic.
- Phase 3 update paper `docs/kse2026/main.tex` va rebuild.
- User yeu cau nen dung subagents de lam song song; da dung subagents cho codebase inspection va downstream patch plan.

## File da sua

Working tree hien tai co cac file modified:

- `infra/gen_uipc_convergence.sh`
  - Chuyen default convergence sang realistic geometry.
  - Them env overrides: `GEL_XY`, `GEL_Z`, `INDENTOR_R`, `MU`, `YOUNGS`, `DEPTH`, `SHEAR`, `DRIVE_RATIO`, `D_HAT`, `CONTACT_RESISTANCE`, `VELOCITY_TOL`, `RES_LEVELS_STR`, `EPS_LEVELS_STR`, `CONV_DIR`.
  - Them `chown` sau container run de tranh root-owned output.

- `infra/gen_uipc_sweep.sh`
  - Chuyen sweep output sang `data/uipc/sweep_realistic`.
  - Scale realistic:
    - depth `U(0.00015, 0.00075)`
    - radius `U(0.002, 0.006)`
    - shear scale `g * mu * 0.001`
  - Them final aggregate ra `data/uipc/shear_res24_avg_swept_REALISTIC.npz`.
  - Them env overrides: `GEL_RES`, `EPS_VELOCITY`, `D_HAT`, `CONTACT_RESISTANCE`, `VELOCITY_TOL`, `GEL_XY`, `GEL_Z`, `SWEEP_DIR`.

- `src/novbts/groundtruth/tacex_uipc_extract_shear.py`
  - Doi marker sampling tren top face tu nearest-neighbor KDTree sang bilinear interpolation tren structured top grid.
  - Neu top grid khong rectangular thi fallback ve nearest.
  - Luu provenance `marker_sampling` vao npz.

- `src/novbts/groundtruth/aggregate_uipc_replicates.py`
  - So sanh string-safe trong `_same`.
  - Check/copy `marker_sampling` qua averaged frames va final sweep aggregate.

- `src/novbts/groundtruth/aggregate_uipc_convergence.py`
  - Assert cac convergence fields paired dung:
    - `coords`
    - `params`
    - `mode`
    - `d_hat`
    - `contact_resistance`
    - `marker_sampling`
  - Luu `marker_sampling` vao convergence report.

- `src/novbts/operator/fem_benchmark.py`
  - Load GT provenance tu selected `--data` npz.
  - Solver FPS lay tu `solve_time_s` trong input npz, khong scan stale FEM/PhysX files nua.
  - Them `gt_path`, `gt_provenance`, data-derived `param_box`.
  - Them generic `gt_solver`, giu alias `physx_fem_shear_solver` de compat.

- `src/novbts/report/make_kse_figs.py`
  - Figure generation prefer `throughput_fps["gt_solver"]`, fallback alias cu.

## Phase 0 artifacts va ket qua hien tai

Artifacts chinh da tao:

- Existing/single checks:
  - `data/uipc/realistic_phase0/smoke/uipc_gt_shear.npz`
  - `data/uipc/realistic_phase0/singles/{normal,stick,partial,full}/uipc_gt_shear.npz`

- Convergence/probe dirs:
  - `data/uipc/conv_realistic/`
  - `data/uipc/conv_realistic_ev0005/`
  - `data/uipc/conv_realistic_ev0005_dhat0002/`
  - `data/uipc/conv_realistic_ev00025/`
  - `data/uipc/conv_realistic_ev0001/`
  - `data/uipc/conv_realistic_bilinear_ev00025/`
  - `data/uipc/conv_realistic_bilinear_ev00025_depth030/`
  - `data/uipc/conv_realistic_bilinear_ev00025_depth030_res24_28/`
  - `data/uipc/conv_realistic_bilinear_ev00025_depth030_vtol001/`

- K-replicate checks:
  - `data/uipc/realistic_phase0/krep_ev0005/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_res24_28/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_vtol001_res20/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_vtol001_res24/`
  - `data/uipc/realistic_phase0/krep_bilinear_ev00025_depth030_vtol001_res28/`

Important numbers:

- 4 regime single checks at realistic geometry passed and produced valid npz:
  - normal mode 0
  - stick mode 1
  - partial mode 2
  - full mode 3

- Earlier default-ish settings failed convergence:
  - `eps=0.001`, `d_hat=0.0001`, `depth=0.45mm`: res20->24 tangential rel-L2 about `42%`.
  - `eps=0.0005`: improved but still about `11.5%`.

- Bilinear marker sampling alone did not fix convergence:
  - `eps=0.00025`, `depth=0.45mm`: res20->24 tangential about `11.4%`.

- Shallower validation point helped:
  - `depth=0.30mm`, `eps=0.00025`, bilinear: res20->24 tangential about `3.77%`, normal about `11.6%`.

- Tight velocity tolerance was the key for repeatability:
  - `depth=0.30mm`, `eps=0.00025`, `d_hat=0.0001`, `velocity_tol=0.001`, bilinear.
  - K=3 res24:
    - tangential rep noise `1.56%`
    - normal rep noise `0.42%`
  - K=3 res28:
    - tangential rep noise `2.18%`
    - normal rep noise `0.43%`
  - K=3 averaged res24->28:
    - overall `1.14%`
    - normal `1.18%`
    - tangential `0.95%`
  - K=3 averaged res20->24:
    - overall `11.70%`
    - normal `11.63%`
    - tangential `12.05%`

Interpretation hien tai:
- `res20` qua coarse cho thin-layer geometry.
- `res24` va `res28` co ve da vao plateau khi dung K=3 + `velocity_tol=0.001`.
- Phase 0 co dau hieu tot o `res24->28`, nhung chua xong eps-axis test.

## Loi/test hien tai

Verification da chay thanh cong:
- `py_compile` cho cac Python files modified:
  - `tacex_uipc_extract_shear.py`
  - `aggregate_uipc_replicates.py`
  - `aggregate_uipc_convergence.py`
  - `fem_benchmark.py`
  - `make_kse_figs.py`
- `bash -n` cho:
  - `infra/gen_uipc_convergence.sh`
  - `infra/gen_uipc_sweep.sh`
- `rtk docker ps` sau interrupt gan day: khong co container dang chay.

Known issue / dang dang do:
- Eps-axis convergence run vua bi user interrupt, chua tao artifact:
  - intended dir: `data/uipc/conv_realistic_eps_axis_vtol001`
  - `rtk find data/uipc/conv_realistic_eps_axis_vtol001` luc do tra `0`.
- Chua chay production sweep Phase 1.
- Chua tao final realistic npz.
- Chua rerun downstream Phase 2.
- Chua update/rebuild paper Phase 3.

Possible gotcha:
- Some shell one-liners truoc do bi quote issue khi print Python f-string, nhung data aggregate van OK.
- Container writes root-owned files; scripts now chown, but manual docker probes phai nho chown.

## Viec con lai

Immediate next steps:

1. Hoan tat eps-axis convergence tai res24 voi chosen Phase-0 knobs:

```bash
rtk proxy env CONV_DIR=data/uipc/conv_realistic_eps_axis_vtol001 \
  GEL_XY=0.020 GEL_Z=0.003 INDENTOR_R=0.004 MU=0.6 YOUNGS=1.0e5 \
  DEPTH=0.00030 SHEAR=0.00036 DRIVE_RATIO=0.60 \
  D_HAT=0.0001 CONTACT_RESISTANCE=1.0e9 VELOCITY_TOL=0.001 \
  RES_LEVELS_STR="24" EPS_LEVELS_STR="0.001 0.0005 0.00025 0.0001" \
  bash infra/gen_uipc_convergence.sh

rtk proxy .venv-gate2/bin/python -m novbts.groundtruth.aggregate_uipc_convergence \
  --conv-dir data/uipc/conv_realistic_eps_axis_vtol001
```

2. Neu eps-axis sach, chot Phase 0 params:
   - `gel_res=24`
   - `gel_xy=0.020`
   - `gel_z=0.003`
   - bilinear marker sampling
   - `eps_velocity=0.00025`
   - `d_hat=0.0001`
   - `velocity_tol=0.001`
   - `contact_resistance=1.0e9`
   - consider production depth range:
     - original task says `0.15-0.75mm`
     - Phase 0 validation is cleanest at `0.30mm`; 0.45mm showed mesh drift at res20->24, but res24->28/K=3 with tight tol was good at 0.30mm.
     - Need decide whether to keep full original depth range or narrow upper depth before full sweep.

3. Optional but useful before full production:
   - production smoke:

```bash
rtk proxy env SWEEP_DIR=data/uipc/sweep_realistic_phase0 \
  GEL_RES=24 EPS_VELOCITY=0.00025 D_HAT=0.0001 VELOCITY_TOL=0.001 \
  bash infra/gen_uipc_sweep.sh 1 1 3
```

4. When Phase 0 accepted, run Phase 1 detached with `nohup`, not Codex background:

```bash
rtk proxy nohup env \
  SWEEP_DIR=data/uipc/sweep_realistic \
  GEL_RES=24 EPS_VELOCITY=0.00025 D_HAT=0.0001 VELOCITY_TOL=0.001 \
  bash infra/gen_uipc_sweep.sh 63 40 3 \
  > logs/uipc_sweep_realistic.log 2>&1 &
```

Expected final output:
- `data/uipc/shear_res24_avg_swept_REALISTIC.npz`

5. After final npz exists:
   - load and count frames/classes
   - verify all 4 regimes populated
   - verify K=3 averaged
   - verify solve-time provenance

6. Phase 2 downstream reruns with:
   - `--data data/uipc/shear_res24_avg_swept_REALISTIC.npz`
   - backup old JSONs as `*.PRE_REALISTIC.json` before overwrite.

7. Phase 3 paper update:
   - update `docs/kse2026/main.tex`
   - remove stale claims/numbers:
     - `0.975`
     - `0.341`
     - `50\times50`
     - `490/430`
     - `flat-punch`
   - rebuild `pdflatex -> bibtex -> pdflatex -> pdflatex`.

## Tieu chi hoan thanh

Goal chi complete khi tat ca dung:

- Phase 0:
  - realistic geometry confirmed:
    - `20 x 20 x 3 mm`
    - structured tet box only
  - smoke/single/convergence clean enough.
  - chosen params documented:
    - `gel_res`
    - `eps_velocity`
    - `d_hat`
    - `velocity_tol`
    - `contact_resistance`
    - depth/shear scales
  - all 4 regimes populated in checks.

- Phase 1:
  - final npz exists and loads:
    - `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  - N about `2520`
  - all 4 regimes populated
  - K=3 averaged
  - no overwrite of old `data/uipc/shear_res24_avg_swept.npz`

- Phase 2:
  - all downstream JSONs regenerated with realistic npz:
    - `runs/phase3_fem/benchmark.json`
    - `runs/phase3_fem/vbts_baselines.json`
    - `runs/phase4/policy_servo.json`
    - `runs/phase5/sensor_build.json`
    - `runs/phase5/sensor_inverse_multiframe.json`
    - `runs/phase6/env_demo.json`
  - provenance points to realistic npz.
  - figures regenerated:
    - `docs/kse2026/figs/fidelity_speed.png`
    - `docs/kse2026/figs/policy_servo_curve.png`
    - `docs/kse2026/figs/sensor_gt_vs_fno.png`

- Phase 3:
  - paper uses real new geometry `20 x 20 x 3 mm`.
  - false `1.3%` validator claim removed/reframed.
  - speedup uses real IPC `solve_time_s` from realistic npz.
  - flat-punch OOD claim removed unless a real flat-punch dataset is generated.
  - RQ5 reports per-regime cosine/round-trip on realistic data.
  - limitations updated to sim-to-real/no physical sensor, not old too-large gel.
  - build has 0 undefined refs.

- Final acceptance grep clean:

```bash
rtk grep "0.975|0.341|50\\\\times50|490/430|flat-punch" docs/kse2026 src runs
```

## 2026-06-30 report reground cleanup

- Rewrote current phase/report markdowns to use only realistic IPC/UIPC metrics:
  - `docs/bao_cao_giai_doan3_rq_results.md`
  - `docs/bao_cao_tong_ket_phase3-6.md`
  - `docs/bao_cao_phase7_gipc_convergence.md`
  - `docs/kse2026/referee_report.md`
- Removed generated stale report artifacts from `docs/`:
  - `docs/bao_cao_tong_ket_phase3-6.pptx`
  - `docs/bao_cao_tong_ket_phase3-6.pdf`
  - `docs/bao_cao_giai_doan3.pdf`
  - `docs/bao_cao_phase7_gipc_convergence.pdf`
  - `docs/bao_cao_giai_doan5_sensor.pdf`
- Removed untracked pre-realistic figure backups and rehearsal figure copies after current figures were verified.
- Current root-level report files now point to:
  - dataset `data/uipc/shear_res24_avg_swept_REALISTIC.npz`
  - N=2520, K=3, gel `20 x 20 x 3 mm`
  - Phase 3 FNO `0.041 / 1.6 deg`, MLP `0.501 / 83.9 deg`, speedup `83204x`
  - Phase 4 autograd vs ES `300` vs `19200` forward evals
  - Phase 5 multi-frame inverse `15.51% / 3.79 deg`
  - Phase 6 reward gap closed `99.93%`, gradcheck diagnostic `passed=false`
- Verification:
  - stale headline grep over current reports is clean for old numeric literals.
  - `infra/verify_realistic_reground.py` passes `REALISTIC_REGROUND_ACCEPTANCE_OK`.
