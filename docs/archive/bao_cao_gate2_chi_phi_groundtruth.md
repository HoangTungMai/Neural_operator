# Bao cao Gate 2 - Chi phi sinh ground-truth vat ly

Ngay kiem tra: 2026-06-05

## Ket luan ngan

Trang thai dieu kien con lai cua Gate 2: **GO CO DIEU KIEN**.

Co 2 nguon bang chung:
- TacEx/UIPC benchmark logs da co san tren may: do duoc chi phi simulator vat ly that cho GelSight Mini/UIPC/FOTS marker motion.
- Physics-proxy benchmark moi chay: elastic half-space Green's function tren GPU de do chi phi sinh field hang loat khi khong qua Isaac/UIPC.

Ket luan thuc dung:
- Neu dung TacEx/UIPC 1 env nhu log hien co, sinh 20k contact frames la kha thi trong khoang **~25 phut den ~1 gio**, tuy cach tinh chi lay contact-frame hay tinh wall-clock ca episode/reset.
- 50k-100k frames la kha thi trong **vai gio**, khong phai nhieu ngay.
- Diem chua xac minh: custom data extractor cho dung bai toan marker displacement VBTS cua ta, voi split contact primitive/material rieng.

## Nguon 1 - TacEx/UIPC logs that tren may

Repo local:
```text
/home/tungmai/CODE/TacEx
```

Benchmark script:
```text
/home/tungmai/CODE/TacEx/scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py
```

Log da co:
```text
/home/tungmai/CODE/TacEx/scripts/benchmarking/tactile_sim_performance/logs/UipcEnv/
```

Moi log:
- GPU: NVIDIA RTX 2000 Ada Generation.
- Env: UipcEnv, `num_envs = 1`.
- Sensor config gom `marker_motion` va `camera_depth`.
- Marker config: 11 x 9 = 99 markers.
- Marker motion simulator: `FOTSMarkerSimulator`.
- Physics/backend: UIPC trong Isaac Lab/TacEx.

Bang ket qua:
```text
envs_1_2026-05-17-16_37_04.txt
- contact frames: 1183
- total wall time: 233.99 s
- physics: 64.67 ms/contact-frame
- tactile: 8.24 ms/contact-frame
- GPU memory: 43.39%

envs_1_2026-05-18-07_49_04.txt
- contact frames: 1186
- total wall time: 151.90 s
- physics: 62.41 ms/contact-frame
- tactile: 9.79 ms/contact-frame
- GPU memory: 43.33%

envs_1_2026-05-27-11_04_20.txt
- contact frames: 1187
- total wall time: 224.99 s
- physics: 61.31 ms/contact-frame
- tactile: 7.69 ms/contact-frame
- GPU memory: 43.55%

envs_1_2026-05-27-11_25_06.txt
- contact frames: 1183
- total wall time: 151.47 s
- physics: 64.29 ms/contact-frame
- tactile: 8.85 ms/contact-frame
- GPU memory: 43.26%
```

Tom tat:
```text
physics + tactile component: ~69-74 ms/contact-frame
component throughput: ~13.5-14.5 contact frames/s
wall-clock throughput, tinh ca episode/reset/render overhead: ~5.1-7.8 contact frames/s
```

Ngoai suy 1 env:
```text
20k contact frames:
- component-only: ~23-25 phut
- wall-clock tu log: ~43-66 phut

50k contact frames:
- component-only: ~58-62 phut
- wall-clock tu log: ~1.8-2.7 gio

100k contact frames:
- component-only: ~1.9-2.1 gio
- wall-clock tu log: ~3.6-5.5 gio
```

Danh gia:
- Kha thi cho PoC va dataset paper-scale nho-vua.
- Chua nen commit dataset rat lon truoc khi co extractor tu dong va chay lai benchmark voi contact primitive dung cua de tai.
- 99 markers thap hon grid 32x32/64x64 trong P0 neural benchmark, nen neu can marker day hon thi chi phi post-processing/data size tang.

## Nguon 2 - Physics-proxy benchmark moi chay

Script:
```text
scripts/gate2_groundtruth_cost_check.py
```

Mo hinh:
- Linear elastic half-space Green's function voi Gaussian normal/shear load patches.
- Sinh displacement field `(u_x, u_y, u_z)` tren marker grid.
- Day la physics-proxy co y nghia vat ly, **khong thay the FEM/GIPC/MPM**.

Ket qua compute-only:
```text
20k frames, 32x32 markers, 8 load patches, batch 512
- generate time: 0.468 s
- throughput: 42,694 frames/s
- raw dataset size: 234 MB
- peak VRAM: 0.178 GiB

20k frames, 64x64 markers, 8 load patches, batch 128
- generate time: 1.044 s
- throughput: 19,159 frames/s
- raw dataset size: 938 MB
- peak VRAM: 0.178 GiB
```

Save test:
```text
2k frames, 32x32 markers, saved npz
- output: data/gate2_proxy_sample_2k_32.npz
- file size: 24 MB
- compute time inside script: 0.090 s
- end-to-end command time: ~1.6 s
```

Danh gia:
- Neu dung analytic/proxy physics de pretrain/smoke-test, data khong phai nut that.
- Nut that nam o Isaac/UIPC/FEM step, contact setup, extraction, va sim-to-real fidelity.

## Quyet dinh Gate 2 sau dieu kien ground-truth

**Gate 2: GO CO DIEU KIEN, nghieng ve GO cho PoC.**

Ly do:
- Training tren 1 GPU da qua P0.
- TacEx/UIPC logs cho thay sinh physical contact frames tren RTX 2000 Ada 16GB la trong tam voi quy mo 20k-100k frames.
- Physics-proxy cho thay pipeline data/neural operator khong bi gioi han boi post-processing field generation.

Dieu kien truoc khi chuyen sang Giai doan 2 that:
1. Tao data extractor tu TacEx/UIPC: output `params`, `coords`, `disp`, optional `mode`.
2. Chay lai benchmark voi primitive dung: normal press, shear stick, press -> shear -> slip.
3. Do frame/hour sau khi tat render/debug neu co the.
4. Xac nhan marker representation: 99 marker GelSight Mini that hay grid noi suy 32x32/64x64.
5. Xac dinh dataset target ban dau: 10k-20k frames cho PoC, sau do 50k-100k neu Gate 3 can.

Rui ro con lai:
- TacEx benchmark hien co la ball rolling, khong phai contact primitive rieng cua de tai.
- Shell hien tai khong co `isaaclab` trong PATH, nen chua chay lai benchmark truc tiep trong phien nay.
- UIPC/TacEx co overhead Isaac/render; can script headless extractor rieng neu muon sinh data sach va nhanh.
