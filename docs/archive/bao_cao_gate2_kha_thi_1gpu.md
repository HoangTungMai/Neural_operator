# Bao cao Gate 2 - Kha thi 1 GPU

Ngay kiem tra: 2026-06-05

## Ket luan ngan

Trang thai Gate 2: **GO CO DIEU KIEN, NGHIENG VE GO CHO POC**.

Ly do: da cai duoc PyTorch CUDA rieng trong `.venv-gate2` va chay P0 synthetic marker-field tren RTX 2000 Ada 16GB. DeepONet nho train duoc voi 20k frames, marker grid 32x32 va 64x64, VRAM thap hon gioi han 16GB.

Chi phi sinh ground-truth vat ly da duoc kiem tra bo sung trong `bao_cao_gate2_chi_phi_groundtruth.md`. TacEx/UIPC logs tren may cho thay 20k-100k contact frames kha thi trong vai gio voi 1 env; can tao extractor rieng de bien benchmark nay thanh dataset dung cho de tai.

## Moi truong

GPU/driver:
```text
GPU: NVIDIA RTX 2000 Ada Generation
VRAM: 16380 MiB
Driver: 580.159.03
CUDA reported by nvidia-smi: 13.0
```

Python he thong van la CPU-only:
```text
python: /usr/bin/python3
torch: 2.11.0+cpu
torch.cuda.is_available(): False
torch.version.cuda: None
```

Moi truong CUDA rieng:
```text
.venv-gate2
torch: 2.12.0+cu130
torch.version.cuda: 13.0
torch.cuda.is_available(): True
GPU: NVIDIA RTX 2000 Ada Generation
VRAM visible to PyTorch: 15.565 GiB
```

Dung moi truong nay:
```bash
source .venv-gate2/bin/activate
```

## Script P0

Script:
```text
scripts/gate2_p0_gpu_check.py
```

Script nay:
- Sinh synthetic marker displacement field co normal/shear/slip-like transition.
- Train `MLP coordinate regressor` hoac `TinyDeepONet`.
- Do train time, inference throughput, VRAM peak.
- In ket qua JSON.

## Ket qua P0

```text
MLP, 4096 frames, 32x32 markers, batch 128, 5 epochs
- train time: 4.76 s
- train time/epoch: 0.95 s
- val MSE: 0.00327
- inference: 17,356 frames/s
- peak VRAM: 1.33 GiB

DeepONet, 4096 frames, 32x32 markers, batch 128, 5 epochs
- train time: 3.17 s
- train time/epoch: 0.63 s
- val MSE: 0.00326
- inference: 16,819 frames/s
- peak VRAM: 0.90 GiB

DeepONet, 20,000 frames, 32x32 markers, batch 128, 10 epochs
- train time: 26.01 s
- train time/epoch: 2.60 s
- val MSE: 0.000896
- inference: 17,977 frames/s
- peak VRAM: 4.31 GiB

DeepONet, 20,000 frames, 64x64 markers, batch 32, 10 epochs
- train time: 57.43 s
- train time/epoch: 5.74 s
- val MSE: 0.000470
- inference: 8,356 frames/s
- peak VRAM: 1.91 GiB
```

Luu y:
- Benchmark dung synthetic field, chua phai ground-truth FEM/GIPC/MPM.
- VRAM cua 64x64 thap hon 32x32 vi batch size giam tu 128 xuong 32.
- Script ban dau OOM o validation 64x64 vi evaluate toan bo validation set mot lan; da sua thanh validation theo batch.

## Lenh tai lap

Kiem tra CUDA:
```bash
.venv-gate2/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Benchmark:
```bash
.venv-gate2/bin/python scripts/gate2_p0_gpu_check.py --model mlp --frames 4096 --marker-side 32 --batch-size 128 --epochs 5
.venv-gate2/bin/python scripts/gate2_p0_gpu_check.py --model deeponet --frames 4096 --marker-side 32 --batch-size 128 --epochs 5
.venv-gate2/bin/python scripts/gate2_p0_gpu_check.py --model deeponet --frames 20000 --marker-side 32 --batch-size 128 --epochs 10
.venv-gate2/bin/python scripts/gate2_p0_gpu_check.py --model deeponet --frames 20000 --marker-side 64 --batch-size 32 --epochs 10
```

## Quyet dinh

GO:
- PyTorch CUDA chay duoc tren RTX 2000 Ada 16GB. Dat.
- DeepONet nho train duoc voi 4096-20000 frames, marker grid 32x32. Dat.
- DeepONet nho train duoc voi 20k frames, marker grid 64x64 bang batch nho hon. Dat.
- Peak VRAM < 12GB o PoC synthetic. Dat.
- Inference nhanh hon muc can cho RL surrogate toy loop. Dat voi synthetic benchmark.

Ket luan: **tien de 2 duoc xac minh o muc compute/training PoC**. 1 GPU 16GB du kha thi cho DeepONet nho tren marker-field synthetic 32x32-64x64.

Chi phi ground-truth:
- TacEx/UIPC logs that tren may: physics + tactile khoang 69-74 ms/contact-frame voi 1 env, 99 markers.
- Ngoai suy 20k contact frames: khoang 23-25 phut component-only, hoac 43-66 phut wall-clock theo log.
- Ngoai suy 100k contact frames: khoang 1.9-2.1 gio component-only, hoac 3.6-5.5 gio wall-clock theo log.
- Physics-proxy half-space tren GPU sinh 20k frames 32x32 trong 0.47s va 64x64 trong 1.04s, nen post-processing field khong phai nut that.

Dieu kien con lai truoc khi chuyen thanh GO manh:
- Tao TacEx/UIPC extractor that: `params`, `coords`, `disp`, optional `mode`.
- Chay lai voi contact primitive dung cua de tai: normal press, shear stick, press -> shear -> slip.
- Do data loader/train voi dataset that, khong chi synthetic/proxy.
