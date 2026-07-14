#!/usr/bin/env bash
# REGEN v2 — GT hoi tu (quyet dinh user 2026-07-13, "regen day du").
#   res32 + gel_nz24 + subdiv4 + d_hat 2.5e-5 (GHEP DOI: d_hat/sagitta ~7.5)
#   + velocity_tol 1e-5 (=> tat dinh, K=1) + dt 0.005 (80/20/160/20)
# Preflight do: 109.3 s/frame -> 2520 frame = 76.5h = 3.19 ngay.
# Sai lech hinh dang vs reference tot nhat: ~5% (GT cu: 24-30%).
# Resumable: combo da xong duoc SKIP.
cd "/home/tungmai/CODE/Neural operator"
export SWEEP_DIR=data/uipc/sweep_v2_converged
export OUT_DATA=data/uipc/shear_res32_nz24_sd4_converged.npz
export GEL_RES=32 GEL_NZ=24 INDENTOR_SUBDIV=4 D_HAT=2.5e-5
export EPS_VELOCITY=0.000025 VELOCITY_TOL=0.00001 CONTACT_RESISTANCE=1.0e9
export GEL_BOTTOM_BC=fixed GEL_CONSTRAINT_STRENGTH=0 INDENTOR_CONSTRAINT_STRENGTH=30000
export DT=0.005 PRESS_STEPS=80 SETTLE_STEPS=20 SHEAR_STEPS=160 SHEAR_SETTLE=20
export GEL_XY=0.020 GEL_Z=0.003
exec bash infra/gen_uipc_sweep.sh 63 40 1
