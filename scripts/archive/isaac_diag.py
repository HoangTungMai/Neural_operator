import sys, glob, importlib, traceback
out = []
def log(*a): out.append(" ".join(str(x) for x in a)); open("/work/diag_result.txt","w").write("\n".join(out)+"\n")

log("python:", sys.executable)
log("version:", sys.version.split()[0])
for name in ["isaaclab", "omni.isaac.lab", "isaaclab.app", "omni.isaac.lab.app",
             "isaacsim", "omni.isaac.core", "isaacsim.core.api"]:
    try:
        m = importlib.import_module(name)
        log(f"IMPORT OK: {name}  ->  {getattr(m,'__file__','?')}")
    except Exception as e:
        log(f"IMPORT FAIL: {name}  ->  {type(e).__name__}: {e}")

log("--- site-packages dirs containing 'lab' or 'isaac' ---")
for p in sys.path:
    for pat in ["*isaaclab*", "*isaac*lab*", "*isaac_lab*"]:
        for hit in glob.glob(p + "/" + pat)[:5]:
            log("  ", hit)
log("--- /workspace ---")
for hit in glob.glob("/workspace/*")[:30]:
    log("  ", hit)
log("DIAG_DONE")
