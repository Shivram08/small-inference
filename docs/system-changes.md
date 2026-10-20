# System changes

Every change to the machine's state made for this project (GPU clocks, power settings), who made it, and how to undo it. Settings that need admin access are made by the user only.

| When | Change | Made by | Undo |
|---|---|---|---|
| 2026-10-19 ~18:20 | GPU SM clock locked at 1,927 MHz (`nvidia-smi -lgc 1927,1927`, admin PowerShell on Windows), so batch-1 decode timings don't depend on the driver's idle heuristic | user | `nvidia-smi -rgc` (admin PowerShell) |
| 2026-10-20 06:46 | **Clock lock ended**: Windows Update (`MoUsoCoreWorker.exe`, then `TrustedInstaller.exe` twice) restarted the laptop to install updates, interrupting the item-11b rerun in repetition 2. After the reboot: NVIDIA driver unchanged (617.42), enforced power limit 78.5 W, SM clock unlocked. Not caused by this project | Windows Update | re-lock with `nvidia-smi -lgc 1927,1927` (admin PowerShell) before further timing runs |
