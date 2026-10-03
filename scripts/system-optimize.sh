#!/bin/bash
# AMD RDNA4 system-level tweaks for LLM inference
# Run with sudo / 需要 sudo 执行
# Reference: https://github.com/ggml-org/llama.cpp/discussions/21043
#
# !! MEASURED ON AN R9700 (2026-10-03): these are NOT a free win.
#    Decode +13..+20%, but prefill -9..-16% (peak pp 264 -> 229 t/s @247K, i.e.
#    15.6 -> 18.0 min to ingest a full 262144 window). The decode gain comes from
#    power_dpm_force_performance_level=high alone; ASPM contributes nothing
#    measurable on this box. Skip this script if you ingest long prompts.
#    See README section "System-level".

set -e

echo "[1/2] Setting PCIe ASPM to performance mode (+~10% decode)..."
echo performance | tee /sys/module/pcie_aspm/parameters/policy

echo "[2/2] Setting GPU power level to high (stable clocks)..."
# Try card1 first, fall back to card0
if [ -f /sys/class/drm/card1/device/power_dpm_force_performance_level ]; then
    echo high | tee /sys/class/drm/card1/device/power_dpm_force_performance_level
elif [ -f /sys/class/drm/card0/device/power_dpm_force_performance_level ]; then
    echo high | tee /sys/class/drm/card0/device/power_dpm_force_performance_level
fi

echo ""
echo "Done. Verify with:"
echo "  cat /sys/module/pcie_aspm/parameters/policy"
echo "  cat /sys/class/drm/card1/device/power_dpm_force_performance_level"
echo ""
echo "Note: These settings reset on reboot."
echo "To persist, add this script to /etc/rc.local or a systemd service."
