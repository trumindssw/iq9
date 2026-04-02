#!/bin/bash

# =============================================================================
# build-and-deploy-stmmac.sh
# copy this modyule at this path - "/home/roopak/Documents/iq9/build-qcom-wayland/tmp-glibc/work-shared/qcs9075-iq-9075-evk/kernel-source"
# Build stmmac kernel module and copy to IQ9 device
# Usage: ./build-and-deploy-stmmac.sh
# =============================================================================

set -e  # exit on any error

# ── Colors for output ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR=$(pwd)
KERNEL_BUILD=${SCRIPT_DIR}/../../../work/qcs9075_iq_9075_evk-qcom-linux/linux-qcom-custom/6.6/build
STMMAC_SRC=${SCRIPT_DIR}/../../../work/qcs9075_iq_9075_evk-qcom-linux/linux-qcom-custom/6.6/kernel/drivers/net/ethernet/stmicro
STMMAC_KO=${STMMAC_SRC}/stmmac/stmmac.ko

# ── Device details ────────────────────────────────────────────────────────────
DEVICE_IP="192.168.101.4"
DEVICE_USER="root"
DEVICE_PATH="/root/"

# ── Print banner ──────────────────────────────────────────────────────────────
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}   STMMAC Module Build and Deploy Script    ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# ── Step 1: Verify paths ──────────────────────────────────────────────────────
echo -e "${YELLOW}[Step 1] Verifying paths...${NC}"

if [ ! -d "${KERNEL_BUILD}" ]; then
    echo -e "${RED}ERROR: Kernel build directory not found:${NC}"
    echo "       ${KERNEL_BUILD}"
    exit 1
fi
echo -e "${GREEN}  ✓ Kernel build dir : ${KERNEL_BUILD}${NC}"

if [ ! -d "${STMMAC_SRC}" ]; then
    echo -e "${RED}ERROR: STMMAC source directory not found:${NC}"
    echo "       ${STMMAC_SRC}"
    exit 1
fi
echo -e "${GREEN}  ✓ STMMAC source dir: ${STMMAC_SRC}${NC}"
echo ""

# ── Step 2: Build the module ──────────────────────────────────────────────────
echo -e "${YELLOW}[Step 2] Building stmmac kernel module...${NC}"
echo "         Running: make -C ${KERNEL_BUILD} M=${STMMAC_SRC} modules"
echo ""

make -C ${KERNEL_BUILD} M=${STMMAC_SRC} modules

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed!${NC}"
    exit 1
fi
echo ""
echo -e "${GREEN}  ✓ Build successful!${NC}"
echo ""

# ── Step 3: Verify .ko was created ───────────────────────────────────────────
echo -e "${YELLOW}[Step 3] Verifying stmmac.ko...${NC}"

if [ ! -f "${STMMAC_KO}" ]; then
    echo -e "${RED}ERROR: stmmac.ko not found at:${NC}"
    echo "       ${STMMAC_KO}"
    exit 1
fi

# Show module info
echo -e "${GREEN}  ✓ Module found: ${STMMAC_KO}${NC}"
echo "  Size: $(du -h ${STMMAC_KO} | cut -f1)"
echo "  Type: $(file ${STMMAC_KO} | cut -d: -f2)"
echo ""

# ── Step 4: Copy to device ────────────────────────────────────────────────────
echo -e "${YELLOW}[Step 4] Copying stmmac.ko to device ${DEVICE_IP}...${NC}"
echo "         Running: scp ${STMMAC_KO} ${DEVICE_USER}@${DEVICE_IP}:${DEVICE_PATH}"
echo ""

scp ${STMMAC_KO} ${DEVICE_USER}@${DEVICE_IP}:${DEVICE_PATH}

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: SCP failed! Check:${NC}"
    echo "  1. Device is powered on and connected"
    echo "  2. IP address is correct: ${DEVICE_IP}"
    echo "  3. SSH is enabled on device"
    echo "  4. Network connection is active"
    exit 1
fi

echo ""
echo -e "${GREEN}  ✓ Module copied successfully to ${DEVICE_USER}@${DEVICE_IP}:${DEVICE_PATH}${NC}"
echo ""
