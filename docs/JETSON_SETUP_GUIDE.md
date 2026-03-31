# Jetson Orin Nano Super Developer Kit - Complete Setup Guide

**SKU:** 945-13766-0000-000
**Target:** VLA model deployment with vla-edge toolkit
**Dev machine:** Mac Air M3
**Date:** 2026-03-29

---

## Table of Contents

1. [What's in the Box](#1-whats-in-the-box)
2. [What You Need (Not Included)](#2-what-you-need-not-included)
3. [Critical Knowledge Before You Start](#3-critical-knowledge-before-you-start)
4. [Step 1: Prepare SD Cards on Mac](#step-1-prepare-sd-cards-on-mac)
5. [Step 2: Physical Setup](#step-2-physical-setup)
6. [Step 3: Firmware Update via JetPack 5.1.3](#step-3-firmware-update-via-jetpack-513)
7. [Step 4: QSPI Updater - Bridge to JetPack 6](#step-4-qspi-updater---bridge-to-jetpack-6)
8. [Step 5: Boot JetPack 6.2 and OOBE](#step-5-boot-jetpack-62-and-oobe)
9. [Step 6: Post-Boot Configuration](#step-6-post-boot-configuration)
10. [Step 7: SSH from Mac](#step-7-ssh-from-mac)
11. [Step 8: Dev Environment for VLA](#step-8-dev-environment-for-vla)
12. [Peripherals and CI Runner](#jetson-setup-peripherals)
13. [Troubleshooting](#jetson-setup-troubleshooting)

See also: [JETSON_SETUP_PART2.md](JETSON_SETUP_PART2.md) for peripherals, GitHub Actions runner, and troubleshooting.

---

## 1. What's in the Box

The developer kit (945-13766-0000-000) includes:

- Jetson Orin Nano 8GB module (pre-installed on carrier board)
- Reference carrier board with pre-installed 802.11ac WiFi + BT 5.0 module
- 19V DC power supply with barrel jack connector
- Quick start card

**It does NOT come pre-flashed with JetPack.** You must flash your own microSD card.

Source: [NVIDIA Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)

---

## 2. What You Need (Not Included)

| Item | Spec | Notes |
|------|------|-------|
| microSD card (x2 ideal) | 64GB+, UHS-1, V30, Class 10 | Samsung EVO Select 128GB or SanDisk Extreme. Two cards avoids re-flashing. |
| DP-to-HDMI adapter | Active adapter recommended | Board has DisplayPort ONLY. No HDMI. |
| Monitor | Any with DP or HDMI input | For initial setup only. |
| USB keyboard + mouse | Standard USB-A | For OOBE. |
| Ethernet cable | Cat5e or better | For internet during setup and SSH. |

**CRITICAL: The board has DisplayPort output, NOT HDMI.** Without a DP cable or DP-to-HDMI adapter, you will see nothing on screen. This is the #1 "won't boot" misdiagnosis.

Source: [NVIDIA Forums - DisplayPort](https://forums.developer.nvidia.com/t/jetson-orin-nano-displayport-or-hdmi-port-possible/362244)

---

## 3. Critical Knowledge Before You Start

### The board ships with OLD firmware
Factory firmware is older than v36.0 and is NOT compatible with JetPack 6.x. Booting JetPack 6.2 directly will fail (black screen, boot loop, hang at logo). This is why your previous boot attempts likely failed.

### Two-stage firmware update required
1. Boot JetPack 5.1.3 to update firmware to 35.5.0
2. Install QSPI updater package to bridge firmware to JetPack 6.x
3. Then boot JetPack 6.2

### Power supply
- Uses the **bundled 19V DC barrel jack** (center-positive, 5.5x2.5mm)
- Input range: 7-20V DC
- Do NOT try to power via USB-C. The USB-C port is data only.

### SD card requirements
- Minimum: 64GB, UHS-1, Class 10
- Recommended: 128GB, UHS-1, V30, A2
- Cheap/slow cards cause boot failures and filesystem corruption

Sources: [Jetson AI Lab Setup](https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/), [Matt Dixon Medium](https://medium.com/@matt.dixon1010/jetson-orin-nano-super-developer-kit-initial-setup-fccba1d46b09)

---

## Step 1: Prepare SD Cards on Mac

### 1a. Download both images

**JetPack 5.1.3** (firmware update only):
```
https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v5.0/jp513-orin-nano-sd-card-image.zip
```

**JetPack 6.2** (your final OS):
```
https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/jp62-orin-nano-sd-card-image.zip
```

Both are multi-GB. Download while you read.

Source: [JetPack 5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513), [JetPack 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)

### 1b. Install Balena Etcher

Download from: https://etcher.balena.io/#download-etcher (free, works on Apple Silicon)

### 1c. Flash cards

1. Insert microSD card into Mac (USB adapter if needed)
2. Open Etcher, click "Flash from file", select the `.zip` image
3. Select your SD card as target
4. Click "Flash!" and enter Mac password
5. Wait ~10-15 min for flash + verification
6. Label the card ("JP 5.1.3" or "JP 6.2")

**If you only have ONE card:** Flash 5.1.3 first. Re-flash with 6.2 after the firmware step.

### 1d. Mac command line alternative

```bash
diskutil list external                    # Find your SD card (e.g., /dev/disk4)
diskutil unmountDisk /dev/diskN           # Unmount it
# Flash (replace N):
unzip -p ~/Downloads/jp513-orin-nano-sd-card-image.zip | sudo dd of=/dev/rdiskN bs=1m status=progress
diskutil eject /dev/diskN                 # Eject when done
```

---

## Step 2: Physical Setup

**Follow this exact order:**

1. **Insert JetPack 5.1.3 SD card** into the microSD slot on the **underside** of the board. Push until click.
2. **Connect display** via DisplayPort (or DP-to-HDMI adapter).
3. **Connect USB keyboard and mouse** to USB 3.1 Type-A ports (blue ports).
4. **Connect Ethernet** to router/switch.
5. **Turn on monitor** and select correct input.
6. **Plug in 19V power supply.** Board powers on automatically (no power button).
7. **Watch for green LED** on the board.

---

## Step 3: Firmware Update via JetPack 5.1.3

### What to expect

1. Green LED lights up, fan starts spinning
2. NVIDIA logo appears (~10-30 sec)
3. System boots into Ubuntu

**If black screen for 2+ minutes:** This is normal on first boot. The firmware update is scheduling in the background. Wait 3-5 minutes, unplug power, wait 10 sec, plug back in. The update runs on this boot. Wait another 2-3 min for the OOBE.

### Complete JetPack 5.1.3 OOBE

When the setup wizard appears:
1. Accept NVIDIA EULA
2. Select language, keyboard, timezone
3. Create username and password - **write these down**
4. Wait for initial config (~2-5 min)

### Verify firmware

Open terminal (Ctrl+Alt+T):
```bash
sudo nvbootctrl dump-slots-info
```
Expected: firmware version **5.0-35550185** (or 35.5.0). If still older, reboot and check again.

---

## Step 4: QSPI Updater - Bridge to JetPack 6

This is the step most guides skip. It bridges firmware from JetPack 5 to JetPack 6 compatibility.

```bash
# On the Jetson terminal (still running JetPack 5.1.3):
sudo apt-get update
sudo apt-get install nvidia-l4t-jetson-orin-nano-qspi-updater

# Reboot to apply QSPI update (takes 1-2 min, screen may go blank)
sudo reboot
```

After reboot completes:
```bash
sudo poweroff
```

**Remove the JetPack 5.1.3 SD card. Insert the JetPack 6.2 SD card.**

After this update, the firmware is no longer compatible with JetPack 5.1.3. Do not re-insert the 5.1.3 card.

Source: [Jetson AI Lab QSPI Update](https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/)

---

## Step 5: Boot JetPack 6.2 and OOBE

1. With JetPack 6.2 card inserted, plug in power
2. Wait for NVIDIA logo. There may be 1-2 automatic reboots as firmware finalizes to v36.4.3
3. Ubuntu 22.04 OOBE appears

Complete the setup:
- Language, keyboard, timezone (America/Los_Angeles for SF)
- Computer name: `jetson-vla` (becomes the hostname)
- Username/password: pick and **write down**

### Verify

```bash
cat /etc/nv_tegra_release        # Should show R36
nvcc --version                    # Should show CUDA 12.6
df -h /                           # Check disk space
```

---

## Step 6: Post-Boot Configuration

### Enable MAXN SUPER mode (67 TOPS)

**GUI:** Click NVIDIA icon in top bar > Power Mode > MAXN SUPER

**CLI:**
```bash
sudo nvpmodel -q --verbose        # List available modes
sudo nvpmodel -m 3                # Switch to MAXN SUPER
# If MAXN SUPER not listed:
sudo rm -rf /etc/nvpmodel.conf && sudo reboot
# Then retry: sudo nvpmodel -m 3
```

Source: [NVIDIA Forums - nvpmodel](https://forums.developer.nvidia.com/t/setting-orin-nano-power-mode-via-cli/344753)

### System updates + essentials

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip git curl wget htop nano build-essential \
    cmake pkg-config libopenblas-dev liblapack-dev v4l-utils avahi-daemon
```

### Install jetson-stats (jtop)

```bash
sudo pip3 install -U jetson-stats
sudo reboot
# After reboot: run "jtop" for GPU/CPU/temp/power dashboard
```

Source: [jetson-stats](https://github.com/rbonghi/jetson_stats)

---

## Step 7: SSH from Mac

### Find Jetson IP

On the Jetson:
```bash
ip addr show eth0 | grep "inet "
```

### SSH from Mac

```bash
# By IP
ssh krishnam@192.168.1.105
# By hostname (avahi makes this work)
ssh krishnam@jetson-vla.local
```

### Key-based auth (no password)

On Mac:
```bash
ssh-keygen -t ed25519 -C "kayjee1994@gmail.com"   # if you don't have a key
ssh-copy-id krishnam@jetson-vla.local
```

### Mac SSH config

Add to `~/.ssh/config`:
```
Host jetson
    HostName jetson-vla.local
    User krishnam
    IdentityFile ~/.ssh/id_ed25519
```
Now: `ssh jetson`

### Direct Ethernet (no router)

On Mac: System Settings > Network > Ethernet > IPv4: Manual, IP 192.168.1.1, Subnet 255.255.255.0

On Jetson:
```bash
sudo nmcli connection modify "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.1.2/24
sudo nmcli connection up "Wired connection 1"
```
SSH: `ssh krishnam@192.168.1.2`

---

## Step 8: Dev Environment for VLA

### Python

JetPack 6.2 ships with **Python 3.10**. NVIDIA's wheels target cp310. Stay on 3.10 for Jetson.

```bash
python3 --version                 # Python 3.10.x
python3 -m pip install --upgrade pip
```

### PyTorch (from JPL wheels - NOT pip)

```bash
python3 -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.8.0 True Orin (nvgpu)
```

Source: [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048), [JPL Wheels](https://pypi.jetson-ai-lab.io/jp6/cu126)

### ML dependencies

```bash
pip3 install numpy==1.26.1 transformers huggingface-hub safetensors pillow \
    opencv-python-headless scipy tqdm
```

### llama.cpp (LLM backbone - TensorRT-LLM is BROKEN on Orin Nano)

```bash
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)
```

### vla-edge toolkit

```bash
cd ~ && git clone https://github.com/krishnam94/vla-edge.git && cd vla-edge
pip3 install -e ".[dev]"
vla-edge check && vla-edge version
```

### Torch-TensorRT (vision encoders)

```bash
pip3 install torch-tensorrt --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

Continue to [JETSON_SETUP_PART2.md](JETSON_SETUP_PART2.md) for peripherals, CI runner, and troubleshooting.

---

## Quick Reference

| What | Value |
|------|-------|
| Board | Jetson Orin Nano Super 8GB |
| SKU | 945-13766-0000-000 |
| AI Performance | 67 TOPS (MAXN SUPER) |
| GPU | Ampere, 1024 CUDA cores |
| CPU | 6-core ARM Cortex-A78AE |
| RAM | 8GB LPDDR5 |
| Power | 19V DC barrel jack (bundled) |
| Display | DisplayPort only (no HDMI) |
| USB | 4x USB 3.1 Type-A |
| Network | Gigabit Ethernet + WiFi 802.11ac |
| OS | Ubuntu 22.04 (JetPack 6.2) |
| CUDA | 12.6 |
| Python | 3.10 |
| PyTorch | 2.8.0 (JPL wheels) |
