# Jetson Setup - Peripherals, CI Runner, and Troubleshooting

Continuation of [JETSON_SETUP_GUIDE.md](JETSON_SETUP_GUIDE.md).

---

## Step 9: Connect Peripherals

### USB Camera (VLA image input)

1. Plug USB camera into a USB 3.1 Type-A port (blue)
2. Verify:
```bash
v4l2-ctl --list-devices                          # List video devices
v4l2-ctl -d /dev/video0 --list-formats-ext       # Camera capabilities

python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Captured: {ret}, shape: {frame.shape if ret else None}')
cap.release()
"
```

### USB Microphone

```bash
arecord -l                        # List audio input devices
arecord -d 5 -f cd test.wav      # Test 5-second recording
```

### USB Speakers

```bash
aplay -l                          # List audio output devices
aplay test.wav                    # Test playback
```

### Set default audio devices

```bash
pactl list short sources          # List inputs
pactl list short sinks            # List outputs
pactl set-default-source <name>   # Set default mic
pactl set-default-sink <name>     # Set default speaker
```

---

## Step 10: GitHub Actions Self-Hosted Runner

This lets vla-edge CI run GPU and Jetson tests on actual hardware.

### Create runner on GitHub

1. Go to https://github.com/krishnam94/vla-edge/settings/actions/runners
2. Click "New self-hosted runner"
3. Select Linux, ARM64
4. Copy the token from the displayed commands

### Install on Jetson

```bash
mkdir ~/actions-runner && cd ~/actions-runner

# Download ARM64 runner (check GitHub for latest version)
curl -o actions-runner-linux-arm64.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.333.1/actions-runner-linux-arm64-2.333.1.tar.gz

tar xzf ./actions-runner-linux-arm64.tar.gz

# Configure (paste your token)
./config.sh --url https://github.com/krishnam94/vla-edge --token YOUR_TOKEN

# Prompts:
# Runner group: Enter (default)
# Runner name: jetson-orin-nano
# Labels: self-hosted,Linux,ARM64,jetson,gpu
# Work folder: Enter (default)
```

### Install as auto-start service

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

### Use in workflows

```yaml
jobs:
  jetson-tests:
    runs-on: [self-hosted, jetson]
    steps:
      - uses: actions/checkout@v4
      - name: Run Jetson tests
        run: pytest -m jetson
```

Source: [GitHub Docs - Self-hosted runners](https://docs.github.com/en/actions/reference/runners/self-hosted-runners), [Arm Learning Paths](https://learn.arm.com/learning-paths/laptops-and-desktops/self_hosted_cicd_github/create-self-hosted-runner-github/)

---

## Troubleshooting Guide

### No display output at all

**Symptoms:** Green LED lit, fan spins, nothing on monitor.

**Fixes:**
1. **Wrong port.** Board has DisplayPort, NOT HDMI. Get a DP-to-HDMI adapter (active recommended).
2. **Monitor input.** Manually select DP or HDMI input on monitor.
3. **Try different monitor.** Some don't auto-detect DP.
4. **Wait longer.** First boot can be 2-3 min of black screen during firmware update.

Source: [NVIDIA Forums](https://nvidia-jetson.piveral.com/jetson-orin-nano/jetson-orin-nano-wont-boot-wont-even-display-anything/)

### NVIDIA logo then black screen with blinking cursor

**Symptoms:** Splash screen shows, then black with cursor.

**Fixes:**
1. **Skipped firmware update.** This is the #1 cause. If you booted JetPack 6.2 without the 5.1.3 firmware bridge, go back to Step 3 in the main guide.
2. **Bad SD card.** Re-flash with Etcher (it verifies writes).
3. **Wait 5 minutes.** First boot is slow.
4. **Power cycle.** Unplug, wait 10 sec, replug.

Source: [NVIDIA Forums](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-black-screen-with-blinking-cursor-after-nvidia-logo-jetpack-6-2-sd/347871)

### Boot loop (keeps restarting)

**Symptoms:** Logo appears, black screen, board reboots, repeat.

**Fixes:**
1. **Power supply.** Must use bundled 19V barrel jack, NOT USB-C.
2. **SD card corruption.** Re-flash. Format as ext4 before flashing if new card.
3. **Overheating.** Ensure fan is connected and spinning. Don't block airflow.
4. **Different SD card.** Cheap cards cause corruption.

Source: [NVIDIA Forums](https://nvidia-jetson.piveral.com/jetson-orin-nano/jetson-orin-nano-dev-kit-boot-issues/)

### SD card not detected

**Symptoms:** Boots to UEFI menu or "no bootable device".

**Fixes:**
1. **Not fully inserted.** Push until click. Slot is on the **underside**.
2. **Flash failed.** Re-flash with Etcher (verifies the write).
3. **Bad card.** Use Samsung or SanDisk, UHS-1 or better.
4. **Check boot order.** Press Esc during boot to enter UEFI. Check SD card is in boot order.

### Power issues / random shutdowns

**Symptoms:** Shuts off under load, USB disconnects, display flickers.

**Fixes:**
1. **Use bundled 19V supply.** Not USB-C, not phone charger.
2. **Remove extra USB devices** during setup. Add camera/mic/speakers after.
3. **Check barrel jack** is fully seated.
4. **Monitor with jtop.** Board draws 7-25W depending on mode.

### Can't enter recovery mode

**Procedure:**
1. Power off (unplug barrel jack)
2. Locate J14 button header on carrier board
3. Bridge pins 9 (FC REC) and 10 (GND) with jumper wire
4. While holding bridge, plug in power
5. Wait 2 sec, release bridge
6. Connect USB-C from Jetson to computer

**Mac note:** Recovery USB from Mac is unreliable. For SDK Manager flashing, you need Ubuntu (native or UTM VM with USB passthrough). SDK Manager is Linux-only.

Source: [NVIDIA Forums](https://forums.developer.nvidia.com/t/how-to-get-into-recovery-mode/250525)

### WiFi not working

```bash
lspci | grep -i wireless         # Check if card is detected
nmcli device status               # Check network devices
```

If not detected: power off, reseat the M.2 WiFi card. Some cards have JetPack 6.2 driver issues. Use Ethernet instead (more reliable for dev work).

```bash
sudo apt-get update
sudo apt-get install linux-firmware
sudo reboot
```

Source: [NVIDIA Forums](https://forums.developer.nvidia.com/t/new-jetson-orin-nano-developer-kit-does-not-work-with-installed-aw-cb375nf-wifi-card/350957)

### Serial debug console (last resort)

If display shows nothing and you need boot logs:

1. Get a USB-to-TTL serial cable (3.3V, e.g., FTDI TTL-232R-3V3)
2. Connect to serial debug header: GND-GND, RX-TX, TX-RX
3. On Mac:
```bash
ls /dev/tty.usb*
screen /dev/tty.usbserial-XXXXX 115200
```
4. Power on Jetson. Detailed boot logs appear in serial console.

Source: [JetsonHacks](https://jetsonhacks.com/2019/04/08/jetson-nano-serial-debug-console/)

---

## Flashing from Mac (if SD card method fails)

SDK Manager only runs on Ubuntu. Mac workarounds:

1. **UTM VM (recommended):** Emulate Ubuntu 22.04 AMD64 on Mac M3 via UTM. Enable USB passthrough for Jetson in recovery mode. Install SDK Manager inside VM. One user confirmed this works on M3.
   Source: [DEV Community - JetPack 6.2 on Mac M3](https://dev.to/minwook/nvidia-sdk-manager-from-macm3-for-jetson-orin-nano-9b0)

2. **Borrow a Linux machine:** Any Ubuntu 22.04 x86_64 machine with USB ports works. Flash takes 15-30 min.

3. **Docker (limited):** SDK Manager Docker images exist but USB passthrough from Mac to Docker is not supported, making this impractical for flashing.

The SD card method described in this guide does NOT require a Linux machine. Only recovery-mode flashing does.

---

## All Sources

- [NVIDIA Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- [Jetson AI Lab Initial Setup](https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/)
- [Matt Dixon - Initial Setup](https://medium.com/@matt.dixon1010/jetson-orin-nano-super-developer-kit-initial-setup-fccba1d46b09)
- [NVIDIA User Guide - Hardware](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html)
- [NVIDIA User Guide - Software](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/software_setup.html)
- [JetPack 5.1.3 SDK](https://developer.nvidia.com/embedded/jetpack-sdk-513)
- [JetPack 6.2 SDK](https://developer.nvidia.com/embedded/jetpack-sdk-62)
- [PyTorch for Jetson Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [JPL PyTorch Wheels](https://pypi.jetson-ai-lab.io/jp6/cu126)
- [jetson-stats (jtop)](https://github.com/rbonghi/jetson_stats)
- [GitHub Actions Self-Hosted Runners](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [Balena Etcher](https://etcher.balena.io/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [Ajeet Raina - Jetson Super Guide](https://github.com/ajeetraina/jetson-orin-nano-super-guide)
- [DroneBot Workshop](https://dronebotworkshop.com/jetson-orin-nano/)
- [DEV Community - SDK Manager on Mac M3](https://dev.to/minwook/nvidia-sdk-manager-from-macm3-for-jetson-orin-nano-9b0)
