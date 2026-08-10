# Minus — HDCP 1.4 Setup for Radxa Rock 5B Plus

Automated HDCP 1.4 enablement for the HDMI RX (input) port on the Rock 5B Plus.

Once configured, the device can capture HDCP-protected video from sources like FireTV, Roku, PS5, Xbox, Apple TV, and other consumer HDMI devices at up to 4K@60Hz.

## Requirements

- Radxa Rock 5B Plus running Debian Bookworm
- A valid HDCP 1.4 sink key file (`.bin`, exactly 308 bytes)
- Root access on the device

## Quick Start

```bash
cd /home/radxa/Minus/hdcp
sudo ./setup_hdcp_device.sh /path/to/your_hdcp_key.bin
```

The script handles the entire setup automatically:

1. **Validates** the key file (correct size, type ID, KSV bit parity)
2. **Converts** the key to the byte order the hardware expects (little-endian byte-swap)
3. **Installs U-Boot with OP-TEE** (BL32 firmware required for HDCP key loading)
4. **Patches the device tree** to enable HDCP hardware, add the vendor storage partition, and set the `hdcp1x-enable` flag
5. **Compiles the kernel module** on-device if kernel headers are available (or uses a pre-compiled fallback)
6. **Deploys** the key and kernel module for automatic loading on boot
7. **Creates a systemd service** so HDCP starts automatically after every reboot

After the reboot, HDCP is active. No further configuration needed.

## Verifying HDCP

```bash
# Check HDCP status
cat /sys/class/misc/hdmirx_hdcp/support   # Should be: 1
cat /sys/class/misc/hdmirx_hdcp/enable    # Should be: 1
cat /sys/class/misc/hdmirx_hdcp/status    # Should be: HDCP1.4: Authenticated success

# Check service
systemctl status hdcp-enable.service      # Should be: active (exited)

# Check module
lsmod | grep vendor_fix                   # Should show vendor_fix
```

## Capturing Video

Connect an HDCP source to the HDMI RX port, then:

```bash
# Capture a single frame
gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=1 \
  ! videoconvert ! jpegenc ! filesink location=/tmp/capture.jpg

# Continuous stream to display
gst-launch-1.0 v4l2src device=/dev/video0 \
  ! videoconvert ! autovideosink

# Record to file
gst-launch-1.0 v4l2src device=/dev/video0 \
  ! video/x-raw,format=NV16 ! videoconvert \
  ! x264enc ! mp4mux ! filesink location=/tmp/recording.mp4
```

> **Note:** The HDMI RX uses the multiplanar V4L2 API. Use GStreamer — `ffmpeg` does not support multiplanar capture on this device.

## HDCP Key Requirements

The key file must be:
- Exactly **308 bytes**
- Raw DCP LLC format: `type_id(4 bytes) + KSV(5 bytes) + device_keys(280 bytes) + padding(19 bytes)`
- Type ID must be `0x02000000` (Sink/Receiver)
- KSV must have exactly 20 ones and 20 zeros in binary (HDCP 1.4 spec requirement)
- Each device should use a **unique key** — do not reuse keys across devices

## Files

| File | Description |
|------|-------------|
| `setup_hdcp_device.sh` | Setup script — run with `sudo` and pass your key file |
| `u-boot-rk2410_2017.09-1_arm64.deb` | U-Boot package with OP-TEE/BL32 for Rock 5B Plus |
| `vendor_fix.c` | Kernel module source — compiled on-device automatically by the setup script |
| `vendor_fix.ko` | Pre-compiled kernel module (fallback if on-device compilation isn't available) |

## Kernel Module Compilation

The `vendor_fix.ko` kernel module must match the running kernel version. The setup script handles this automatically:

1. **If kernel headers are installed** (`linux-headers-$(uname -r)`): the script compiles `vendor_fix.c` on-device for the exact running kernel. No cross-compilation needed.

2. **If kernel headers are NOT available**: the script falls back to the pre-compiled `vendor_fix.ko`. This may fail if the kernel version doesn't match.

3. **If the module already exists and matches the running kernel**: the script skips compilation entirely.

To install kernel headers manually:
```bash
sudo apt install linux-headers-$(uname -r)
```

To force recompilation on the next run, delete the existing module:
```bash
rm /home/radxa/vendor_fix.ko
sudo ./setup_hdcp_device.sh /path/to/your_key.bin
```

## Troubleshooting

### `support` returns 0

| Check | Command | Meaning |
|-------|---------|---------|
| OP-TEE loaded? | `ls /dev/tee0` | If missing, U-Boot install failed — re-run setup |
| vnvm partition? | `cat /proc/mtd \| grep vnvm` | If missing, DTB patch failed |
| hdcp1x-enable? | `ls /proc/device-tree/hdmirx-controller@fdee0000/hdcp1x-enable` | If missing, DTB patch failed |
| Service running? | `systemctl status hdcp-enable.service` | Should be `active (exited)` |
| Module loaded? | `lsmod \| grep vendor_fix` | If missing, module failed to load (kernel mismatch?) |

### Status shows "Authenticated failed"

The HDCP handshake was attempted but the key was rejected. Possible causes:
- Invalid or corrupted key file
- Key not in the correct 308-byte DCP format
- Key reused from a revoked device

### Status shows "Unknown status"

No HDCP source is connected, or the source hasn't initiated the handshake yet. This is normal when nothing is plugged into the HDMI RX port. Plug in an HDCP source (FireTV, Roku, etc.) and check again after a few seconds.

### Black frame captured

If `status` shows `Authenticated success` but captures are black, the source may be using HDCP 2.x instead of 1.4. This setup only supports HDCP 1.4. Most consumer devices fall back to 1.4 automatically.

### No HDMI signal detected

```bash
v4l2-ctl -d /dev/video0 --get-dv-timings
# If "Link has been severed": unplug and replug the HDMI cable
# If "No locks available": the signal is unstable, replug and wait a few seconds
```

### Module fails to load (kernel mismatch)

If `dmesg | grep vendor_fix` shows version errors:
```bash
# Install kernel headers and re-run the setup
sudo apt install linux-headers-$(uname -r)
rm /home/radxa/vendor_fix.ko
sudo ./setup_hdcp_device.sh /path/to/your_key.bin
```

### SD card interference

If an SD card with a Radxa image is plugged in, U-Boot may load the unmodified device tree from the SD card instead of eMMC. The setup script handles this automatically, but for best results remove the SD card.

### Boot device detection

The script auto-detects whether the device boots from eMMC or SD card and flashes U-Boot to the correct device. If you move the SD card to a different slot or change boot media, re-run the setup script.

## Re-running the Script

The script is idempotent — it detects components that are already installed and skips them. Safe to re-run at any time, for example:
- To update the HDCP key
- After a kernel update (recompiles the module automatically)
- After moving to different boot media
