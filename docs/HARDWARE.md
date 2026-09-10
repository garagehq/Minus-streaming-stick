# Hardware and OS

What to buy to build a Minus, and what software the board needs to be running.
Everything below is either a vendor specification (linked) or a measurement
from the development unit, which is noted as such.

---

## Bill of materials

### Required

| Part | What to buy | Approx. | Why this part |
|---|---|---|---|
| Single-board computer | [Radxa ROCK 5B+](https://radxa.com/products/rock5/5bp/) (full-size HDMI in) or [ROCK 5B](https://radxa.com/products/rock5/5b/) (micro-HDMI in), **8 GB RAM recommended, 4 GB minimum** | $130 - $190 | The RK3588 is the only affordable SoC with a **4K@60 HDMI receiver** (`rk_hdmirx`), plus the VPU that encodes 4K JPEG in hardware and the NPU that runs OCR. The HDMI input is the non-negotiable part: boards without it cannot do passthrough. |
| AI accelerator | [Radxa AICore AX-M1](https://radxa.com/products/aicore/ax-m1/) (Axera AX8850, 24 TOPS INT8, 8 GB LPDDR4X, M.2 2280 M-key, ≤ 8 W) | ~$150 | Runs minus-v0.1, the 450M vision-language model, at ~370 ms per frame. The RK3588's own NPU is already busy with OCR and is not fast enough for the VLM on top of it. |
| Storage | 64 GB+ eMMC module or A2-rated microSD | $15 - $30 | The OS, plus screenshots collected for training. The dev unit runs a 64 GB card. |
| Power supply | USB-C **PD** adapter, 30 W or better (Radxa's own 30 W unit is a safe default) | ~$20 | The board plus the accelerator draw real current under 4K passthrough. A 5 V/2 A phone charger will brown out and cause random pipeline restarts. |
| Cooling | Heatsink **and** fan (Radxa's ROCK 5B heatsink+fan, or a 40 mm fan on the PWM header) | $10 - $25 | Not optional. Measured on the dev unit: sustained 4K60 passthrough sits at **80 - 85 °C with the fan at full**, which is where the SoC starts thermal-throttling. Minus has a [thermal governor](../src/thermal.py) that drops to 30 fps when this happens, but passive cooling alone will throttle constantly. |
| HDMI cables | 2x certified High Speed / 18 Gbps, as short as practical | $10 | Source → Minus in, Minus out → TV. Marginal cables show up as "No Signal" at 4K60; Minus can fall back to YCbCr 4:2:0 to halve the bandwidth, but a good cable is better. |

Rough total: **$350 - $450** depending on RAM, storage and where you buy.

### Optional

| Part | Use | Docs |
|---|---|---|
| IR LED + resistor on GPIO pin 38 | Switches an HDMI switch's input so autonomous mode can rotate between streaming devices | [IR_TRANSMITTER.md](IR_TRANSMITTER.md) |
| WS2812B 8-LED strip on SPI0 MOSI (pin 19) | Status indicator: idle, blocking, no signal, error | [STATUS_LEDS.md](STATUS_LEDS.md) |
| IR receiver (TSOP38238 class) on pin 3 | Bench-tested only, not wired into the app | [IR_RECEIVER.md](IR_RECEIVER.md) |
| HDCP 1.4 sink key | Capturing HDCP-protected sources | [../hdcp/README.md](../hdcp/README.md) |
| USB WiFi adapter with external antenna | The RK3588's internal antenna has short range, which matters if you use the setup hotspot | [../README.md](../README.md) |

### Notes on choosing parts

**RAM: 4 GB works, 8 GB is the comfortable choice.** Measured on the dev unit
shortly after a restart: the main process plus the OCR and VLM worker
subprocesses total roughly **1.2 GB resident**, and ustreamer adds ~140 MB. A
known glibc allocator behaviour lets that climb toward ~1.7 GB between HDMI
signal-loss events (see the arena note in [CLAUDE.md](../CLAUDE.md)), so 4 GB
leaves little headroom for anything else on the box. The accelerator has its
own 8 GB and does not consume host RAM for the model.

**M.2 slots.** The AX-M1 is M.2 2280 M-key. The ROCK 5B+ has **two** M-key
slots (PCIe 3.0 x2 each), so the accelerator and an NVMe SSD can coexist. The
ROCK 5B has **one**, so on that board the accelerator takes the only M.2 slot
and you boot from eMMC or microSD.

**ROCK 5B vs ROCK 5B+.** Both carry the RK3588 with the 4K@60 HDMI receiver, so
both work. The 5B+ uses a full-size HDMI input connector instead of micro-HDMI,
adds the second M.2 slot, 2.5 GbE and onboard eMMC. The development unit is a
ROCK 5B+ with 16 GB.

**The 2.5 GbE Realtek NIC (5B+).** It needs the out-of-tree `r8125-dkms`
driver on the Radxa image. If you use Ethernet, do not let package cleanup
remove it: keep `Unattended-Upgrade::Remove-Unused-Dependencies` set to
`"false"`, which is what Minus writes into its
[unattended-upgrades policy](../src/unattended_upgrades.py). WiFi is
unaffected.

---

## Operating system

The development unit runs, and this is the recommended configuration:

| Component | Version | How to check |
|---|---|---|
| Distribution | **Debian 12 (bookworm), arm64** | `cat /etc/os-release` |
| Image | Radxa official `rock-5b-plus_bookworm_*` (rsdk build) | `cat /etc/rsdk/config.yaml` |
| Kernel | **Radxa BSP 6.1.x** (`6.1.84-8-rk2410` here) | `uname -r` |
| Python | 3.11.2 | `python3 --version` |
| GStreamer | 1.22.9 with `gstreamer1.0-rockchip1` | `gst-launch-1.0 --version` |

Download images from [Radxa's docs](https://docs.radxa.com/en/rock5/rock5b/download)
or the [radxa-build releases](https://github.com/radxa-build/rock-5b-plus/releases).
Either the KDE or the CLI image works. Minus needs no desktop, and `install.sh`
disables the display manager because Minus takes the DRM/KMS master lock
directly; the CLI image saves you that step and some RAM.

### You must use the Radxa BSP kernel, not mainline

This is the single most common way a build fails. Four things Minus depends on
exist only in Rockchip's 6.1 BSP tree that Radxa ships:

- **`rk_hdmirx`**, the HDMI receiver driver that presents `/dev/video0`. Mainline has no equivalent.
- **Rockchip MPP**, the hardware JPEG encoder and decoder (`librockchip-mpp`).
- **RGA**, the 2D accelerator used for the zero-copy format conversions in the patched ustreamer.
- **`rknpu2`**, the NPU runtime PaddleOCR runs on.

A mainline or Armbian-mainline kernel will boot the board and give you a
desktop, and then Minus will find no capture device. Ubuntu also works in
principle if it is a Radxa BSP image, but Debian 12 is what everything here is
tested against.

Because of that, Minus's automatic security updates **blacklist** the kernel,
u-boot, Rockchip MPP/RGA, GStreamer, Mesa and RKNN packages. Upgrade those
deliberately, never unattended. See "Automatic security updates" in the
[README](../README.md).

### Third-party runtimes

Two pieces come from outside Debian and Radxa:

**Axera AXCL host driver** (for the AX-M1). Installed on the dev unit as
`axclhost` 3.6.6 from the M5Stack APT repository:

```
deb [signed-by=/etc/apt/keyrings/StackFlow.gpg] https://repo.llm.m5stack.com/m5stack-apt-repo axclhost main
```

Verify the card is present and healthy with `lspci | grep -i axera` (expect
`1f4b:0650`) and `axcl-smi info`. The Python binding is `axengine` (0.1.3 here),
installed with `pip3 install --break-system-packages axengine`.

**RKNN runtime** (for OCR). `rknpu2-rk3588` 2.3.0 from Radxa's
`rk3588-bookworm` repository, with `rknnlite` 2.3.0 on the Python side. Both
are already present on a current Radxa image.

If the VLM fails to load with a message about `libaxcl_*.so`, see the broken
symlink entry in [CLAUDE.md](../CLAUDE.md); it is a known packaging issue with
a one-line fix.

---

## Models

Minus uses two models, and they are distributed differently.

| Model | Where it runs | Where it comes from |
|---|---|---|
| **PaddleOCR PP-OCRv3** (detection + recognition) | RK3588 NPU | **In this repository**, at [`models/paddleocr/`](../models/paddleocr/). Nothing to download. |
| **minus-v0.1** (450M vision-language ad classifier) | Axera AX-M1 | [Hugging Face: TheGarageDev/Minus-v0.1](https://huggingface.co/TheGarageDev/Minus-v0.1) → `/home/radxa/axera_models/minus-v0.1/` |

The OCR models are ~11 MB and ship with the code, so a fresh clone can read ad
UI text immediately. See [`models/paddleocr/README.md`](../models/paddleocr/README.md)
for what each file does, checksums, and how to re-convert them. The VLM is
~300 MB and is downloaded separately.

Both locations can be overridden: `MINUS_OCR_MODEL_DIR` and
`MINUS_VLM_MODEL_DIR`.

---

## Assembly and first boot

1. Fit the heatsink and fan to the board before anything else.
2. Seat the AX-M1 in an M.2 M-key slot and screw it down. On the ROCK 5B+, either slot works.
3. Flash the Radxa Debian 12 image to eMMC or microSD, boot, and finish the usual first-boot setup.
4. Confirm the hardware is seen:
   ```bash
   v4l2-ctl -d /dev/video0 --info      # expect "rk_hdmirx"
   lspci | grep -i axera               # expect 1f4b:0650
   axcl-smi info                       # accelerator temperature and memory
   ```
5. Connect the source device to the HDMI **input** and the TV to the HDMI **output**, then follow [DEPLOYMENT.md](DEPLOYMENT.md).

If `/dev/video0` does not exist, you are almost certainly on a mainline kernel;
see the BSP section above. If it exists but reports no signal, check that the
source is on the input port rather than the output.
