# Mali-G610 GPU setup (OpenCL + Vulkan) on Radxa ROCK 5B+ / RK3588

How to get working OpenCL 3.0 and Vulkan 1.3 compute on the RK3588's
Mali-G610 under Radxa OS (Debian 12 bookworm, BSP kernel
`6.1.84-*-rk2410`), and the traps that crashed or wasted time on minus-2.

**Read first: Minus does not need the GPU.** We tried it to speed up ASR
and it lost to the CPU on every model (see [Results](#results)). Speech
recognition now runs on the **NPU** (SenseVoice, `src/sensevoice_npu.py`).
Only enable the GPU on another unit if you have a different workload that
benefits; the stock unit (panthor driver, no overlay) is the tested
production configuration.

Reference: Radxa's own guide, https://docs.radxa.com/en/rock5/rock5b/radxa-os/mali-gpu

## Background: two kernel drivers, one GPU

| | `panthor` (default) | `bifrost_kbase` (Rockchip/ARM) |
|---|---|---|
| Device-tree node | `gpu` (enabled) | `gpu_mali` (disabled by default) |
| Userspace | Mesa (panfrost/panvk) | ARM proprietary `libmali` |
| Compute on this image | **none usable** (see traps 2 and 3) | OpenCL 3.0 + Vulkan 1.3 |
| Firmware | `/lib/firmware/arm/mali/arch10.8/mali_csffw.bin` | embedded in the module |
| Device file | `/dev/dri/renderD*` | `/dev/mali0` |

Both drivers ship in the kernel (`kernel/drivers/gpu/drm/panthor/`,
`kernel/drivers/gpu/arm/bifrost/`). Which one binds is decided **only by
the device tree at boot**.

## Setup (about 10 minutes, one reboot)

### 1. Switch the device tree to `gpu_mali` (overlay + reboot)

Radxa ships the overlay disabled as
`/boot/dtbo/rk3588-mali-enable.dtbo.disabled`. It disables the `gpu` node
and enables `gpu_mali` with its `vdd_gpu_s0` / `vdd_gpu_mem_s0` supplies.

```bash
# Either: sudo rsetup -> Overlays -> enable "Enable Mali GPU"
# Or by hand:
sudo mv /boot/dtbo/rk3588-mali-enable.dtbo.disabled /boot/dtbo/rk3588-mali-enable.dtbo
sudo u-boot-update                       # regenerates /boot/extlinux/extlinux.conf
grep fdtoverlays /boot/extlinux/extlinux.conf   # must list rk3588-mali-enable.dtbo
sudo reboot
```

After the reboot:

```bash
lsmod | grep -E 'bifrost|panthor'   # expect bifrost_kbase, NOT panthor
ls -l /dev/mali0                    # crw-rw---- root video
```

Your user must be in the `video` group to open `/dev/mali0` (the `radxa` user
already is). `/etc/modprobe.d/panfrost.conf` blacklists panfrost, which is
correct and harmless here.

### 2. Get ARM's `libmali` (userspace) without replacing system Mesa

Use the **g24p0** build. It matches the kernel's kbase g25p0 closely enough,
and it is the first that ships a working Vulkan ICD. Unpack it privately
rather than installing it. The package conflicts with Mesa's GL/EGL and would
change the rest of the graphics stack.

```bash
mkdir -p ~/mali && cd ~/mali
apt-get download libmali-valhall-g610-g24p0-gbm      # from the Radxa apt repo
dpkg -x libmali-valhall-g610-g24p0-gbm_*_arm64.deb g24
ls g24/usr/lib/aarch64-linux-gnu/libmali.so.1.9.0    # the one library does GL, CL and VK
```

### 3. Point the OpenCL and Vulkan loaders at it

```bash
sudo apt install -y ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers \
                    clinfo libvulkan1 vulkan-tools
mkdir -p ~/mali/icd
LIB=$HOME/mali/g24/usr/lib/aarch64-linux-gnu/libmali.so.1.9.0

# OpenCL: an .icd file is just the library path
echo "$LIB" > ~/mali/icd/mali.icd

# Vulkan: api_version MUST be 1.3.276 (see trap 5), not the packaged 1.0.5
cat > ~/mali/icd/mali_vk.json <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": { "library_path": "$LIB", "api_version": "1.3.276" }
}
EOF
```

### 4. Verify

```bash
OCL_ICD_VENDORS=~/mali/icd clinfo -l
#   Platform #0: ARM Platform
#    `-- Device #0: Mali-G610 r0p0

VK_ICD_FILENAMES=~/mali/icd/mali_vk.json vulkaninfo --summary | grep -E 'apiVersion|deviceName|driverInfo'
#   apiVersion = 1.3.276
#   deviceName = Mali-G610
#   driverInfo = v1.g24p0-00eac0...
```

Export the two variables only for the processes that need the GPU. Setting
them globally makes every Vulkan/CL program on the box use libmali.

### 5. Building Vulkan compute code (e.g. ggml / whisper.cpp / llama.cpp)

Debian's `libvulkan-dev` headers (1.3.239) are too old for current ggml,
which fails with errors about `LayerSettingEXT`. Use newer headers and the
shader compiler:

```bash
sudo apt install -y libvulkan-dev glslc spirv-headers
git clone --depth 1 -b v1.4.304 https://github.com/KhronosGroup/Vulkan-Headers ~/Vulkan-Headers
cmake -B build-vk -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$HOME/Vulkan-Headers/include
cmake --build build-vk -j4
VK_ICD_FILENAMES=~/mali/icd/mali_vk.json ./build-vk/bin/<tool> ...
# ggml_vulkan: 0 = Mali-G610 (Mali-G610) | uma: 1 | fp16: 1 | warp size: 16 | shared memory: 32768
```

### Reverting to the stock driver

```bash
sudo mv /boot/dtbo/rk3588-mali-enable.dtbo /boot/dtbo/rk3588-mali-enable.dtbo.disabled
sudo u-boot-update && sudo reboot      # panthor binds again
```

`~/mali` can stay; nothing uses it unless the environment variables are set.

## Traps to avoid on other units

1. **Never swap GPU drivers on a running system.** Unbinding panthor and
   running `modprobe bifrost_kbase` without the overlay **hard-hung minus-2,
   which then rebooted**. With the stock device tree the `gpu_mali` node is
   disabled and its regulators are not described, so kbase drives an
   unpowered GPU. Only the overlay plus a reboot is safe.
2. **Mesa `panvk` does not support panthor in this Mesa (24.2, oibaf
   PPA).** `vulkaninfo` on the stock driver shows only llvmpipe (CPU). Do
   not spend time on it. Newer Mesa (25.x) adds panvk-on-panthor, but that
   is not packaged for this image.
3. **Mesa rusticl (OpenCL) is not installable.** `mesa-opencl-icd` needs a
   `libclang-cpp15` that conflicts with the installed LLVM. There is no
   OpenCL on the panthor path.
4. **ggml's OpenCL backend rejects Mali.** It supports only Adreno and
   Intel: `ggml_opencl: unsupported GPU 'Mali-G610 r0p0' ... drop
   unsupported device`. Forcing it through as "Intel" (source patch)
   **segfaults at model load**. Use the Vulkan backend for ggml.
5. **The libmali Vulkan ICD lies about its API version.** The packaged
   JSON says `api_version: 1.0.5` while the driver implements 1.3.276. The
   loader then won't route Vulkan 1.1+ entry points, and ggml crashes in
   `vkGetDeviceQueue2` on a **NULL function pointer** (`#0 0x0000000000000000`
   under gdb). Write your own ICD JSON with `1.3.276` (step 3).
6. **Firmware confusion.** kbase g25p0 has the CSF firmware **built into
   the module**, so `mali_csffw.bin` is not needed for it. The
   `/lib/firmware/arm/mali/arch10.8/mali_csffw.bin` file belongs to panthor.
   A symlink at `/lib/firmware/mali_csffw.bin` is harmless but unnecessary.
7. **Use g24p0, not g13p0.** The older g13p0 `libmali` still in the repo
   has no usable Vulkan.
8. **GPU devfreq.** The GPU governor is `simple_ondemand`, ramping
   300-1000 MHz. Short compute bursts may finish before it ramps. For
   benchmarks pin it, then put it back:
   ```bash
   echo performance | sudo tee /sys/class/devfreq/fb000000.gpu-mali/governor
   echo simple_ondemand | sudo tee /sys/class/devfreq/fb000000.gpu-mali/governor
   ```
   Pinning did not change the result below.
9. **Thermals.** Minus runs the SoC near its 85°C trip during 4K passthrough
   and the kernel throttles the CPU. The GPU shares the same power and
   thermal budget, so GPU work is not "free" headroom.

## Results

Benchmark: whisper.cpp v1.9.5, Vulkan backend, a 5 s clip of NBA
commentary, 3 CPU threads, Minus running.

| Model | GPU (Vulkan, Mali) encode | CPU encode | GPU total | CPU total |
|---|---|---|---|---|
| tiny.en | 621 ms/run | **158 ms/run** | 7.0 s | **2.9 s** |
| base.en | 854 ms/run | **469 ms/run** | 9.3 s | **5.3 s** |
| tiny.en, GPU at 1 GHz | 542 ms/run | | | |

Decoding (batched) was also 2x slower on the GPU (14 vs 7 ms/token for
tiny). The Mali reports no matrix cores and no integer dot product
(`int dot: 0 | matrix cores: none`), so ggml's matmul kernels have nothing
to accelerate them. The A76 cores with NEON dot product win.

What Minus uses instead, for name detection on the same audio:

| Engine | Where | Cost per window | Name recall | Word timing vs captions |
|---|---|---|---|---|
| **SenseVoice-small (RKNN)** | **NPU core 1** | **~0.38 s, fixed** | **85%** (3 s windows) | **+0.10 to +0.22 s** |
| Moonshine medium-streaming | 3 CPU cores | ~1.1 s p50, 1.7 s p95 | 88% (2.5 s) | +0.22 to +0.40 s |
| sherpa-onnx KWS (zipformer 3.3M) | CPU | RTF 0.07 | 27% | n/a |

The NPU path freed three CPU cores and cut the A/V delay line from 5 s to
2 s. See `tests/sensevoice_npu_eval.py`.
