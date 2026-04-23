# Troubleshooting

Common issues when running the CLI locally.

## PyTorch fails to import on Windows (WinError 126)

**Symptoms**

- Any command that imports PyTorch (`train` / `test` / `merge`) fails immediately.
- You see an error like:
  - `OSError: [WinError 126] ... Error loading "...torch_python.dll" or one of its dependencies.`

**Cause**

PyTorch was installed, but Windows can't load its native DLL dependencies. This is usually caused by a missing or broken Microsoft Visual C++ runtime, or an incompatible PyTorch wheel.

**Fix (Windows)**

1) Install or repair **Microsoft Visual C++ Redistributable 2015–2022 (x64)**.

2) Reinstall PyTorch in the active venv (CPU-only example):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If you need CUDA, install the CUDA wheel from the official selector on `pytorch.org` instead of the CPU index URL.

3) Verify import works:

```powershell
python -c "import torch; print(torch.__version__)"
```

## CUDA / GPU not detected (PyTorch uses CPU)

**Symptoms**

- CLI logs `Detected device: cpu` even though you have an NVIDIA GPU.
- CLI logs `PyTorch appears to be installed as CPU-only (torch.version.cuda is None)`.
- `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` prints something like:
  - `2.x.x+cpu None False`
- `nvidia-smi` still works and shows the GPU correctly.

**Cause**

Your virtual environment has a **CPU-only** PyTorch build installed (the `+cpu` suffix), so `torch.cuda.is_available()` will always be `False` regardless of NVIDIA drivers.

On Windows this often happens because CUDA-enabled PyTorch wheels are not available for the Python version you used (commonly **Python 3.13**).

Example confirmed case:

```text
torch 2.11.0+cpu
torch.version.cuda None
torch.cuda.is_available() False
```

At the same time, `nvidia-smi` may show a healthy GPU and driver, for example an RTX 4080 with a recent NVIDIA driver. That only proves the driver side is installed; it does not mean the current Python environment has a CUDA-enabled PyTorch wheel.

**Fix (Windows, recommended)**

1) Create a new venv with Python **3.12**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

2) Install a CUDA-enabled PyTorch wheel (pick the exact command from the official PyTorch install selector):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3) Install the rest of the dependencies:

```powershell
pip install -r requirements.txt
```

4) Verify CUDA is visible:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

**Notes**

- `nvidia-smi` showing a GPU and a “CUDA Version” only confirms the driver side; PyTorch still needs a CUDA-enabled wheel.
- If `pip install ... cu128` fails with “No matching distribution”, switch to Python 3.12 (or another PyTorch-supported version) and retry.
- The CUDA version shown by `nvidia-smi` is the maximum supported by the driver. It does not have to match the wheel suffix exactly; choose one of the CUDA wheel variants supported by the official PyTorch selector.
