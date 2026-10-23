# LeNet-5 Python 环境

项目虚拟环境位于 `.venv`，使用本机的 Python 3.8.8（64 位）。

- PyTorch：2.4.1 + CUDA 12.1
- torchvision：0.19.1 + CUDA 12.1
- NumPy：1.24.4

预编译的 PyTorch 包包含运行所需的 CUDA 库，普通训练无需单独安装 CUDA Toolkit。
安装版本参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v241)。

## 使用

在项目目录的 PowerShell 中直接运行，无需先激活环境：

```powershell
.\.venv\Scripts\python.exe net.py
.\.venv\Scripts\python.exe check_gpu.py
```

也可以先激活，再使用 `python`：

```powershell
.\.venv\Scripts\Activate.ps1
python check_gpu.py
```

如果 PowerShell 的执行策略阻止激活，使用上面的完整解释器路径即可。

VS Code 已配置默认解释器路径。如果当前窗口此前选过其他解释器，
执行 `Python: Select Interpreter`，选择本项目的 `.venv\Scripts\python.exe`，然后新建终端。

模型和输入数据都需要放到 GPU，例如：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images, labels = images.to(device), labels.to(device)
```

## 重建环境

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.0.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` 记录了本次安装的全部依赖版本；需要复现完全相同的版本时，
将上面的 `requirements.txt` 替换为 `requirements-lock.txt`。

如果本机代理引起 `ProxyError`，可在安装命令后添加
`--proxy http://127.0.0.1:7897`（仅适用于该端口的本机代理正在运行时）。
