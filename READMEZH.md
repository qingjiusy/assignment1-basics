# CS336 2025 春季作业 1：基础

有关本次作业的完整说明，请参阅作业讲义：
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

如果你发现作业讲义或代码中有任何问题，欢迎提交 GitHub issue，或打开一个包含修复的 pull request。

## 配置

### 环境
我们使用 `uv` 管理环境，以确保可复现性、可移植性和易用性。
建议在[这里](https://github.com/astral-sh/uv#installation)安装 `uv`，也可以运行 `pip install uv` 或 `brew install uv`。
我们建议你花一点时间阅读[这里](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)关于使用 `uv` 管理项目的说明（不会后悔的！）。

现在你可以使用下面的命令运行仓库中的任意代码：
```sh
uv run <python_file_path>
```
必要时，环境会自动解析并激活。

### 运行单元测试


```sh
uv run pytest
```

一开始，所有测试都应该因为 `NotImplementedError` 而失败。
要将你的实现连接到测试，请完成
[./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据
下载 TinyStories 数据以及 OpenWebText 的一个子样本：

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
