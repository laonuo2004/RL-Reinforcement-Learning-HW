# RL-Reinforcement-Learning-HW

北京理工大学 23 级计科大三强化学习刘驰班作业

## 关于腾讯开悟平台代码手动同步

由于开悟平台 IDE 处于隔离网络环境当中，连不上 Github 等平台，因此无法使用 Git 来同步代码；~~此外 IDE 里也无法上传下载文件~~（"新增训练任务"后可以下载代码文件，不过还是没法上传），这样一来，我们就只能在它提供的 IDE 当中编写代码了。虽然我对于平台提供的 IDE 没有什么意见，但是我还是更加希望在自己熟悉的环境当中开发，愉快使用各种 AI 工具。那么有没有什么办法可以方便地同步代码呢？

除了 Git 与文件传输这两种途径以外，我们还可以通过复制粘贴来快速同步代码，这在修改量较小的情况下比较方便，当修改量较大时，可以在一台机器上将项目打包压缩后转 Base64 编码，复制编码到另一台机器上再解码还原，这样一来就可以实现代码的快速同步。可以借助 [pack_and_encode.sh](腾讯开悟平台实验/pack_and_encode.sh) 和 [decode_and_unpack.sh](腾讯开悟平台实验/decode_and_unpack.sh) 这两个脚本来实现。

### Case 1: 将开悟平台上的代码同步到本地

初次进入开发环境时会看见一些代码文件，我们需要首先将它们同步到本地以便于开发。

> 注意：开悟平台点击 "进入开发" 后进入的工作目录 (`/data/projects/{project_name}`) 当中，存放的不是代码文件本身，而是**符号链接 (symlink)**。真正的代码文件存放在 `/workspace/code/` 当中 (具体路径视实际情况而定)，需要进入该目录下进行同步。

1. 首先创建一个新的脚本文件 `pack_and_encode.sh`，将 [pack_and_encode.sh](腾讯开悟平台实验\pack_and_encode.sh) 复制进去。
2. 然后执行如下命令：
   ```bash
   # 注意可能要多执行几次才能成功
   sh ./pack_and_encode.sh /workspace/code/

   ```
3. 将输出的文件复制到当前路径下：
   ```bash
   cp /workspace/code/encoded_output.txt .

   ```
4. 在本地项目下创建 `encoded_output.txt` 文件，将编码后的内容粘贴进去。
5. 在本地执行 [decode_and_unpack.sh](腾讯开悟平台实验/decode_and_unpack.sh) 脚本，将编码后的内容解码还原：
   ```bash
   sh ./decode_and_unpack.sh /path/to/your/project/

   ```

同步完成。

### Case 2：本地编辑代码后同步到开悟平台上

其实就是 Case 1 反过来，不过需要留意的是代码应该同步到 `workspace/code/` 当中 (具体路径视实际情况而定)，而非点击 "进入开发" 后第一眼所看到的目录。

1. 本地项目打包编码：
   ```bash
   # 同样可能要多执行几次才能成功
   sh ./pack_and_encode.sh /path/to/your/project/

   ```
2. 开悟平台新建文件 `encoded_output.txt`，将编码后的内容粘贴进去。同时新建脚本文件 `decode_and_unpack.sh`，将 [decode_and_unpack.sh](腾讯开悟平台实验/decode_and_unpack.sh) 复制进去。
3. 随后复制编码到代码实际存放位置当中去，并解码：
   ```bash
   cp ./encoded_output.txt /workspace/code/
   sh ./decode_and_unpack.sh /workspace/code/
   
   ```

同步完成。

## 三个实验的相关说明

- [峡谷漫步](https://github.com/laonuo2004/RL-Reinforcement-Learning-HW/tree/main/%E8%85%BE%E8%AE%AF%E5%BC%80%E6%82%9F%E5%B9%B3%E5%8F%B0%E5%AE%9E%E9%AA%8C/gorge_walk)