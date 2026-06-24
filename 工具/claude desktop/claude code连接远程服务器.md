# Claude Code 连接远程服务器

> 本机环境:Windows 10/11(系统自带 OpenSSH 客户端,`ssh` 命令在 PowerShell / CMD 中直接可用)。
> 本文为图片教程中「第二步:服务器配置代理」的 **Windows 版**操作步骤(原图为 mac 操作)。

## 服务器配置代理

> **目标**:通过 SSH 反向隧道,把本机(Windows)的代理共享给远程服务器,使远程上的 Claude Code 能访问 `api.anthropic.com`。
>
> **前提**:
> - 本机已运行代理客户端(Clash Verge / Clash for Windows / v2rayN 等),并记下其**本地 HTTP 代理端口**:Clash 默认 `7890`,v2rayN 的 HTTP 端口默认 `10809`。下文统一以 `7890` 为例,**端口不同请自行替换**。
> - 已在「第一步」用 Claude 的 Create Remote 创建好别名(下文示例为 `wby-9`)。

### 1. 配置本机 SSH(把本机代理暴露给远程)

#### 1.1 创建 .ssh 目录(已存在可跳过)

在 PowerShell 中执行:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
```

> Windows 下**无需 `chmod`**:OpenSSH 使用 NTFS ACL 控制权限,目录按默认即可。

#### 1.2 编辑 SSH 配置文件

用记事本打开(文件不存在会自动新建):

```powershell
notepad "$env:USERPROFILE\.ssh\config"
```

加入以下内容(别名 `wby-9` 必须与 Claude 里 Create 的名字一致):

```text
Host wby-9
    HostName <你的服务器地址>
    User <你的用户名>
    Port <你的端口>
    RemoteForward 7890 127.0.0.1:7890
```

> - `RemoteForward 7890 127.0.0.1:7890`:在远程开一个 `127.0.0.1:7890`,把流量经隧道传回**本机**的 `127.0.0.1:7890`(即本机代理端口)。
> - 若本机代理端口不是 7890(如 v2rayN 用 10809),把后半段改为 `127.0.0.1:10809`。

#### 1.3 验证别名与权限

```powershell
ssh wby-9      # 能正常登录即说明别名配置生效,登录后可先 exit 退出
```

> 一般无需手动设权限。若使用私钥且提示 `Permissions are too open` / `bad permissions`,用 `icacls` 收紧**私钥文件**权限(把路径换成你的私钥):
>
> ```powershell
> icacls "$env:USERPROFILE\.ssh\id_rsa" /inheritance:r /grant:r "$($env:USERNAME):R"
> ```

### 2. 建立反向隧道

在 PowerShell(或 CMD)中执行 —— Windows 10/11 自带 OpenSSH,命令与 mac 完全一致:

```powershell
ssh -N wby-9
```

> - 因为 1.2 已在 config 写了 `RemoteForward`,这里直接 `ssh -N wby-9` 即可建立隧道。
> - 等价的**显式写法**(若没在 config 写 `RemoteForward`,改用这条):
>
>   ```powershell
>   ssh -N -R 7890:127.0.0.1:7890 wby-9
>   ```
>
>   ⚠️ 两种写法**二选一**,不要同时使用,否则会因重复绑定端口报 `remote port forwarding failed for listen port 7890`。
> - 命令执行后会**一直挂起不返回,这是正常现象,不要关闭该窗口**。它的作用是:在远程开 `127.0.0.1:7890` 并经隧道把流量传回本机代理。只要这个窗口开着,隧道就一直在。
> - (可选)想让隧道更稳定,可在 config 的该 Host 下追加两行:`ServerAliveInterval 30` 与 `ServerAliveCountMax 3`。

### 3. 在远程 Ubuntu 上设置代理环境变量

SSH 登录远程后编辑 `~/.bashrc`,把代理变量加到**文件最顶部**(一定要在 `.bashrc` 中「非交互式 shell 直接 return」的判断**之前**,否则 Claude Code 这类非交互式启动方式读不到):

```powershell
ssh wby-9
```

登录后在**远程**执行:

```bash
vim ~/.bashrc
```

在最顶部加入:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
```

保存后使其立即生效:

```bash
source ~/.bashrc
```

> 这一步是在**远程服务器**上操作,与本机是 Windows 还是 mac 无关。

### 4. 验证链路

**另开一个新的 PowerShell 窗口**(保持第 2 步的隧道窗口不关闭),登录服务器测试链路是否打通:

```powershell
ssh wby-9
```

在**远程**执行:

```bash
curl --proxy http://127.0.0.1:7890 https://api.anthropic.com/v1/models
```

> - 若返回 **JSON**(而不是 `Connection refused` 之类),说明反向隧道 + 远程环境变量都已生效。
> - 务必确认第 2 步的隧道窗口仍开着,否则远程 `127.0.0.1:7890` 不通。
> - 注:图片原文 `127.0.0.1:789` 为笔误,正确端口是 **7890**。
