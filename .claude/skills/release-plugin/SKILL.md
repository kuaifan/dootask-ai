---
name: release-plugin
description: 用于发布 DooTask AI 插件新版本的完整流程。当用户提到"发布"、"打 tag"、"出新版本"、"release"、"上架"、"推到 DooTask AppStore"、"打标签发版"、"v0.x.x"、"升级版本号"等任何与本仓库版本发布相关的意图时，必须使用本技能 —— 即使用户只是说"准备发版"或"该发新版了"也要触发。本技能会引导完成提交代码、打 tag、推送、监控 GitHub Action 直至发布成功，避免遗漏步骤导致发布失败。
---

# 发布 DooTask AI 插件

本仓库使用 gitflow 工作流，通过推送 `v*` 格式的 tag 触发 GitHub Action 自动构建并发布到 DockerHub、GitHub Container Registry 和 DooTask AppStore。

## 触发条件回顾

只有 **推送到 `main` 的 tag**（且 tag 名以 `v` 开头）才会触发实际的构建发布。普通的 push 或 PR 不会发布。这意味着发布的"开关"就是那个 tag。

## 发布流程

按顺序执行下面的步骤。每一步都先与用户确认再操作 —— 发布是公开行为且涉及多个外部系统（Docker Hub、GHCR、DooTask AppStore），不可逆。

### 1. 确认当前状态干净

```bash
git status
git log --oneline -10
```

确认：
- 工作区干净，没有未提交的改动
- 当前在 `main` 分支
- 本地与远程同步（`git fetch && git status` 检查是否落后于 origin/main）

如果有未提交的改动，先和用户确认是要包含进本次发布、还是要 stash/丢弃。

### 2. 决定新版本号

```bash
git tag --sort=-creatordate | head -5
```

看最新 tag 后，与用户确认新版本号。遵循语义化版本（SemVer）：

- `v0.x.Y` patch：bugfix、小修小补
- `v0.X.0` minor：新功能、模型新增、不破坏兼容性的重构
- `vX.0.0` major：破坏性变更

**版本号格式必须是 `v` 开头**，如 `v0.5.1`，否则 workflow 的 `tags: [ 'v*' ]` 不会触发。

### 3. 推送代码到远程

```bash
git push origin main
```

确认远程 main 已经有要发布的提交。

### 4. 打 tag 并推送

```bash
git tag v0.5.1                      # 替换为实际版本号
git push origin v0.5.1
```

> 注意：tag 一旦推送到远程，就会立即触发 Action。在推送前再次确认版本号无误。
> 如果误推了 tag，可以用 `git push --delete origin v0.5.1 && git tag -d v0.5.1` 删除（但如果 Action 已经发布到 DooTask AppStore，删除 tag 不会撤回已发布的版本）。

### 5. 监控 GitHub Action

```bash
gh run list --workflow=release.yml --limit 3
gh run watch                        # 实时跟踪最新 run
```

或在浏览器打开：`https://github.com/<owner>/<repo>/actions`

Action 会执行：
1. 多架构构建 Docker 镜像（linux/amd64、linux/arm64）
2. 推送镜像到 GitHub Container Registry 和 Docker Hub
3. 打包 `dootask-plugin/` 目录为 tar.gz
4. 通过 `dootask/appstore-action@v3` 发布到 DooTask AppStore（appid: `ai`，作为系统应用）

### 6. 验证发布结果

Action 成功后，确认：
- Docker Hub 上 `kuaifan/dootask-ai` 出现新 tag
- GHCR 上对应仓库出现新 tag
- DooTask AppStore 中 `ai` 插件版本已更新

## 常见问题

**Action 没触发**：检查 tag 是否以 `v` 开头，是否已经 `git push origin <tag>`（仅 `git tag` 不会推送到远程）。

**Docker 登录失败**：通常是 Secrets 过期或缺失，见下方"GitHub Secrets 配置"注释。

**DooTask AppStore 发布失败**：检查 `dootask-plugin/` 目录结构是否正确，以及 `DOOTASK_USERNAME` / `DOOTASK_PASSWORD` 是否有效。

**版本号已存在**：DooTask AppStore 不允许重复版本号；要发布修复版本，须递增 patch 号（例如 `v0.5.1` → `v0.5.2`）。

<!--
================================================================================
GitHub Secrets 配置（仅供参考，本技能不会也无法配置这些密钥）
================================================================================

发布流程依赖以下 4 个 Repository Secrets，需在 GitHub 仓库的
  Settings -> Secrets and variables -> Actions -> Repository secrets
中预先配置完成。这是仓库管理员的一次性工作，不是每次发布都需要做：

  - DOOTASK_USERNAME : DooTask AppStore 用户名
  - DOOTASK_PASSWORD : DooTask AppStore 密码
  - DOCKERHUB_USERNAME : Docker Hub 用户名
  - DOCKERHUB_TOKEN    : Docker Hub Access Token（不是登录密码）

如果首次发布或 Action 报登录失败，请提醒用户检查上述 Secrets 是否齐全且有效。
GHCR（ghcr.io）使用内置的 GITHUB_TOKEN，无需额外配置。
================================================================================
-->
