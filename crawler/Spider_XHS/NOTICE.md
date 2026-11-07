# 第三方代码说明（NOTICE）

本目录 `Spider_XHS/` 是第三方开源项目，**不是本项目原创代码**。

| 项 | 内容 |
|---|---|
| 项目 | Spider_XHS — The All-in-One Manager for XHS |
| 上游仓库 | https://github.com/cv-cat/Spider_XHS |
| 作者 | cv-cat |
| 获取方式 | 微信接收的 `Spider_XHS-master.zip`（2026-10-22 快照，master 分支） |
| 收录日期 | 2026-11-07 |

## 许可

上游 `README.md` 中的 License 徽章标注为 **MIT**，但其 GitHub 仓库（master 分支，2026-11-07 核对）**根目录下没有 LICENSE / LICENSE.md 文件**，本快照中同样缺失。

因此这里不附许可正文，仅作事实记录。若要公开发布本仓库或再分发这部分代码，建议先向上游作者确认授权（README 中的联系方式：992822653@qq.com），或补入上游正式许可文本。

## 本地改动

只改了一处，为保证仓库里不残留可能有效的凭据：

- `main.py` 第 28 行 `NOTE_URL` 的示例链接中，真实 `xsec_token` 值已替换为占位符 `<note_id>` / `<xsec_token>`。

其余文件与快照逐字节一致，未做任何修改。

## 用途限制

本项目仅用它采集公开的小红书图文帖用于学术研究。使用时请遵守平台服务条款、控制请求频率，不要把 cookie 或抓取数据用于其他用途。
