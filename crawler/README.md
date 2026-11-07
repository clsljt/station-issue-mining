# 爬虫（code/crawler/）

本项目 10 个站的小红书数据就是用它采的。**这部分是第三方代码**，不是本项目原创，来源与许可见 [`Spider_XHS/NOTICE.md`](Spider_XHS/NOTICE.md)。

| 项 | 内容 |
|---|---|
| 上游 | [cv-cat/Spider_XHS](https://github.com/cv-cat/Spider_XHS) |
| 快照 | commit `23dab80`，2026-10-22 |
| 本地改动 | 只抹掉了 `main.py` 示例链接里的真实 `xsec_token` |
| 本目录新增 | `flatten_notes.py`（本项目写的后处理脚本） |

---

## 一、它在本项目里的位置

```
采集（crawler/Spider_XHS）           ← 你在这里
   ↓ 产出 <昵称>_<user_id>/<标题>_<note_id>/
拍平（crawler/flatten_notes.py）     ← 本项目新增
   ↓ 产出 <note_id>/{info.json, detail.txt, image_*.jpg}
落库（data/raw/station_data/<站>/<站>_media/）
   ↓
分析（code/pipeline/process_comments.py → analyze_notes.py）
```

爬虫原始落盘是**两级目录**（`昵称_用户ID/标题_笔记ID/`），而 `analyze_notes.py` 要求帖子目录直接以 `note_id` 命名、`info.json` 就在里面。中间的这步「拍平」原先没留下脚本，现在补上了。

---

## 二、环境

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.10+ | `pip install -r Spider_XHS/requirements.txt` |
| Node.js | 20+ | 签名脚本（`xhs_utils/**/js/*.js`）由 node 执行，缺了会报错 |
| Cookie | 小红书网页版 | 放进 `.env`，见下 |

```bash
cd code/crawler/Spider_XHS
pip install -r requirements.txt
cp .env.example .env        # 然后编辑 .env
```

`.env` 内容只有一个字段：

```
COOKIES='你的完整 cookie 字符串'
```

> `.env` 已被上游 `.gitignore` 排除，**不会进仓库**。也不要把它贴进聊天或文档。

---

## 三、怎么用

具体接口（`Data_Spider.spider_some_search_note`、`spider_user_all_note`、`spider_note` 等）以上游 [`Spider_XHS/README.md`](Spider_XHS/README.md) 为准，这里只记本项目实际用到的部分：

1. **采集**：按站点关键词搜索（如「北京西站」），把笔记列表连同配图、评论落盘。
   `save_choice='all'` 会同时写 `info.json` + `detail.txt` + `image_*.jpg` + 汇总 xlsx。
2. **拍平**：把两级目录规整成 `<note_id>/`。

```bash
cd code/crawler
python flatten_notes.py \
  --src  ~/xhs_dump/北京西站 \
  --dst  ../../data/raw/station_data/XiZhan/XiZhan_media \
  --move            # 移动而不是复制，省一份空间
```

3. **接回主流程**：评论 xlsx 用 `code/pipeline/process_comments.py` 拆进各帖子目录，然后跑 `analyze_notes.py`。

---

## 四、产出与现有数据的对应

| 爬虫产出 | 本项目数据 |
|---|---|
| `<昵称>_<user_id>/<标题>_<note_id>/info.json` | `data/raw/station_data/<站>/<站>_media/<note_id>/info.json` |
| `<…>/detail.txt` | 同上目录下的 `detail.txt` |
| `<…>/image_0.jpg …` | 同上目录下的 `image_*.jpg` |
| `*_notes.xlsx` | `data/raw/station_xlsx/<站>/`（历史副本）、`<站>/<站>_notes.xlsx` |
| `*_comments.xlsx` | `<站>/<站>_comments.xlsx`（再由 `process_comments.py` 拆成 `comment.json`） |

字段对齐情况：`analyze_notes.py` 读的是 `title` / `desc` / `liked_count` / `comment_count` / `collected_count` / `tags` / `upload_time`，都能在 `info.json` 里找到。评论字段（`content` / `like_count` / `nickname`）在 `process_comments.py` 里映射。

---

## 五、合规提醒

- 只采公开笔记，仅用于本项目的学术研究。
- 控制请求频率，不要并发轰炸；上游自带重试，建议保留默认间隔。
- 不要提交 cookie、登录态（`_live_session.json`、`creator_cookie.json` 等，上游 `.gitignore` 已排除）。
- 不要将采集数据二次分发。

---

## 六、已知问题

- 上游 README 标注 MIT，但仓库没有 LICENSE 文件，公开发布前建议向作者确认（见 `Spider_XHS/NOTICE.md`）。
- 签名逻辑随平台前端更新会失效，属于这类爬虫的常态；失效时以上游最新提交为准。
- 本项目使用的数据采集于 2026-06，当时的接口版本可能与快照不一致。
