# NDLOCR-Lite Server — 開発者向けドキュメント

---

## 目次

1. [ファイル構成](#ファイル構成)
2. [API リファレンス](#api-リファレンス)
3. [設定リファレンス](#設定リファレンス)
4. [アーキテクチャ](#アーキテクチャ)
5. [GPUバックエンド利用](#gpuバックエンド利用)

---

## ファイル構成

```
ndlocr-lite-server/
├── server/
│   ├── main.py               # FastAPI アプリ本体
│   └── templates/
│       └── index.html        # Web UI（SPA、ビルド不要）
├── src/
│   ├── model/                # ONNX モデルファイル（要配置）
│   ├── config/               # クラス定義 YAML
│   ├── ocr.py                # OCR パイプライン
│   └── parseq.py             # PARSEQ 認識器
├── tools/
│   ├── patch_dynamic_batch_v2.py  # ONNX 動的バッチ化パッチャー
│   ├── convert_fp16.py            # ONNX FP16 変換
│   └── benchmark_batch.py         # バッチ推論ベンチマーク
├── config.toml.example       # 設定ファイルのテンプレート
├── run.bat / run.sh           # 起動スクリプト
└── dev-doc.md                 # このファイル
```

モデルファイルは別途 `src/model/` に配置が必要です。
1 つでも欠けていると起動を拒否します。

```
src/model/
├── deim-s-1024x1024.onnx
├── parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx        # バッチ=1
├── parseq-ndl-16x256-30-tiny-192epoch-tegaki3_dynamic.onnx      # 動的バッチ
├── parseq-ndl-16x256-30-tiny-192epoch-tegaki3_dynamic_fp16.onnx # 動的バッチ + FP16
├── parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx
├── parseq-ndl-16x384-50-tiny-146epoch-tegaki2_dynamic.onnx
├── parseq-ndl-16x384-50-tiny-146epoch-tegaki2_dynamic_fp16.onnx
├── parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx
├── parseq-ndl-16x768-100-tiny-165epoch-tegaki2_dynamic.onnx
└── parseq-ndl-16x768-100-tiny-165epoch-tegaki2_dynamic_fp16.onnx
```

---

## API リファレンス

### `GET /api/status`

サーバーのモデル・デバイス情報を返します。

**レスポンス例**
```json
{
  "model": "NDLOCR-Lite",
  "device_default": "amdgpu",
  "cuda_available": false
}
```

---

### `POST /api/jobs`

ファイルをアップロードして OCR ジョブを登録します。処理は非同期で行われ、`job_id` を即座に返します。

**リクエスト** : `multipart/form-data`

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `file` | file | **必須** | 画像または PDF |
| `dpi` | int | `220` | PDF ラスタライズ解像度 |
| `linebreak_mode` | string | `none` | `none` / `paragraph` / `compact` |
| `reading_order` | string | `auto` | `auto` / `ltr_ttb` / `rtl_ttb` |

**レスポンス例**
```json
{
  "job_id": "a1b2c3d4-...",
  "state": "queued"
}
```

---

### `GET /api/jobs/{job_id}`

ジョブの状態を返します。処理中は進捗情報のみ、完了時は結果も含みます。

**レスポンス（処理中）**
```json
{
  "state": "queued | processing | canceling",
  "message": "ページ 3/20 完了...",
  "total_pages": 20
}
```

**レスポンス（完了）**
```json
{
  "state": "done | canceled | error",
  "message": "完了",
  "total_pages": 20,
  "results": [
    {
      "page": 1,
      "text": "整形済みテキスト",
      "raw": "整形前テキスト",
      "html": "<!DOCTYPE html>...",
      "json": null,
      "error": null,
      "layout_preview_b64": "...(base64 JPEG)...",
      "blocks": [
        {
          "id": "line-0",
          "type": "line_main",
          "order": 0,
          "box": [x1, y1, x2, y2],
          "text": "認識テキスト",
          "is_vertical": true
        }
      ],
      "summary": { "line_main": 12, "line_title": 2 }
    }
  ],
  "task": "text",
  "linebreak_mode": "none",
  "page_count": 20,
  "device": "amdgpu",
  "reading_order": "auto",
  "error_detail": null
}
```

`state` 一覧：

| state | 意味 |
|---|---|
| `queued` | キュー待ち（前のジョブが処理中） |
| `processing` | 推論中 |
| `canceling` | キャンセル要求を受付、処理中 |
| `done` | 正常完了 |
| `canceled` | キャンセル完了 |
| `error` | エラー終了（`error_detail` に詳細） |
| `not_found` | 存在しない job_id（サーバー再起動後など） |

完了ジョブは完了から **5 分後** に自動削除されます。

---

### `POST /api/jobs/{job_id}/cancel`

処理中断を要求します。PDF の現在処理中チャンクが完了した後に中断されます。

**レスポンス例**
```json
{ "message": "中断要求を受け付けました" }
```

---

### `linebreak_mode` の動作

| 値 | 動作 |
|---|---|
| `none` | 各ブロックを 1 行として `\n` で区切る |
| `paragraph` | 同一タイプのブロックを連結、タイプが変わると空行を挟む |
| `compact` | すべてのブロックを区切りなしに結合 |

---

## 設定リファレンス

`config.toml`（リポジトリルート）で設定します。存在しない場合はデフォルト値で起動します。

```toml
[server]
host = "127.0.0.1"   # 外部公開時は "0.0.0.0"
port = 7860

[runtime]
device = "auto"       # "auto" / "amdgpu" / "vulkan" / "cpu"

[processing]
page_workers      = 2      # PDF 並列処理ページ数 ＝ DEIM インスタンス数
batch_inference   = "auto" # "auto"（GPUバックエンドで有効） / "true" / "false"
max_batch         = 16     # PARSEQ バッチサイズ上限（VRAM 使用量に影響）
precision         = "auto" # "auto"（GPU→fp16、CPU→fp32） / "fp16" / "fp32"

[vram]
reload              = "never"  # "never" / "always" / "auto"
reload_threshold_gb = 0.0      # "auto" 時のリロード閾値 GB（0 = 無効）

[cpu]
intra_op_threads = 1   # CPU モード時のスレッド数（-1 で onnxruntime 自動）
```

---

---

## アーキテクチャ

### ジョブ処理フロー

```
POST /api/jobs
  └─ file_bytes をメモリ読み込み
  └─ _jobs[job_id] = { state: "queued", ... }
  └─ ThreadPoolExecutor.submit(_process_sync)  ← 即座に返る
  └─ future.add_done_callback(_on_job_done)
  └─ { job_id, state: "queued" } を返す

_process_sync（ワーカースレッド）
  └─ _inference_lock.acquire()  ← 直列化（同時推論は 1 件のみ）
  └─ state = "processing"
  └─ PDF なら _iter_pdf_pages() でチャンク単位に変換しながら処理
  └─ infer_pages() / infer_image() で推論
  └─ _inference_lock.release()
  └─ results を返す

_on_job_done（コールバック）
  └─ state = "done" / "canceled" / "error"
  └─ results を _jobs[job_id] に格納
  └─ GPUバックエンドなら必要な後処理を実行

GET /api/jobs/{job_id}
  └─ _jobs から state + 完了時は results を返す
```

### スレッド安全性

| モデル | スレッド安全 | 実装 |
|---|---|---|
| PARSEQ 認識器 × 3 | ✅ `InferenceSession.run()` はスレッドセーフ | 全ワーカーで共有 |
| DEIM 検出器 | ❌ `preprocess()` が `self.image_width/height` を書き換える | `queue.Queue` でプール管理 |

同時推論リクエストは `_inference_lock`（`threading.Lock`）で直列化されます。
`_request_executor`（`ThreadPoolExecutor(max_workers=2)`）により最大 2 件まで受付し、2 件目は推論ロック待ち状態（`queued`）でキューイングされます。

### PDF 処理パイプライン

```
PDF Nページ
  └─ _render_worker（バックグラウンドスレッド）
       └─ pypdfium2 でページ画像を生成
       └─ MAX_PAGE_WORKERS 枚ごとにチャンクを queue に投入
  └─ infer_pages（メインワーカースレッド）
       Phase 1: DEIM 並列検出（チャンク内を page_workers 数で並列）
       Phase 2: 全ページ行画像を 1 回の cascade_batch に投入（クロスページバッチ）
       Phase 3: ページ結果を組み立て、進捗を更新
```

レンダリングと推論をオーバーラップさせることで、GPU アイドル時間を削減しています。

### VRAM 管理

`vram.reload` 設定により、リクエスト完了後の PARSEQ ONNX セッションのライフサイクルを制御できます。

| 設定値 | 動作 |
|---|---|
| `never` | セッションを保持し続ける（ウォームキャッシュ、推奨） |
| `always` | 毎リクエスト後にセッションを解放・再ロード |
| `auto` | VRAM 使用量が `reload_threshold_gb` を超えた場合のみ解放 |

DEIM セッションは常に保持されます（入力形状固定で VRAM 使用量が一定なため）。

---

## GPUバックエンド利用

本フォークの主対象はAMD GPU / MIGraphXです。CPUフォールバックとVulkan Execution Providerにも対応します。CUDAは前段フォーク由来の互換経路として残る場合がありますが、既定の説明・実測・運用要件ではありません。

### 対応デバイス

| 設定値 | Execution Provider | 用途 |
|---|---|---|
| `amdgpu` | `MIGraphXExecutionProvider` | AMD GPUの推奨設定 |
| `vulkan` | `VulkanExecutionProvider` | Vulkan対応環境の比較用 |
| `cpu` | `CPUExecutionProvider` | フォールバック・基準測定 |
| `auto` | 利用可能なGPUを自動選択 | AMD/MIGraphXを優先 |

実際に選択されたExecution Providerは`GET /api/status`の`onnxruntime_providers`で確認してください。GPUを使う場合、モデルの初回ロード時にコンパイルが発生することがあります。永続キャッシュを設定し、同じモデル・入力形状・実行環境で再利用してください。
