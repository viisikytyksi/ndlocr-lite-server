# NDLOCR-Lite Server（AMD / MIGraphX）

[ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite) をHTTPサービスとして利用するためのフォークです。

## このリポジトリの経緯

実装の経緯は次の通りです。

```text
ndl-lab/ndlocr-lite
  -> ponpaku/ndlocr-lite-server       （CUDA向けサーバーフォーク）
  -> viisikytyksi/ndlocr-lite-server  （本リポジトリ。AMD/MIGraphX対応）
```

本リポジトリは、前段のサーバー実装を引き継ぎつつ、AMD GPUで実用的に動かすことを主目的にしています。CUDA向けフォークに由来するコードや説明が残っている場合がありますが、CUDAの性能値を本リポジトリの実績として扱いません。

## 現在の対応状況

- **AMD / MIGraphX**: AMD Radeon RX 7900 XTX、`gfx1100`、ROCm 10環境で実機検証済み
- **CPU**: フォールバックとして利用可能
- **Vulkan**: 実行経路とベンチマーク用ツールを含む。LLMserverでの速度比較は未確定
- **CUDA**: 前段フォーク由来の互換コードが残る場合はあるが、現行の主対象・検証結果ではない

### 上流README準拠のAMD実機ベンチマーク

上流READMEの`tools/benchmark_batch.py`に合わせ、LLMserver（RX 7900 XTX / `gfx1100` / ROCm 10）で、256×16のPARSEQ行画像16枚、ウォームアップ2回・測定5回で実行しました。各値は同じ16枚を1回処理する平均時間です。

| 実行環境 | 精度 | 逐次: 16行 | バッチ: 16行 | バッチ速度向上 |
|---|---|---:|---:|---:|
| CPU | FP32 | 341.9 ms (21.4 ms/行) | 292.8 ms (18.3 ms/行) | 1.17× |
| AMD / MIGraphX | FP16 | 105.8 ms (6.6 ms/行) | 105.4 ms (6.6 ms/行) | 1.00× |

AMD行は安定運用中の`max_batch=1`・既存MIGraphXキャッシュを使った測定であり、`read_batch()`は16回の単行推論に分割されます。そのため、AMDの真のbatch16性能ではありません。真のbatch16は本番キャッシュと分離して120秒制限で試しましたが、MIGraphXコンパイル完了前に終了したため未掲載です。上表のCPUバッチとAMD行を同じ意味のバッチ性能として比較しないでください。

再測定コマンド:

```bash
python tools/benchmark_batch.py --device CPU --repeat 5 --warmup 2
python tools/benchmark_batch.py --device amdgpu --precision fp16 --repeat 5 --warmup 2 --max-batch 1
```

この測定はPARSEQ行認識のマイクロベンチマークであり、DEIM検出・PDF変換・HTTP処理を含むPDF全体の処理時間ではありません。
## 主な機能

- 画像・PDFのアップロードとOCR
- FastAPI + UvicornによるWeb UI / JSON REST API
- PDFページの画像化とページ単位の並列処理
- PARSEQ行認識のバッチ処理（`processing.max_batch`で上限指定）
- ONNX Runtime Execution Providerの実体確認と`/api/status`での状態表示
- MIGraphXのコンパイル済みモデルキャッシュによる再起動後の再利用

## セットアップ

Python 3.11〜3.13を使用してください。AMD GPU環境では、ROCmとMIGraphXを環境に合わせて先に用意し、その後に依存関係をインストールします。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-amdgpu.txt
cp config.toml.example config.toml
```

`config.toml`では、AMD GPUを使う場合は`[runtime]`の`device = "amdgpu"`を指定できます。最初は`processing.max_batch = 1`で動作確認し、安定してから負荷を上げてください。ROCm/MIGraphXのパッケージ名やインストール方法は、GPUとROCmの組み合わせによって異なります。

## 起動

```bash
python server/main.py
```

既定の待受ポートは`7860`です。Tailscaleなど限定したネットワークから利用する場合は、`config.toml`の`[server] host`を実際の待受アドレスに設定してください。

起動後の確認:

```bash
curl http://127.0.0.1:7860/api/status
```

## APIの概要

- `GET /api/status`: モデル、既定デバイス、利用可能なExecution Providerを返す
- `POST /api/jobs`: `file`（画像またはPDF）をmultipartで送信してジョブを作成
- `GET /api/jobs/{job_id}`: ジョブの状態と結果を取得

詳細な設定・API仕様・アーキテクチャは[dev-doc.md](./dev-doc.md)を参照してください。

## コンパイルキャッシュ

MIGraphXは初回実行時にモデルをコンパイルすることがあります。`ORT_MIGRAPHX_MODEL_CACHE_PATH`を設定すると、同じモデル、入力形状、実行オプション、ROCm/ONNX Runtime/MIGraphX環境で、生成済みキャッシュを再利用できます。環境や設定を変えた場合は再コンパイルになることがあります。

初回起動で問題が出る場合は、`max_batch=1`、`page_workers=1`、`MIOPEN_FIND_MODE=FAST`から切り分けてください。キャッシュを作った後の通常の再起動で、毎回コンパイルすることを前提にはしていません。

## 参考

- [ndl-lab/ndlocr-lite](https://github.com/ndl-lab/ndlocr-lite)
- [ponpaku/ndlocr-lite-server](https://github.com/ponpaku/ndlocr-lite-server)
- [viisikytyksi/ndlocr-lite-server](https://github.com/viisikytyksi/ndlocr-lite-server)
- [ONNX Runtime MIGraphX Execution Provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

## ライセンス

原典のライセンスおよびモデルの利用条件に従います。詳細は原典リポジトリを確認してください。
