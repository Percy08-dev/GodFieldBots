# Tokio・PyO3の公式資料とプレイヤー実行方式の比較条件

2026-11-19確認。対象は[単一PythonプロセスとRustプレイヤーの設計調査](rust-player-runtime-research.md)。提示されたベストプラクティスを、ライブラリが保証する動作、GodFieldで必要になる処理、実験によって決める選択に照らした。

**設計に反映する結論。** 最初の比較は、A「1試合1 Tokioタスク」とB「固定Rustワーカーに複数試合を割り当てる方式」で行う。両方に同じ推論器、同じバッチ形成担当、同じ保存・HTTP・候補生成処理を接続する。CPU処理がイベント配送を妨げる実測があれば、C「Tokio＋明示したサイズのRayonプール」を加える。これらの優劣は公式資料からは決まらない。今回は既存のCPUモデルとスレッド型推論サービスの動作を確認した段階で、A/B/Cのスループット比較は未実施。

## 指針の照合結果

| 論点 | 公式資料で確認したこと | このプロジェクトでの使い方 |
| --- | --- | --- |
| I/O待ち・CPU処理 | Tokioは多数のI/O待ちに向き、CPU並列だけが目的ならRayonが候補。長くyieldしない計算は他のfutureの進行を遅らせる。[Tokio Tutorial](https://tokio.rs/tokio/tutorial#when-not-to-use-tokio) | 短い配送・状態更新は通常の関数で行う。判断には候補探索があり、現行の探索制限は1秒。1ターン全体を常に短いと仮定しない。Simulatorが状態遷移を所有するので、各ターンは観測・必要なヒント・Actの受理を経て進む。連続して処理できるのは、既に届いたイベントに基づくローカル処理。 |
| ブロッキング処理 | `spawn_blocking` は完了する同期処理をblocking poolへ送る。実行開始後の `abort` は効かず、`shutdown_timeout` 後も処理が残り得る。常駐ループには専用スレッドが候補。[Tokio 1.54.1 API](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html) | 毎ターン無条件で投入すると待ち行列とスレッド数が増える。CPU重処理の同時数を制御し、常駐writerなどは専用所有者が終了確認を行う。締切を超えた重処理の結果は失効させ、使用中のバッファは実処理終了まで保持する。 |
| キュー・応答 | bounded `mpsc` は複数送信・単一受信、上限はメッセージ件数。`oneshot` で個別に応答できる。正のlimitでの `recv_many` は利用可能なメッセージを上限まで返し、空の間は少なくとも1件または閉鎖まで待つ。満杯になるまで待つAPIではない。[Tokio Channels](https://tokio.rs/tokio/tutorial/channels)・[Receiver::recv_many](https://docs.rs/tokio/latest/tokio/sync/mpsc/struct.Receiver.html#method.recv_many) | 単一受信者をバッチ形成担当にできる。独自lock-freeキューは初期比較に不要。キュー待ち中の送信者が既に大きな配列を保持する場合もメモリに計上する。件数上限に加え実バイトと所有期間を記録する。 |
| 締切 | `timeout_at` は絶対時刻でfutureの待ちを終える。yieldしない同期処理の実行時間をプリエンプトする仕組みではない。[timeout_at](https://docs.rs/tokio/latest/tokio/time/fn.timeout_at.html)・[timeout](https://docs.rs/tokio/latest/tokio/time/fn.timeout.html) | 形成開始点を明記し、最初の有効要求からB件または同じ絶対締切で発行する。現行TorchRLとの比較では最初のdrain後を起点とする。追加受信のたびに窓を引き直さない。要求自身の有効期限は別に維持し、返送後・サンプリング前にも検査する。 |
| GIL | PyO3 0.26で `with_gil → attach`、`allow_threads → detach` と改称。`detach` 中のRust処理はPython APIに触れず、GIL有効環境で別スレッドがPythonを使える。Mutex・GILの取得順が逆転するとデッドロックし得る。[PyO3移行ガイド](https://pyo3.rs/main/migration.html)・[0.30.1 Python token](https://docs.rs/pyo3/0.30.1/pyo3/marker/struct.Python.html#deadlocks)・[Parallelism](https://pyo3.rs/main/parallelism) | Rustでキュー待ち・joinするPython側の呼出は必要に応じてdetachし、Rustの試合状態のロックを持ったままPythonへ戻らない。GILを外しても同じPython担当スレッドが同期的に塞がれば、その担当にしか実行できない推論は進まない。推論要求を処理できる担当を常に残す。Rustの並列度は別途設定する。 |
| CPU配列・GPU転送 | rust-numpyの `IntoPyArray`、`from_owned_array` 等では対応するRust割り当てをPython側が所有でき、`torch.from_numpy` は対応dtypeのCPUメモリを共有する。個別の `pin_memory().to(..., non_blocking=True)` がかえって遅い例もある。[rust-numpy 0.30.0](https://docs.rs/numpy/latest/numpy/array/struct.PyArray.html)・[PyTorch 2.14 from_numpy](https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html)・[pinningの実測と同期条件](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) | 実運用のTorchは2.14.1+cu132。2.15チュートリアルの速度を流用せず、最終dtypeで作った配列・owner寿命・通常転送を基準にする。候補数とイベント長の異なる要求には連続領域＋offset/lengthを比較。密なpadding、コピー一括化、pinningは独立した変数とし、他席の長さがスコアを変えないことを確認する。 |
| runtimeとプロセス | Tokioは同期ホストと共存できる。current-thread runtimeは駆動中だけタスクが進む。Tokioを使った親からforkして即execせず、子でTokioを使う構成はサポート外。PyTorchもCUDAを使う子プロセスではspawn/forkserverを指定する。[同期コードとの接続](https://tokio.rs/tokio/topics/bridging)・[TokioのUnix fork制約](https://docs.rs/tokio/1.54.1/tokio/runtime/#unix-fork)・[PyTorch 2.14のmultiprocessing](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html) | 学習実行の寿命でruntimeを保持する。推論処理中もHTTPや停止通知が進むようにする。外部Simulatorは別実行ファイルとして起動する。旧worker起動経路を併用する移行試験では、初期化済みTokio/CUDAのfork継承を避ける。 |
| 終了 | Tokio公式は、停止の決定、通知、終了待ちを区別する。`CancellationToken`・`TaskTracker` はtokio-utilの機能。`TaskTracker::close()` 後にもspawnは可能で、閉鎖かつ追跡対象が空になった時に `wait()` が完了する。[Graceful Shutdown](https://tokio.rs/tokio/topics/shutdown)・[TaskTracker](https://docs.rs/tokio-util/latest/tokio_util/task/struct.TaskTracker.html#method.close) | 新規受付を止め、要求を完了または明示的に失効させ、oneshot受信側もエラーで復帰できるようにする。チャネルの未処理メッセージと予約済みpermitを解放し、CPUタスク・保存・GPU処理の実完了を照合してからPython側を終了する。`TaskTracker` だけではGPU転送や永続化の完了を証明できない。 |

## asyncioをつなぐ条件

`pyo3-async-runtimes` **0.30.0は2026-10-15公開**。PyO3依存 `^0.30` は本リポジトリの0.30.1とsemver上整合する。Tokio統合は `tokio-runtime` featureを使い、`future_into_py` はRust FutureをPython awaitableへ、`into_future` はPython awaitableをRust Futureへ変換する。`init`／`init_with_runtime` はruntime構成、属性付きmainはRustが起動を所有する場合の入口。既存のPython起動型拡張にmain属性を加える必要はない。PyO3本体の `async fn` 公開は0.30.1の `experimental-async` featureに属し、await時のdetachやメソッドの借用寿命に制約がある。[0.30.0リリース](https://github.com/PyO3/pyo3-async-runtimes/releases/tag/v0.30.0)・[crateとfeatures](https://docs.rs/crate/pyo3-async-runtimes/0.30.0)・[TokioモジュールのAPI](https://docs.rs/pyo3-async-runtimes/latest/pyo3_async_runtimes/tokio/index.html)・[PyO3本体のasync](https://pyo3.rs/main/async-await)

Pythonの既存asyncioタスクがRustのFutureを直接awaitする要件があれば、このcrateが候補になる。固定したPython推論担当がGILを解放してバッチを待ち、結果をRustのキューに返す方式もある。後者はPythonのasyncioループ上で同期的に待つと他のタスクを止めるので、どのスレッドがループと推論を駆動するかを明示する。提示例の `async def` の中で同期的に呼ぶ `model(...)` やCPUへの結果取得も、その呼出から戻るまで同じイベントループを進めない。Pythonのasyncioを使う場合はこの区間も測る。Bots nativeの現行Cargo.lockにTokio／pyo3-async-runtimesは入っておらず、この調査ではいずれも新規にビルドしていない。

upstreamの **issue #90** は2026-08-04の報告で、確認時点でOpen。報告者は**0.29系**で、キャンセルされたFutureの完了通知が閉鎖済みasyncioループの `call_soon_threadsafe` に届き、`RuntimeError: Event loop is closed` が出るタイミング競合を記載している。0.30.0での再現性は未確認。導入する場合の終了テストに採用する。[upstream issue #90](https://github.com/PyO3/pyo3-async-runtimes/issues/90)

## AとBを比較するときに固定するもの

- **推論担当を共通化する。** Rustでバッチを確定する構成ならPythonはそのバッチを処理する。TorchRLで形成する構成ならRustは個別要求を引き渡す。両方で形成待ちを行うと比較する待機窓が変わる。Pythonから返すのは各要求の順序付きスコアとvalue。各プレイヤーが自分のRNGで選択し、SimulatorのAct ACKを照合する。
- **Wの意味をそろえる。** AはTokioタスクが試合を保持し、Bは固定ワーカーが試合状態を所有し完了イベントで再開する。Tokioのスケジューラスレッド、追加のCPUプール、writer、PyTorchのintra-opスレッド、外部Simulatorを記録し、総 CPU 資源をそろえる。BのワーカーはHTTP・推論・保存の応答待ち中にも他試合を進める。work stealingのコストと固定割当の偏りは、両方とも測定対象。[PyTorch set_num_threads](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html)
- **計測条件。** 同一の試合・候補生成・checkpoint・seed・学習予算・実効バッチ条件で、起動と定常を分ける。試合/秒、全参加者のdecision/秒、MLP要求数、実効バッチの分布、キュー時間と推論往復のp50/p95/p99、CPU利用率・実スレッド数、GPU利用率・H2D時間、PyO3呼出数/要求、PSSと保存キューのバイト数・滞留時間を記録する。短いカーネルのGPU利用率はサンプリングで取り逃がし得るため、CUDAイベントを使う計測は別枠にする。tracingのspanは試合ID・要求IDで追跡し、他席の非公開データを参加者へ返さない。[Tokio Tracing](https://tokio.rs/tokio/topics/tracing)
- **選定方法。** A/Bを同じ実機で交互に動かし、論理分離・学習同値・遅延応答・停止・保存の照合を通った結果だけで判断する。CPU探索がTokioワーカーを長く占有した場合に、同じ候補処理をRayonに移したCを比較する。Python境界の呼出回数や平均バッチサイズだけから高速化率は推定できない。
