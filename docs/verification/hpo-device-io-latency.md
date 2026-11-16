# 推論・CPU・待ち時間とHTTP接続再利用の調査

## 結論

HTTPセッション／TCP接続を毎回作り直してはいなかった。32試合・3346往復で
AsyncClient生成16回、TCP connect成功16回、接続失敗0。同じstreamを再利用した
応答3330回で、各クライアントは1本のHTTP/1.1接続を使っていた。groupごとに
1clientが2試合を扱うコード上の寿命と実測が一致する。TCP生成区間の合計は
0.044029秒である。

一方、「推論」の旧policy_msはGPU単独ではなかった。neural方策465回の
host経過8.423204秒には、GPU呼出し・転送・同期に
関係するhost窓5.859108秒、token化とCPU samplingのhost窓1.292278秒、
それ以外の検証などのhost残差1.271818秒がある。
GPU関連窓にもCPU側の投入処理が含まれるので、GPUカーネルのbusy時間とは扱わない。
非neural方策146回のpolicy時間2.248332秒も別に示す。

## 条件・修正・検証

- MLP、192幅・2層、seed17、同時2試合／4worker、16試合×2更新、epoch1。
  以前と同じD1,000,000、600秒期限、Source2GiB/numeric512MiB/collector2GiB、
  parent/save1GiBを維持。親は正常なincomplete/update_capでありHPO完了ではない。
- コードrevision `9829e62ae104748aebfbf6cf42d62afa17bb4509`。
  production変更は1行のみ。すでにpolicy.last_timingで計測し、receiverの既存
  optional契約が許可していた6項目をCardInstanceEpisode.selectから渡す。
  新しいprotocol項目、同期、Tensor演算、sampling/RNG、creatorは追加・変更しない。
- REDでneural3caseの欠落を再現し、CPU195PASS/既存CUDAskip12、実HOST CUDA12PASS
  （51deselected、13.69秒）、scoped Ruff/mypy/差分検査と独立レビューPASS。
- parent-onlyの使い捨てobserverは同じ実際の関数を1回呼び、8MiBの共通上限、
  weakなstream識別、PID/starttime照合、元のhook/trace/error/taskを保持。
  GPU同期やnetwork request/JSON parseを追加しない。本文・header・queryは記録しない。
- 通常計測と詳細計測を別の新規runで順次実行。両方32自然試合、611行動、314学習
  decision、2更新。実receiptとstrict2GiB保存原本監査は両方PASS、ファイルstat不変。
- 詳細runはdrop/観測失敗/欠落0、465neural SELECTすべて6項目あり。両runの終了、
  CPU/接続集計と区間合計の独立saved-resultレビューもREADY。
- 凍結SHA256 `e80d3cf9800d6669b521a96f5b31b4b3fa326289bece733818174822c7564042`。
  生span・CPUcounter・接続集計・設定・receiptは
  `runs/hpo-producer-owned/assessment/device-io-detail-v1/`。

## 1. neural推論と、それ以外の方策を分ける

以下は465回のneural方策選択の**延べhost経過時間**。実CPU秒／GPU busy秒ではない。

| 区分 | 内容 | 延べ秒 |
| --- | --- | ---: |
| GPU関連host | Tensor構築／転送投入 | 0.951745 |
| GPU関連host | forward投入側 | 4.653921 |
| GPU関連host | 既存GPU同期・結果のfinite確認 | 0.152808 |
| GPU関連host | 結果回収・CPUコピー | 0.100634 |
| **GPU関連host小計** | **CPU投入や待ちを含む** | **5.859108** |
| CPU寄りhost | token化 | 1.273823 |
| CPU寄りhost | CPU sampling | 0.018455 |
| その他host残差 | policyの入口・候補やselectionの検証など | 1.271818 |
| **neural方策全体** | **465回** | **8.423204** |
| 非neural方策 | random/heuristic等、146回、上記とは別 | 2.248332 |

forward投入はCPUのCUDA呼出しも行う。Tensor構築／コピー投入は先行stream待ちを
含み得る。結果回収にもGPU演算とcopy待ちがあり、sync窓にはfiniteチェックがある。
したがって5.859秒をGPU演算正味、1.292秒を実CPU時間と呼ぶことはできない。
前回の混在した「推論10秒」から切り分けた、実際のhost側処理境界である。

## 2. CPUは実行時間、待ちは観測区間として別に示す

### 実CPU使用時間

| 対象 | user CPU秒 | system CPU秒 | 合計CPU秒 | 範囲 |
| --- | ---: | ---: | ---: | --- |
| 収集の親プロセス | 16.428788 | 0.875936 | 17.304724 | 2回のCOLLECT全体、全thread |
| actor4プロセスの観測窓合計 | 15.680000 | 3.130000 | 18.810000 | 各batchで最初のSELECT前から最後のSELECT後、途中の制御処理も含む |

actor窓はPID・開始時刻・batchで分けた8区間。100Hzカウンタの端点誤差を保守的に
見た合計上限は0.32秒。個々のSELECTのCPU差分合計15.63秒には、多数の端点による
24.44秒の保守的な誤差上限があるため、細かいCPU割合の根拠には使わない。
親は高分解能process_time_nsとgetrusageを使用する。親の同期区間ではthread CPUも
計測したが、入れ子のものは重ねて加算しない。プロセスが並行するため、親とactorの
CPU秒を収集wall 27.229701秒へ足すこともできない。

### 待ちを含む区間

| 観測区間 | 延べhost秒 | 注意 |
| --- | ---: | --- |
| HTTPの応答header待ち | 5.745246 | network/server/OS schedulingを分離しない |
| HTTP応答本文受信 | 0.105768 | buffered readの観測区間 |
| HTTP要求header送信 | 0.673209 | 送信・scheduling等を含む |
| HTTP要求本文送信 | 0.324460 | 同上 |
| 既存GPU同期・結果確認 | 0.152808 | 上のGPU関連窓の内数 |

HTTP traceはHTTP往復全体9.699979秒の内側に
あり、GPU同期もneural窓の内側にある。別の試合が並行してCPUで動く間のawaitも
あり、これらを合算して「総I/O wait」とはしない。

**wall−CPUを全部I/O waitにすることはできない。** 残りにはGPU完了待ち、IPC、
OS scheduling、実際のI/O待ちが入る。今回、実CPUと観測済みの待ち区間は分けたが、
純GPUカーネルbusy時間と、各processの正味I/O wait秒は未計測のまま残している。

## 3. HTTPを毎回張り直しているか

| 実測項目 | 値 |
| --- | ---: |
| AsyncClient生成／正常close | 16 /16 |
| TCP connect開始／完了／失敗 | 16 /16 /0 |
| 観測した一意のnetwork stream | 16 |
| request／response | 3346 /3346 |
| 同じstreamを再利用した応答 | 3330 |
| stream識別の欠落 | 0 |
| TCP接続生成区間の延べ時間 | 0.044029秒 |

同じHTTP/1.1接続をgroup内の2試合で再利用している。HTTPX0.28.1/httpcore1.0.9の
現在の既定はkeepalive20本、保持5秒。今回の実測では1clientあたりconnect1回で、
毎requestのsession/TCP張り直しは遅さの主因ではない。

request eventhookが既存traceと連結して実connectを数え、response hookが
network_streamをweakな一意serialで識別した。実objectを強参照で保持して寿命を
延ばさず、以前のhookやtraceの戻り・例外も変えない。TCP closeはrequesttraceから
観測できないため未計測。http11.response_closedをTCP closeとして数えていない。

## 限界と次の判断材料

- detailed runの1試合は平均1.541704秒、
  中央値0.939511秒、p95 3.643164秒。
  32試合・2並列での収集wallは27.229701秒。
- off/onのプロセス全体は39.080275／39.400153秒。1組だけなので差を計測負荷や
  高速化の確定値とはしない。本文やゲーム内容、モデル計算を変えない診断である。
- CPU使用は大きく、GPU関連窓にもCPU投入を含む。GPU正味時間が必要なら、実actorの
  明示的なCUDA event計測、または同じcheckpoint/input/cache進行を使う別の許可済み
  推論replayが次の計測になる。replay値を今回の実wallから差し引くことはしない。
- 現時点では最適化を加えていない。接続張り直し仮説は今回の条件では外せる。
  推論CPU側とIPC/decode/scheduling、候補生成や記録処理を優先して見る根拠になる。
