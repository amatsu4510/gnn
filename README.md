# GNN（Global Nearest Neighbor：大域的最近傍法）目標追跡 C++ サンプル

画像解析で抽出された「局所最明点」をフレームごとに入力し、GNN で時系列の対応付けを行う学習用サンプルです。
C++17 + Eigen + 標準ライブラリのみで、ハンガリアン法も自作しています。

## 3つのSTEP

| STEP | フォルダー | 内容 | 解説 |
|---|---|---|---|
| 1 | `step1_basic_gnn/` | 2フレーム間の GNN（ユークリッド距離・ゲーティング・ハンガリアン法） | [docs/02_step1.md](docs/02_step1.md) |
| 2 | `step2_multi_frame/` | トラックID・α-β による位置速度推定・等速予測・Tentative/Confirmed/Coasting の管理 | [docs/03_step2.md](docs/03_step2.md) |
| 3 | `step3_kalman/` | カルマンフィルタ・マハラノビス距離²・カイ二乗ゲート・Joseph 形式 | [docs/04_step3.md](docs/04_step3.md) |

全STEP共通の部品（入力データ型・ハンガリアン法・ダミー拡張・疑似データ生成など）は `common/` にあります。
各STEPは `common/` と自分のフォルダーだけでビルドでき、他のSTEPには依存しません。

## 解説資料

| 資料 | 内容 |
|---|---|
| [docs/01_design.md](docs/01_design.md) | 実装方針、ファイル構成、クラスの責務と変更理由、クラス図・処理フロー図・状態遷移図・シーケンス図（Mermaid） |
| [docs/02_step1.md](docs/02_step1.md) | STEP 1：距離・ゲート・GNN の目的関数と C++ コードの対応 |
| [docs/03_step2.md](docs/03_step2.md) | STEP 2：等速予測・α-β フィルタ・トラック管理 |
| [docs/04_step3.md](docs/04_step3.md) | STEP 3：カルマンフィルタの数式とコード、Eigen の行列の意味と次元、Eigen API の使用意図 |
| [docs/05_hungarian.md](docs/05_hungarian.md) | ハンガリアン法の内部処理、ダミー行・列、禁止要素の扱い、計算量、テスト |
| [docs/06_parameters_and_limits.md](docs/06_parameters_and_limits.md) | パラメータの意味と調整方法、接近・交差時の限界（実験結果）、制約と改善方法、想定される実行結果 |
| [docs/07_reference_review.md](docs/07_reference_review.md) | 添付資料の技術的な誤り・注意点と、本サンプルでの修正内容 |

## 添付資料の誤りと修正（概要）

詳細は [docs/07_reference_review.md](docs/07_reference_review.md)。

1. **p.7**：マハラノビス距離の式 $d^2 = \mathbf{r} S^{-1} \mathbf{r}^\top$ は、r が列ベクトルだと次元が合わない → $\mathbf{r}^\top S^{-1} \mathbf{r}$
2. **p.16**：Q を固定の引数で受け取る設計では、フレーム間隔が変わったときに正しくない → Δt から毎回計算
3. **p.20**：カルマンフィルタの更新関数で対応回数（hits/misses）まで更新している → 責務を分離
4. **p.10 / p.15**：「直近5回中3回で確定」に対し、累計回数しか保持していない → 対応履歴を保持
5. **p.10 / HTML**：仮トラックの削除条件がない → 追加
6. **p.17**（注意点）：ゲート 9.21（99%）は値としては正しいが、実験ではトラックの分裂が 100 回中 12 回起きた → 既定を 99.9%（13.82）に

## Visual Studio 2022 での Eigen の導入

Eigen はヘッダーのみのライブラリなので、ダウンロードして展開し、インクルードパスを通すだけで使えます。

1. https://eigen.tuxfamily.org/ （または https://gitlab.com/libeigen/eigen/-/releases ）から **Eigen 3.4.0** の zip をダウンロード
2. 任意の場所に展開する（例：`C:\libs\eigen-3.4.0`。このフォルダーの直下に `Eigen\Dense` がある）

vcpkg を使っている場合は `vcpkg install eigen3` でも導入できます（CMake の `find_package(Eigen3)` で自動的に見つかります）。

## ビルド・実行方法

### 方法A：Visual Studio 2022 で CMake プロジェクトとして開く（推奨）

1. Visual Studio 2022 の「C++ によるデスクトップ開発」ワークロードに含まれる「Windows 用 C++ CMake ツール」をインストールしておく
2. 「ファイル」→「開く」→「フォルダー」でこのフォルダー（`CMakeLists.txt` がある場所）を開く
3. Eigen を vcpkg で導入していない場合は、「プロジェクト」→「CMake 設定」で「CMake コマンド引数」に次を追加する
   ```
   -DEIGEN3_INCLUDE_DIR=C:/libs/eigen-3.4.0
   ```
4. 上部の「スタートアップ項目の選択」で `step1.exe` などを選んで実行（F5 / Ctrl+F5）
5. テストは「テスト」→「CTest の実行」、または各 `test_*.exe` を実行

`CMakeLists.txt` で MSVC には `/utf-8`（日本語のソース・出力のため必須）、`/W4`、`/permissive-` を指定しています。

### 方法B：ソリューション（.sln）を手で作る場合

実行ファイルごとに1つの「空のプロジェクト」を作ります（`main.cpp` とテストはそれぞれ `main` 関数を持つため、同じプロジェクトには入れられません）。

| プロジェクト | 追加するファイル |
|---|---|
| step1 | `common/` の `.cpp`（`test_hungarian.cpp` を除く4つ）+ `step1_basic_gnn/main.cpp` + `step1_basic_gnn/GNNAssociator.cpp` |
| step1_test | step1 の `main.cpp` を `step1_basic_gnn/test_step1.cpp` に置き換えたもの |
| step2 | `common/` の `.cpp`（同上）+ `step2_multi_frame/` の `.cpp` のうち `test_step2.cpp` 以外すべて |
| step2_test | step2 の `main.cpp` を `test_step2.cpp` に置き換えたもの |
| step3 / step3_test | step2 / step2_test と同様に `step3_kalman/` から |
| test_hungarian | `common/` の `.cpp` すべて（`test_hungarian.cpp` を含む） |

各プロジェクトのプロパティ（構成：すべての構成、プラットフォーム：x64）で次を設定します。

| 項目 | 設定 |
|---|---|
| 構成プロパティ → 全般 → C++ 言語標準 | ISO C++17 標準 (/std:c++17) |
| C/C++ → 全般 → 追加のインクルード ディレクトリ | `C:\libs\eigen-3.4.0;$(SolutionDir)common;$(SolutionDir)step1_basic_gnn`（STEP に合わせて変更） |
| C/C++ → コマンド ライン → 追加のオプション | `/utf-8` |

`/utf-8` を付けないと、日本語環境（CP932）では UTF-8 のソースが正しく読まれず、警告 C4819 や文字化け、コンパイルエラーの原因になります。
コンソールへの日本語出力は、`setupConsole()`（`common/Printing.cpp`）で `SetConsoleOutputCP(CP_UTF8)` を呼んで UTF-8 にしています。

### 方法C：コマンドライン（CMake）

Eigen を `third_party/eigen-3.4.0/`（直下に `Eigen/Dense`）に展開しておけば、`-DEIGEN3_INCLUDE_DIR` を省略しても自動的に使われます。

```bash
cmake -S . -B build -DEIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release        # テスト
./build/step1
./build/step2 overall                    # overall / crossing / approach / uneven
./build/step3 approach
```

### 方法D：g++ で直接ビルド（例：STEP 3）

```bash
g++ -std=c++17 -O2 -I/path/to/eigen-3.4.0 -Icommon -Istep3_kalman \
    step3_kalman/main.cpp step3_kalman/GNNAssociator.cpp step3_kalman/KalmanFilter.cpp \
    step3_kalman/Track.cpp step3_kalman/TrackManager.cpp step3_kalman/ReportPrinter.cpp \
    common/HungarianSolver.cpp common/GatedAssignment.cpp common/ScenarioGenerator.cpp common/Printing.cpp \
    -o step3
```

## 実行ファイル

| 実行ファイル | 内容 |
|---|---|
| `step1` | デモ1：資料 p.8 の例 / デモ2：貪欲法との比較 / デモ3：疑似データ5フレーム |
| `step2 [シナリオ]` | STEP 2 の追跡。各フレームの入力・予測位置・ゲート・コスト行列・対応付け・合計コスト・未割り当て・更新後のトラック・生成/削除を表示し、最後に目標ごとの ID 維持を集計 |
| `step3 [シナリオ]` | STEP 3 の追跡。STEP 2 の表示に加え、位置σ・残差σ（ゲートの大きさ）を表示 |
| `test_hungarian` / `test_step1` / `test_step2` / `test_step3` | 単体テスト（失敗があれば終了コード 1） |

シナリオ：`overall`（既定：新規出現・一時的な未検出・消失・誤検出・順序入れ替え）、`crossing`（X字の交差）、
`approach`（浅い角度の接近。ID が入れ替わる GNN の限界の例）、`uneven`（フレーム間隔が一定でない）

## テストケースとの対応

| 課題のテストケース | テスト |
|---|---|
| 1. 複数目標の通常移動 | test_step1「1_複数目標の通常移動」, test_step2/3「1_2_通常移動と順序入れ替えでIDを維持」 |
| 2. 検出順序が変わる | test_step1「2_検出順序がフレームごとに変わる」, test_step2/3「1_2_…」 |
| 3. 貪欲法では最適にならない | test_hungarian「資料の例_貪欲法では最適にならない」「貪欲法では最適にならない_3x3」, test_step1「3_…」 |
| 4. 新規目標の出現 | test_step1「4_…」, test_step2/3「4_新規目標は仮トラックから確定トラックになる」 |
| 5. 一時的な未検出 | test_step1「5_6_…」, test_step2/3「5_一時的な未検出はCoastingで維持し再検出で同じIDに戻る」 |
| 6. 完全な消失 | test_step2/3「6_完全な消失は規定回数の未検出後に削除」 |
| 7. ランダムな誤検出 | test_step1「7_…」, test_step2/3「7_ランダムな誤検出から確定トラックが生まれない」 |
| 8. 接近・交差 | test_step2/3「8_X字の交差ではIDを維持」「8_浅い角度の接近ではIDが入れ替わり得る（GNNの限界）」 |
| 9. トラック数と最明点数が異なる | test_hungarian「矩形行列_…」, test_step1「9_…」, test_step2/3「9_10_…」 |
| 10. トラックまたは最明点が0件 | test_hungarian「空の入力」, test_step1「10_…」, test_step2/3「9_10_…」 |
| 11. すべての観測がゲート範囲外 | test_step1「11_…」, test_step2/3「11_…」 |
| 12. フレーム間隔が一定でない | test_step2/3「12_フレーム間隔が一定ではない」, test_step3「KF_行列F_H_Qの構造」 |
| ハンガリアン法と全探索の比較 | test_hungarian「全探索との比較_ランダム行列」（3000件）「ダミー拡張_全探索との比較」（2000件） |

乱数はすべてシード固定です。`std::normal_distribution` などの分布クラスは標準ライブラリごとに出力が異なるため、
`std::mt19937` の出力から一様乱数・正規乱数を自前で作り、処理系が違っても同じデータになるようにしています。

## 検証状況

| 環境 | 結果 |
|---|---|
| Linux（WSL2）, GCC 13.3, Eigen 3.4.0, CMake 4.x | ビルド成功（`-Wall -Wextra -pedantic` で警告なし）。テスト 4本すべて成功（11 + 11 + 12 + 19 件）。全デモの実行を確認 |
| 同上 + AddressSanitizer / UndefinedBehaviorSanitizer + `_GLIBCXX_DEBUG` | テスト・デモともにエラーなし |
| **Windows / Visual Studio 2022（MSVC）** | **未実施**（この環境では実行できなかったため）。MSVC 特有の点（`/utf-8`、`NOMINMAX`、`M_PI` を使わない等）は考慮してあるが、実機での確認をお願いします |
