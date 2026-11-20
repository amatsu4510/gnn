# =============================================================================
# GNN（大域的最近傍法）目標追跡サンプル
#   STEP 1〜3 のデモ実行ファイルとテスト実行ファイルをビルドする。
#   Visual Studio 2022 では「フォルダーを開く」でこのファイルを読み込める。
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(GnnTrackingSample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Eigen ----
#   1) Eigen をインストール済みなら find_package で見つかる
#   2) 見つからない場合は -DEIGEN3_INCLUDE_DIR=<Eigen を展開したフォルダー> を指定する
#      （そのフォルダーの直下に Eigen/Dense があること）
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    # third_party/eigen-3.4.0 に展開してあれば、それを既定値として使う
    set(_eigen_default "")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0/Eigen/Dense")
        set(_eigen_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
    endif()
    set(EIGEN3_INCLUDE_DIR "${_eigen_default}" CACHE PATH "Eigen のフォルダー（直下に Eigen/Dense がある場所）")
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
        message(FATAL_ERROR
            "Eigen が見つかりません。-DEIGEN3_INCLUDE_DIR=<Eigen のフォルダー> を指定してください。")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    # SYSTEM 扱いにして、Eigen 内部の警告を表示しないようにする
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# ---- 警告・文字コード ----
if(MSVC)
    # /utf-8 : ソースコードを UTF-8 として読み、実行時の文字列も UTF-8 にする（日本語コメント・出力のため必須）
    add_compile_options(/utf-8 /W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# ---- 全STEP共通部分 ----
add_library(gnn_common STATIC
    common/HungarianSolver.cpp
    common/GatedAssignment.cpp
    common/ScenarioGenerator.cpp
    common/Printing.cpp
)
target_include_directories(gnn_common PUBLIC common)
target_link_libraries(gnn_common PUBLIC Eigen3::Eigen)

enable_testing()

add_executable(test_hungarian common/test_hungarian.cpp)
target_link_libraries(test_hungarian PRIVATE gnn_common)
add_test(NAME test_hungarian COMMAND test_hungarian)

# ---- STEP ごとの実行ファイルを作る関数 ----
#   step_dir/main.cpp      → <name>       （デモ）
#   step_dir/test_<name>.cpp → test_<name> （テスト）
function(add_step name step_dir)
    set(sources ${ARGN})
    list(TRANSFORM sources PREPEND "${step_dir}/")

    add_executable(${name} ${step_dir}/main.cpp ${sources})
    target_include_directories(${name} PRIVATE ${step_dir})
    target_link_libraries(${name} PRIVATE gnn_common)

    add_executable(test_${name} ${step_dir}/test_${name}.cpp ${sources})
    target_include_directories(test_${name} PRIVATE ${step_dir})
    target_link_libraries(test_${name} PRIVATE gnn_common)
    add_test(NAME test_${name} COMMAND test_${name})
endfunction()

add_step(step1 step1_basic_gnn
    GNNAssociator.cpp)

add_step(step2 step2_multi_frame
    GNNAssociator.cpp Track.cpp TrackManager.cpp ReportPrinter.cpp)

add_step(step3 step3_kalman
    GNNAssociator.cpp KalmanFilter.cpp Track.cpp TrackManager.cpp ReportPrinter.cpp)
