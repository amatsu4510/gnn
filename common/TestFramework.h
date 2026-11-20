#pragma once
// ============================================================================
// TestFramework.h
//   標準ライブラリのみで書いた最小限のテスト用マクロ。
//   （Google Test 等の外部ライブラリを使わないという制約のため自作）
//
//   使い方：
//     TEST_CASE("テスト名") { CHECK(条件); CHECK_NEAR(a, b, 許容誤差); }
//     int main() { return gnn::test::runAllTests(); }
// ============================================================================

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace gnn::test {

struct TestEntry {
    std::string name;
    std::function<void()> body;
};

// 登録されたテストの一覧（inline 関数内の static 変数にして、ヘッダーだけで定義できるようにする）
inline std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> entries;
    return entries;
}

inline int& failureCount() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> body) {
        registry().push_back({name, std::move(body)});
    }
};

inline void reportFailure(const char* file, int line, const std::string& message) {
    ++failureCount();
    std::cout << "    [FAIL] " << file << ":" << line << "  " << message << "\n";
}

// 全テストを実行し、失敗が1件でもあれば 1 を返す
inline int runAllTests() {
    int failedTests = 0;
    for (const TestEntry& t : registry()) {
        const int before = failureCount();
        std::cout << "[RUN ] " << t.name << "\n";
        try {
            t.body();
        } catch (const std::exception& e) {
            reportFailure(__FILE__, __LINE__, std::string("例外: ") + e.what());
        }
        const bool ok = failureCount() == before;
        if (!ok) {
            ++failedTests;
        }
        std::cout << (ok ? "[ OK ] " : "[NG  ] ") << t.name << "\n";
    }
    std::cout << "\n" << registry().size() - failedTests << " / " << registry().size() << " テスト成功\n";
    return failedTests == 0 ? 0 : 1;
}

}  // namespace gnn::test

#define GNN_TEST_CONCAT_(a, b) a##b
#define GNN_TEST_CONCAT(a, b) GNN_TEST_CONCAT_(a, b)

// テスト名は日本語の文字列で指定する。関数名は行番号から自動生成する
#define GNN_TEST_CASE_IMPL(description, fn)                                     \
    static void fn();                                                           \
    static const ::gnn::test::Registrar GNN_TEST_CONCAT(fn, _registrar)(description, fn); \
    static void fn()

#define TEST_CASE(description) GNN_TEST_CASE_IMPL(description, GNN_TEST_CONCAT(gnnTest_, __LINE__))

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) ::gnn::test::reportFailure(__FILE__, __LINE__, #cond); \
    } while (0)

#define CHECK_EQ(a, b)                                                                     \
    do {                                                                                   \
        const auto gnn_a_ = (a);                                                           \
        const auto gnn_b_ = (b);                                                           \
        if (!(gnn_a_ == gnn_b_))                                                           \
            ::gnn::test::reportFailure(__FILE__, __LINE__,                                 \
                                       std::string(#a " == " #b "  (") + std::to_string(gnn_a_) + \
                                           " vs " + std::to_string(gnn_b_) + ")");        \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                              \
    do {                                                                                   \
        const double gnn_a_ = (a);                                                         \
        const double gnn_b_ = (b);                                                         \
        if (!(std::abs(gnn_a_ - gnn_b_) <= (eps)))                                         \
            ::gnn::test::reportFailure(__FILE__, __LINE__,                                 \
                                       std::string(#a " ≈ " #b "  (") + std::to_string(gnn_a_) + \
                                           " vs " + std::to_string(gnn_b_) + ")");        \
    } while (0)
