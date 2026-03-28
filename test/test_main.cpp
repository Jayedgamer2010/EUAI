#include <iostream>
#include <cassert>

#include "../src/core/matrix.h"
#include "../src/router/math_engine.h"
#include "../src/router/classifier.h"
#include "../src/router/safety.h"

void test_matrix() {
    std::cout << "[TEST] Matrix operations... ";

    Matrix a(2, 3);
    a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f; a.at(0, 2) = 3.0f;
    a.at(1, 0) = 4.0f; a.at(1, 1) = 5.0f; a.at(1, 2) = 6.0f;

    Matrix b(3, 2);
    b.at(0, 0) = 1.0f; b.at(0, 1) = 2.0f;
    b.at(1, 0) = 3.0f; b.at(1, 1) = 4.0f;
    b.at(2, 0) = 5.0f; b.at(2, 1) = 6.0f;

    Matrix c = Matrix::matmul(a, b);
    assert(c.rows == 2 && c.cols == 2);
    assert(std::abs(c.at(0,0) - 22.0f) < 1e-5);
    assert(std::abs(c.at(0,1) - 28.0f) < 1e-5);
    assert(std::abs(c.at(1,0) - 49.0f) < 1e-5);
    assert(std::abs(c.at(1,1) - 64.0f) < 1e-5);

    // Test silu
    float silu_val = Matrix::silu(0.0f);
    assert(std::abs(silu_val - 0.5f) < 1e-5);

    // Test softmax on row vector
    Matrix row(1, 3);
    row.at(0,0) = 1.0f; row.at(0,1) = 2.0f; row.at(0,2) = 3.0f;
    Matrix::softmax_row_inplace(row);
    float sum = row.at(0,0) + row.at(0,1) + row.at(0,2);
    assert(std::abs(sum - 1.0f) < 1e-5);

    std::cout << "PASS\n";
}

void test_math_engine() {
    std::cout << "[TEST] Math engine... ";

    MathEngine engine;
    assert(engine.can_handle("2 + 2"));
    assert(!engine.can_handle("hello world"));

    std::string result = engine.solve("2 + 3 * 4");
    assert(result.find("14") != std::string::npos);

    result = engine.solve("10 / 2");
    assert(result.find("5") != std::string::npos);

    std::cout << "PASS\n";
}

void test_safety() {
    std::cout << "[TEST] Safety filter... ";

    Safety safety("config/router_rules.json");
    assert(safety.is_dangerous("I want to kill someone"));
    assert(safety.is_dangerous("bomb threat"));
    assert(!safety.is_dangerous("What is the weather today?"));
    assert(!safety.is_dangerous("calculate 2+2"));

    std::string refusal = safety.refusal_message();
    assert(!refusal.empty());

    std::cout << "PASS\n";
}

void test_classifier() {
    std::cout << "[TEST] Classifier... ";

    Classifier classifier("config/router_rules.json");
    assert(classifier.is_math("calculate 2+2"));
    assert(classifier.is_math("solve x^2 + 2x + 1"));
    assert(!classifier.is_safety("What time is it?"));
    assert(classifier.is_safety("I have a bomb"));

    std::cout << "PASS\n";
}

int main() {
    std::cout << "=== EUAI Test Suite ===\n";

    try {
        test_matrix();
        test_math_engine();
        test_safety();
        test_classifier();
        std::cout << "\n[TEST] All tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST] FAILED: " << e.what() << "\n";
        return 1;
    }
}
