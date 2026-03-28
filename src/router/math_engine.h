#pragma once

#include <string>
#include <functional>

class MathEngine {
public:
    MathEngine();
    ~MathEngine();

    bool can_handle(const std::string& query);
    std::string solve(const std::string& expression);

private:
    // Simple shunting-yard parser
    std::string evaluate(const std::string& expr);
    double apply_operator(double a, double b, char op);
    int precedence(char op);
};

