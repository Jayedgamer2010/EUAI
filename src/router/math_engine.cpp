#include "math_engine.h"
#include <iostream>
#include <sstream>
#include <stack>
#include <cctype>
#include <cmath>

MathEngine::MathEngine() = default;
MathEngine::~MathEngine() = default;

bool MathEngine::can_handle(const std::string& query) {
    // Check if query looks like a math expression
    // Simple heuristic: contains digits and operators
    bool has_digit = false;
    bool has_operator = false;
    for (char c : query) {
        if (std::isdigit(c)) has_digit = true;
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') has_operator = true;
    }
    return has_digit && has_operator;
}

std::string MathEngine::solve(const std::string& expression) {
    try {
        std::string result_str = evaluate(expression);
        return "Result: " + result_str;
    } catch (const std::exception& e) {
        return "Error: " + std::string(e.what());
    }
}

double MathEngine::apply_operator(double a, double b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) throw std::runtime_error("Division by zero");
            return a / b;
        case '^': return std::pow(a, b);
        default: throw std::runtime_error("Unknown operator");
    }
}

int MathEngine::precedence(char op) {
    switch (op) {
        case '^': return 3;
        case '*': case '/': return 2;
        case '+': case '-': return 1;
        default: return 0;
    }
}

std::string MathEngine::evaluate(const std::string& expr) {
    std::stack<double> values;
    std::stack<char> ops;

    for (size_t i = 0; i < expr.length(); i++) {
        if (expr[i] == ' ') continue;

        if (std::isdigit(expr[i]) || expr[i] == '.') {
            // Parse number
            std::string num_str;
            while (i < expr.length() && (std::isdigit(expr[i]) || expr[i] == '.')) {
                num_str += expr[i++];
            }
            i--;
            values.push(std::stod(num_str));
        } else if (expr[i] == '(') {
            ops.push(expr[i]);
        } else if (expr[i] == ')') {
            while (!ops.empty() && ops.top() != '(') {
                char op = ops.top(); ops.pop();
                double b = values.top(); values.pop();
                double a = values.top(); values.pop();
                values.push(apply_operator(a, b, op));
            }
            ops.pop(); // Remove '('
        } else if (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/' || expr[i] == '^') {
            // Operator
            while (!ops.empty() && precedence(ops.top()) >= precedence(expr[i])) {
                char op = ops.top(); ops.pop();
                double b = values.top(); values.pop();
                double a = values.top(); values.pop();
                values.push(apply_operator(a, b, op));
            }
            ops.push(expr[i]);
        } else {
            // Unknown character (e.g., letters), skip
            continue;
        }
    }

    while (!ops.empty()) {
        char op = ops.top(); ops.pop();
        double b = values.top(); values.pop();
        double a = values.top(); values.pop();
        values.push(apply_operator(a, b, op));
    }

    if (values.empty()) return "0";
    return std::to_string(values.top());
}
