import sympy as sp

# 定义符号
x0, x1 = sp.symbols("x0 x1")

# 原始表达式
expr = (
    -0.01836136169731617 
    + 0.0010000000474974513 * (
        -1.0809999704360962 
        - 0.026000000536441803 * x0 
        - 0.0010000000474974513 * x1
        ) 
    - 0.0020000000949949026 * (
        -0.37200000882148743 
        + 0.85199999809265137 * x0 
        - 0.0020000000949949026 * x1
        ) 
    - 0.0010000000474974513 * (
        -0.22800000011920929 
        + 1.7130000591278076 * x0 
        + 0.0010000000474974513 * x1
        ) 
    - 0.0040000001899898052 * (
        1.9620000123977661 
        + 2.8020000457763672 * x0 
        + 0.0020000000949949026 * x1
        )
    )

# 展开并合并同类项
simplified_expr = sp.expand(expr)

# 提取系数
const = simplified_expr.subs({x0: 0, x1: 0})
coef_x0 = simplified_expr.coeff(x0)
coef_x1 = simplified_expr.coeff(x1)

print("Simplified expression:")
print(simplified_expr)

print("\nCoefficient form:")
print(f"{const} + ({coef_x0}) * x0 + ({coef_x1}) * x1")