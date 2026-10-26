import numpy as np
import pandas as pd

# =========================
# 1. B(x) = c + a*x0 + b*x1
# =========================
c = -0.0263183620491222
a = -0.0146510014536042
b = -6.00000056996943e-6

# =========================
# 2. grad B
#    因为 B 是线性的，所以 grad B 是常向量
# =========================
grad_B = np.array([a, b])

# =========================
# 3. HighOrd2 dynamics
#    f(x) = [x1, -2400*x1 - 576*x0]
# =========================
def vector_field(x0, x1):
    f0 = x1
    f1 = -2400 * x1 - 576 * x0
    return np.array([f0, f1])

# =========================
# 4. Lie derivative
#    L_f B = grad(B)^T f
# =========================
def lie_derivative(x0, x1):
    f = vector_field(x0, x1)
    return np.dot(grad_B, f)

# =========================
# 5. B=0 line
#    c + a*x0 + b*x1 = 0
#    x0 = -(c + b*x1) / a
# =========================
def x0_on_B0(x1):
    return -(c + b * x1) / a

# =========================
# 6. sample B=0 line inside domain
# =========================
domain_low = -2.2
domain_high = 2.2

x1_min = domain_low
x1_max = domain_high

p_start = np.array([x0_on_B0(x1_min), x1_min])
p_end = np.array([x0_on_B0(x1_max), x1_max])

print("B=0 line segment inside domain:")
print("start:", p_start)
print("end:  ", p_end)

n_samples = 201
ts = np.linspace(0, 1, n_samples)
points = np.array([(1 - t) * p_start + t * p_end for t in ts])

# =========================
# 7. compute and print B, grad B, f, LfB
# =========================
rows = []

for i, (x0, x1) in enumerate(points):
    B_value = c + a * x0 + b * x1

    f = vector_field(x0, x1)
    LfB = np.dot(grad_B, f)

    rows.append({
        "idx": i,
        "x0": x0,
        "x1": x1,
        "B(x)": B_value,

        "f_x0": f[0],
        "f_x1": f[1],

        "LfB": LfB
    })

df = pd.DataFrame(rows)

pd.set_option("display.precision", 6)
pd.set_option("display.max_columns", 10)

print(df)
df.to_csv("lie.csv")
