from sympy import symbols, sin, cos, sqrt, pi, Eq, factor, solveset, S


# DH parameters
a2, a3, a4, d3, d4 = symbols('a2 a3 a4 d3 d4', real=True)

# Joint angles
theta1, theta2, theta3, theta4, theta5, theta6 = symbols(
    'theta1 theta2 theta3 theta4 theta5 theta6', real=True
)


# First bracket f1
f1 = (
    sin(theta1)*cos(theta4 + pi/4)
    + sin(theta1)*cos(theta1)*cos(theta2 + theta3)*cos(theta4 + pi/4)
    + sin(theta4 + pi/4)
)

# Second bracket f2
f2 = (
    -2*a2*a3*sin(theta3)*cos(theta1)*cos(theta2)
    +  a2*a3*sin(theta3)*cos(theta2)
    -2*a2*d4*cos(theta1)*cos(theta2)*cos(theta3)
    +  a2*d4*cos(theta2)*cos(theta3)
    -  a3*sin(theta3)*cos(theta2 + theta3)
    +  a3*d4*sin(theta3)*sin(theta2 + theta3)
    -  a3*d4*cos(theta3)*cos(theta2 + theta3)
    +  d4*sin(theta2 + theta3)*cos(theta3)
)

# Full expression
expr = sqrt(2)*a2 * f1 * f2 * sin(theta5) * cos(theta4)


eq = Eq(expr, 0)

print("\n--- Original Expression ---")
print(expr)

factored = factor(expr)
print("\n--- Factored Expression ---")
print(factored)


print("\n--- Zero Conditions ---")
print("1) a2 = 0")
print("2) sin(theta5) = 0  → theta5 = k*pi")
print("3) cos(theta4) = 0  → theta4 = pi/2 + k*pi")
print("4) f1(theta1,theta2,theta3,theta4) = 0")
print("5) f2(a2,a3,d4,theta1,theta2,theta3) = 0")



# Example: Solve for a3 (returns relationships, not numbers)
sol_a3 = solveset(Eq(expr, 0), a3, domain=S.Complexes)
print("\n--- Solve for a3 ---")
print(sol_a3)

# Example: Solve for theta4 from f1 = 0
sol_theta4_from_f1 = solveset(Eq(f1, 0), theta4, domain=S.Complexes)
print("\n--- Solve for theta4 from f1=0 ---")
print(sol_theta4_from_f1)
