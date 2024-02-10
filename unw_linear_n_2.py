def calc_bounds(l1, l2, t, alpha, beta):
    T_lower = (t - l2) / (beta - alpha)
    T_upper = 1 - ((t - l1) /  alpha)
    L_break = (alpha - (l2 - l1)) / beta
    #if (T_lower <= 1 and T_lower >= 0 and T_upper >= 0 and T_upper <= 1 and T_upper >= T_lower and L_break > T_upper):
    if (T_upper < T_lower):
        if (L_break > T_upper or L_break < T_lower):
            print(l1, " ", l2, " ", t, " ", alpha, " ", beta)
            print('T lower bound: ', T_lower)
            print('T upper bound: ', T_upper)
            print('L1 breakpoint: ', L_break)
            print()


largest = 30

for l1 in range(largest):
    for l2 in range(l1, largest):
        for t in range(l2, largest):
            for alpha in range(1, largest):
                for beta in range(alpha + 1, largest):
                    calc_bounds(l1, l2, t, alpha, beta)

calc_bounds(0,1,30,50,200)
