def nb_year(p0, percent, aug, p):
   y = 0
   while p0 < p:
        p0 = p0 + p0 * (percent / 100) + aug
        y = y + 1
   return y
