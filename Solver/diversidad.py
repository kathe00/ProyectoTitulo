def porcentajesXPLXPT(div, maxDiv):
    XPL = round((div/maxDiv)*100,2)
    XPT = round((abs(div-maxDiv)/maxDiv)*100,2)
    state = -1
    #Determinar estado
    if XPL >= XPT:
        state = 1 # Exploración
    else:
        state = 0 # Explotación
    return XPL, XPT, state

def diversidadHussain(matriz):
    medianas = []
    for j in range(matriz[0].__len__()):
        suma = 0
        for i in range(matriz.__len__()):
            suma += matriz[i][j]
        medianas.append(suma/matriz.__len__())
    n = len(matriz)
    l = len(matriz[0])
    diversidad = 0
    for d in range(l):
        div_d = 0
        for i in range(n):
            div_d = div_d + abs(medianas[d] - matriz[i][d])
        diversidad = diversidad + div_d
    return (1 / (l*n)) * diversidad
