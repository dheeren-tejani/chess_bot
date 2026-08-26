import chess

def perft(b, d):
    if d == 0:
        return 1
    n = 0
    for m in b.legal_moves:
        b.push(m)
        n += perft(b, d - 1)
        b.pop()
    return n

b = chess.Board('r4rk1/1pp1qppp/p1np1n2/2b1p1b1/2B1P1B1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10')
print('d1', perft(b, 1))
print('d2', perft(b, 2))
print('d3', perft(b, 3))
