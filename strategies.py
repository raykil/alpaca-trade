def momentum(BARS, window=3, threshold=100):
    move = 'hold'
    avgPrices = BARS['avgPrice'].to_numpy()
    if len(avgPrices)>=window:
        delta_p = avgPrices[-1] - avgPrices[-window]
        if   (delta_p> threshold): move = 'buy'
        elif (delta_p<-threshold): move = 'sell'
    return move

def reverse_momentum(BARS, window=3, threshold=30):
    move = 'hold'
    avgPrices = BARS['avgPrice'].to_numpy()
    if len(avgPrices)>=window:
        delta_p = avgPrices[-1] - avgPrices[-window]
        if   (delta_p> threshold): move = 'sell'
        elif (delta_p<-threshold): move = 'buy'
    return move


strategy_map = {
    'momentum': momentum,
    'reverse_momentum': reverse_momentum
}