def find_triangular_arbitrage(depths):
    """
    depths: dict of {"BTCUSDT": {"bid":..., "ask":...}, ...}
    Returns list of arbitrage ops if found.
    """
    arb_ops = []
    # USDT -> BTC -> ETH -> USDT
    try:
        rate1 = 1 / depths["BTCUSDT"]["ask"]
        rate2 = rate1 * depths["ETHBTC"]["bid"]
        rate3 = rate2 * depths["ETHUSDT"]["bid"]
        if rate3 > 1.0005:
            arb_ops.append(("USDT","BTC","ETH","USDT", rate3))
    except Exception as e:
        pass
    return arb_ops