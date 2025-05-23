import logging

def find_triangular_arbitrage(
    depths, 
    fee=0.001, 
    threshold=0.0005, 
    dry_run=True, 
    execute_order=None
):
    """
    Search for triangular arbitrage opportunities and optionally execute them.
    depths: dict of {pair: {"bid": float, "ask": float}, ...}
    fee: trading fee per trade (e.g., 0.001 for 0.1%)
    threshold: minimum profitable rate after fees (e.g., 0.0005 = 0.05%)
    dry_run: If True, only simulates order execution.
    execute_order: callable to actually execute the trade sequence (optional for future real API use)
    Returns: list of profitable arbitrage opportunities found.
    """
    arb_ops = []

    # Example: USDT -> BTC -> ETH -> USDT
    try:
        # Step 1: USDT -> BTC (buy BTC with USDT at ask)
        rate1 = 1 / depths["BTCUSDT"]["ask"] * (1 - fee)
        # Step 2: BTC -> ETH (buy ETH with BTC at ask, or sell BTC for ETH at bid)
        rate2 = rate1 * depths["ETHBTC"]["bid"] * (1 - fee)
        # Step 3: ETH -> USDT (sell ETH for USDT at bid)
        rate3 = rate2 * depths["ETHUSDT"]["bid"] * (1 - fee)

        profit = rate3 - 1
        if profit > threshold:
            cycle = ("USDT", "BTC", "ETH", "USDT")
            arb_ops.append({"cycle": cycle, "final_amount": rate3, "profit": profit})
            logging.info(f"Triangular arbitrage opportunity found: {cycle} | Profit: {profit:.6f}")

            if not dry_run and execute_order:
                execute_order(cycle, rate3, profit)
            else:
                logging.info(f"DRY RUN: Would execute {cycle} for profit {profit:.6f}")
    except Exception as e:
        logging.warning(f"Error in triangular arbitrage calculation: {e}")

    return arb_ops