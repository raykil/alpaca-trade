from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from strategies import liveTrade

def makeMove(BARS, strategy, **kwargs):
    return liveTrade(strategy, BARS, **kwargs)

def appendMove(BARS, move):
    BARS.loc[BARS.index[-1], 'move'] = move
    print(BARS.iloc[-1].to_dict())

def placeOrder(config, symbol, OrderInfo):
    move       = OrderInfo['move']
    quantity   = OrderInfo['qty']
    limitPrice = OrderInfo['limit_price']

    if   move == 'sell': orderside = OrderSide.SELL
    elif move == 'buy':  orderside = OrderSide.BUY

    client    = TradingClient(config['api-key'], config['secret-key'], paper=True)
    account   = client.get_account()
    positions = client.get_all_positions()[0]

    order = LimitOrderRequest(
        symbol=symbol,
        limit_price=limitPrice,
        qty=quantity,
        side=orderside,
        time_in_force=TimeInForce.GTC
    )
    can_sell = float(positions.qty_available) > quantity and move == 'sell'
    can_buy  = move == 'buy' and float(account.buying_power) >= quantity * limitPrice
    if can_sell or can_buy:
        submitted = client.submit_order(order_data=order)
        ts = submitted.submitted_at.astimezone(ZoneInfo("America/New_York")).replace(microsecond=0, tzinfo=None)
        print(f"Order submitted at {ts}! (ID: {submitted.id})")
        return str(submitted.id)
    return None

def trackOrder(config, BARS):
    client = TradingClient(config['api-key'], config['secret-key'], paper=True)
    orders = client.get_orders()
    print(f"nOrders: {len(orders)} ({orders[0].symbol if orders else 'none'})")
    fills = {str(o.id): float(o.filled_qty) for o in orders}
    order_rows = BARS[BARS['order_id'].notna()]
    for ts, row in order_rows.iterrows():
        if row['order_id'] in fills:
            BARS.loc[ts, 'order_filled_qty'] = fills[row['order_id']]
    for order in orders:
        print(
            f"time: {order.submitted_at.astimezone(ZoneInfo('America/New_York')).replace(microsecond=0, tzinfo=None)}  "
            f"id:{str(order.id).split('-')[0]}  "
            f"side:{'BUY' if str(order.side) == 'OrderSide.BUY' else 'SELL'}  "
            f"qty: {order.qty}  "
            f"type: {order.type.split('.')[-1]}  "
            f"status:{order.status}  "
            f"filled_qty: {order.filled_qty:<14}  "
            f"lim_price: {round(float(order.limit_price), 5):<14}  "
            f"filled_avgPrice: {order.filled_avg_price}"
        )

def trackAsset(config):
    client    = TradingClient(config['api-key'], config['secret-key'], paper=True)
    account   = client.get_account()
    positions = client.get_all_positions()[0]
    asset_str = f"cash:{account.cash}    buying_power:{account.buying_power}    qty:{positions.qty}    qty_available:{positions.qty_available}"
    return asset_str
