import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter, DayLocator
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_backtest_results(portfolio_value: pd.Series, signals: pd.DataFrame, trades: pd.DataFrame, crypto: str):
    """Plot comprehensive backtest results including equity curve, cumulative returns, drawdown, price with signals, and daily returns."""
    try:
        logger.info(f"Plotting backtest results - Signals columns: {signals.columns.tolist()}")
        if 'close' not in signals.columns:
            logger.error("Error: 'close' not in signals columns")
            raise KeyError("'close' missing from signals for plotting")

        signals.index = pd.to_datetime(signals.index, utc=True).tz_localize(None)
        portfolio_value.index = pd.to_datetime(portfolio_value.index, utc=True).tz_localize(None)
        if not trades.empty:
            trades.index = pd.to_datetime(trades.index, utc=True).tz_localize(None)

        common_index = signals.index
        signals = signals.reindex(common_index).ffill().fillna({'close': 78877.88, 'signal': 0, 'position_size': 0})
        portfolio_value = portfolio_value.reindex(common_index).ffill().fillna(10000)
        if not trades.empty:
            trades = trades.reindex(common_index).ffill().fillna(0)

        signals['close'] = signals['close'].clip(lower=10000, upper=200000)

        # Set extended date range
        start_date = pd.to_datetime('2020-01-01 00:00:00', utc=True).tz_localize(None)
        end_date = pd.to_datetime('2025-03-12 23:00:00', utc=True).tz_localize(None)
        if (end_date - start_date).total_seconds() < 3600:
            logger.warning(f"Date range too short: {start_date} to {end_date}. Using extended range.")
        signals = signals.reindex(pd.date_range(start=start_date, end=end_date, freq='h')).ffill().fillna({'close': 78877.88, 'signal': 0, 'position_size': 0})
        portfolio_value = portfolio_value.reindex(signals.index).ffill().fillna(10000)
        if not trades.empty:
            trades = trades.reindex(signals.index).ffill().fillna(0)

        signals = signals.loc[start_date:end_date].dropna(how='all')
        portfolio_value = portfolio_value.loc[start_date:end_date].dropna(how='all')
        if not trades.empty:
            trades = trades.loc[start_date:end_date].dropna(how='all')

        if signals.empty or portfolio_value.empty:
            logger.error("Empty signals or portfolio_value DataFrame after filtering")
            raise ValueError("No data to plot")

        # Calculate MAPE correctly
        if 'predicted_price' in signals.columns:
            valid_mask = (~signals['close'].isna()) & (~signals['predicted_price'].isna())
            mape = np.mean(np.abs((signals.loc[valid_mask, 'close'] - signals.loc[valid_mask, 'predicted_price']) / signals.loc[valid_mask, 'close'])) * 100
            logger.info(f"Mean Absolute Percentage Error (MAPE) of predictions: {mape:.2f}%")
        else:
            logger.warning("Cannot calculate MAPE: 'predicted_price' not in signals")

        # Downsample data for plotting to avoid MAXTICKS exceeded
        numeric_columns = signals.select_dtypes(include=[np.number]).columns
        signals_ds = signals[numeric_columns].resample('4h').mean().dropna()
        portfolio_value_ds = portfolio_value.resample('4h').mean().dropna()

        plt.figure(figsize=(15, 25))

        # 1. Equity Curve
        ax1 = plt.subplot(5, 1, 1)
        ax1.plot(portfolio_value_ds.index, portfolio_value_ds.values, label='Equity Curve', color='blue')
        plt.title(f'{crypto} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.grid()
        plt.legend()

        # 2. Cumulative Returns
        ax2 = plt.subplot(5, 1, 2)
        initial_value = portfolio_value_ds.iloc[0] if pd.notna(portfolio_value_ds.iloc[0]) else 10000
        cumulative_returns = ((portfolio_value_ds - initial_value) / initial_value) * 100
        total_return = cumulative_returns.iloc[-1] if not pd.isna(cumulative_returns.iloc[-1]) else 0.0
        ax2.plot(cumulative_returns.index, cumulative_returns.values, label='Cumulative Returns', color='green')
        plt.title(f'Cumulative Returns (Total: {total_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid()
        plt.legend()

        # 3. Max Drawdown
        ax3 = plt.subplot(5, 1, 3)
        rolling_max = portfolio_value_ds.cummax()
        drawdown = (portfolio_value_ds - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min() if not pd.isna(drawdown.min()) else 0.0
        ax3.plot(drawdown.index, drawdown.values, label='Drawdown', color='red')
        plt.title(f'Max Drawdown: {max_drawdown:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid()
        plt.legend()

        # 4. Price with Buy/Sell Signals and Trades
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(signals_ds.index, signals_ds['close'].values, label='Price', alpha=0.5, color='blue')
        buy_signals = signals[signals['signal'] == 1].dropna(subset=['close', 'signal'])
        sell_signals = signals[signals['signal'] == -1].dropna(subset=['close', 'signal'])
        trades_profit = trades['profit'].dropna()
        top_trades = trades_profit.nlargest(5).index.union(trades_profit.nsmallest(5).index)
        trade_entries = trades.loc[top_trades].dropna(subset=['entry_price'])
        trade_exits = trades.loc[top_trades].dropna(subset=['exit_price'])
        ax4.scatter(buy_signals.index, signals.loc[buy_signals.index, 'close'], marker='^', color='green', label='Buy Signal', zorder=5, alpha=0.5)
        ax4.scatter(sell_signals.index, signals.loc[sell_signals.index, 'close'], marker='v', color='red', label='Sell Signal', zorder=5, alpha=0.5)
        ax4.scatter(trade_entries.index, trade_entries['entry_price'], marker='o', color='cyan', label='Trade Entry', zorder=5, alpha=0.5)
        ax4.scatter(trade_exits.index, trade_exits['exit_price'], marker='o', color='magenta', label='Trade Exit', zorder=5, alpha=0.5)

        # Add new signal markers
        if 'luxalgo_signal' in signals.columns:
            luxalgo_buys = signals[signals['luxalgo_signal'] == 1].dropna(subset=['close', 'luxalgo_signal'])
            luxalgo_sells = signals[signals['luxalgo_signal'] == -1].dropna(subset=['close', 'luxalgo_signal'])
            ax4.scatter(luxalgo_buys.index, signals.loc[luxalgo_buys.index, 'close'], marker='s', color='lime', label='LuxAlgo Buy', zorder=5, alpha=0.3)
            ax4.scatter(luxalgo_sells.index, signals.loc[luxalgo_sells.index, 'close'], marker='s', color='pink', label='LuxAlgo Sell', zorder=5, alpha=0.3)

        if 'trendspider_signal' in signals.columns:
            trendspider_buys = signals[signals['trendspider_signal'] == 1].dropna(subset=['close', 'trendspider_signal'])
            trendspider_sells = signals[signals['trendspider_signal'] == -1].dropna(subset=['close', 'trendspider_signal'])
            ax4.scatter(trendspider_buys.index, signals.loc[trendspider_buys.index, 'close'], marker='D', color='cyan', label='TrendSpider Buy', zorder=5, alpha=0.3)
            ax4.scatter(trendspider_sells.index, signals.loc[trendspider_sells.index, 'close'], marker='D', color='orange', label='TrendSpider Sell', zorder=5, alpha=0.3)

        if 'smrt_signal' in signals.columns:
            smrt_buys = signals[signals['smrt_signal'] == 1].dropna(subset=['close', 'smrt_signal'])
            smrt_sells = signals[signals['smrt_signal'] == -1].dropna(subset=['close', 'smrt_signal'])
            ax4.scatter(smrt_buys.index, signals.loc[smrt_buys.index, 'close'], marker='*', color='purple', label='SMRT Buy', zorder=5, alpha=0.3)
            ax4.scatter(smrt_sells.index, signals.loc[smrt_sells.index, 'close'], marker='*', color='brown', label='SMRT Sell', zorder=5, alpha=0.3)

        if 'arbitrage_signal' in signals.columns:
            arbitrage_buys = signals[signals['arbitrage_signal'] == 1].dropna(subset=['close', 'arbitrage_signal'])
            arbitrage_sells = signals[signals['arbitrage_signal'] == -1].dropna(subset=['close', 'arbitrage_signal'])
            ax4.scatter(arbitrage_buys.index, signals.loc[arbitrage_buys.index, 'close'], marker='P', color='teal', label='Arbitrage Buy', zorder=5, alpha=0.3)
            ax4.scatter(arbitrage_sells.index, signals.loc[arbitrage_sells.index, 'close'], marker='P', color='gold', label='Arbitrage Sell', zorder=5, alpha=0.3)

        for idx in top_trades:
            if pd.notna(trades.loc[idx, 'entry_price']) and pd.notna(trades.loc[idx, 'exit_price']):
                entry_price = trades.loc[idx, 'entry_price']
                exit_price = trades.loc[idx, 'exit_price']
                profit = (exit_price - entry_price) * signals.loc[idx, 'position_size'] if signals.loc[idx, 'signal'] == 1 else (entry_price - exit_price) * abs(signals.loc[idx, 'position_size'])
                color = 'green' if profit > 0 else 'red'
                ax4.text(idx, exit_price, f'P/L: {profit:.2f}', color=color, fontsize=8, rotation=45)

        plt.title(f'{crypto} Price with Signals and Trades')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()

        # 5. Cumulative P/L
        ax5 = plt.subplot(5, 1, 5)
        initial_value = portfolio_value_ds.iloc[0]
        cumulative_pl = (portfolio_value_ds - initial_value).cumsum()
        ax5.plot(cumulative_pl.index, cumulative_pl.values, label='Cumulative P/L', color='orange')
        plt.title('Cumulative Profit/Loss')
        plt.xlabel('Date')
        plt.ylabel('Profit/Loss (USD)')
        plt.grid()
        plt.legend()

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.xaxis_date()
            ax.xaxis.set_major_locator(DayLocator(interval=30))
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_minor_locator(DayLocator(interval=7))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.autoscale_view()

        plt.tight_layout()
        sanitized_crypto = crypto.replace('/', '-')
        filename = f'{sanitized_crypto}_backtest_results.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Backtest results plotted and saved as '{filename}'")

    except Exception as e:
        logger.error(f"Error plotting backtest results: {e}")
        raise