import pytest
from unittest.mock import MagicMock, patch
from trading.trader_new import execute_trade

@pytest.mark.asyncio
async def test_execute_trade_buy():
    # Mock Gemini exchange response for a successful buy
    mock_response = {"status": "success", "trade_id": 12345, "amount": 0.01, "price": 100.0}

    with patch('trading.trader.exchange.create_order', return_value=mock_response) as mock_order:
        # Simulate the buy trade
        response = await execute_trade('BTC/USD', 1, 0.01)

        # Assertions
        assert response == mock_response
        mock_order.assert_called_once_with('BTC/USD', 'market', 'buy', 0.01)

@pytest.mark.asyncio
async def test_execute_trade_sell():
    # Mock Gemini exchange response for a successful sell
    mock_response = {"status": "success", "trade_id": 67890, "amount": 0.01, "price": 110.0}

    with patch('trading.trader.exchange.create_order', return_value=mock_response) as mock_order:
        # Simulate the sell trade
        response = await execute_trade('BTC/USD', -1, 0.01)

        # Assertions
        assert response == mock_response
        mock_order.assert_called_once_with('BTC/USD', 'market', 'sell', 0.01)
