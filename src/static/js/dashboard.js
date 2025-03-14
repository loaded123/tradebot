// src/static/js/dashboard.js
const ctx = document.getElementById('pnlChart').getContext('2d');
const pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: JSON.parse(document.getElementById('pnl-timestamps').dataset.value || '[]'),
        datasets: [{
            label: 'P&L (USD)',
            data: JSON.parse(document.getElementById('pnl-values').dataset.value || '[]'),
            backgroundColor: 'rgba(0, 119, 204, 0.3)',
            borderColor: 'rgba(0, 119, 204, 1)',
            borderWidth: 1,
            fill: true
        }]
    },
    options: {
        scales: {
            x: { title: { display: true, text: 'Time' } },
            y: { beginAtZero: true, title: { display: true, text: 'Profit/Loss (USD)' } }
        },
        plugins: { legend: { position: 'top' } }
    }
});

function updateDashboard() {
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            if (data) {
                document.getElementById('pnl').textContent = `${(data.pnl || 0).toFixed(2)} USD`;
                document.getElementById('num_trades').textContent = data.num_trades || 0;
                document.getElementById('win_rate').textContent = `${((data.win_rate || 0) * 100).toFixed(2)}%`;
                document.getElementById('sharpe_ratio').textContent = (data.sharpe_ratio || 0).toFixed(2);
                document.getElementById('sortino_ratio').textContent = (data.sortino_ratio || 0).toFixed(2);
                document.getElementById('max_drawdown').textContent = `${(data.max_drawdown || 0).toFixed(2)}%`;
                document.getElementById('last_update').textContent = data.last_update || 'N/A';
                pnlChart.data.labels = data.pnl_timestamps || [];
                pnlChart.data.datasets[0].data = data.pnl_values || [];
                pnlChart.update();
            }
        })
        .catch(error => console.error('Error fetching performance:', error));

    fetch('/api/trades')
        .then(response => response.json())
        .then(data => {
            const tradesTable = document.getElementById('tradesTable');
            tradesTable.innerHTML = '';
            if (data && data.trades && data.trades.length) {
                data.trades.forEach(trade => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="p-2 border-b">${trade.timestamp || 'N/A'}</td>
                        <td class="p-2 border-b">${trade.symbol || 'N/A'}</td>
                        <td class="p-2 border-b">${trade.trade_type || 'N/A'}</td>
                        <td class="p-2 border-b">${(trade.size || 0).toFixed(4)}</td>
                        <td class="p-2 border-b">${(trade.entry_price || 0).toFixed(2)}</td>
                        <td class="p-2 border-b">${trade.exit_price ? trade.exit_price.toFixed(2) : 'N/A'}</td>
                        <td class="p-2 border-b ${trade.profit !== undefined && trade.profit >= 0 ? 'text-green-600' : 'text-red-600'}">${trade.profit !== undefined ? trade.profit.toFixed(2) : 'N/A'}</td>
                    `;
                    tradesTable.appendChild(row);
                });
            } else {
                tradesTable.innerHTML = '<tr><td colspan="7" class="p-2 border-b text-center">No trades available</td></tr>';
            }
        })
        .catch(error => console.error('Error fetching trades:', error));

    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            if (data && data.logs) {
                document.getElementById('logDisplay').textContent = data.logs.join('\n');
            }
        })
        .catch(error => console.error('Error fetching logs:', error));
}

updateDashboard();
setInterval(updateDashboard, 5000);