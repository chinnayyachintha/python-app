import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, jsonify, request, send_file, make_response
import requests
import json
import time
import threading
import os
import finnhub
import pandas as pd

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
# BEST PRACTICE: Get your API key from environment variables.
# Using the key provided in the original code for demonstration:
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "d1nlrb1r01qovv8k2q6gd1nlrb1r01qovv8k2q70")

# Initialize Finnhub client for the calculator functionality
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# --- Dashboard Configuration (Used for the /api/stocks endpoint) ---
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "IBM", "META"]
FETCH_INTERVAL_SECONDS = 30
latest_stock_data = {}
data_lock = threading.Lock()

# --- Stock Data Fetching Function (Dashboard) ---
def fetch_and_update_stock_data(symbol, api_key):
    """Fetches latest quote and company profile data for the dashboard."""
    quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    company_profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={api_key}"

    try:
        quote_response = requests.get(quote_url)
        quote_response.raise_for_status()
        quote_data = quote_response.json()

        profile_response = requests.get(company_profile_url)
        profile_response.raise_for_status()
        profile_data = profile_response.json()

        if not quote_data or quote_data.get('c') is None:
            return

        company_name = profile_data.get('name', symbol)
        
        current_price = quote_data.get('c')
        high_price = quote_data.get('h')
        low_price = quote_data.get('l')
        open_price = quote_data.get('o')
        prev_close_price = quote_data.get('pc')

        change = current_price - prev_close_price
        percentage_change = (change / prev_close_price * 100) if prev_close_price else 0

        with data_lock:
            latest_stock_data[symbol] = {
                "symbol": symbol,
                "company_name": company_name,
                "logo": profile_data.get('logo', ''),
                "current_price": f"{current_price:.2f}",
                "high_price": f"{high_price:.2f}",
                "low_price": f"{low_price:.2f}",
                "open_price": f"{open_price:.2f}",
                "prev_close_price": f"{prev_close_price:.2f}",
                "change": f"{change:.2f}",
                "percentage_change": f"{percentage_change:.2f}",
                "timestamp": int(time.time() * 1000)
            }
            print(f"Updated dashboard data for {symbol} ({company_name})")

    except Exception as e:
        print(f"Error fetching dashboard data for {symbol}: {e}")

# --- Background Thread for Periodic Data Updates (Dashboard) ---
def background_data_updater():
    """Periodically fetches stock data for the dashboard."""
    while True:
        start_time = time.time()
        for symbol in STOCK_SYMBOLS:
            fetch_and_update_stock_data(symbol, FINNHUB_API_KEY)
            time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        time_to_sleep = FETCH_INTERVAL_SECONDS - elapsed_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

# Start the background data update thread when the Flask app starts
updater_thread = threading.Thread(target=background_data_updater, daemon=True)
updater_thread.start()

# --- Plot Generation Function ---
def generate_plot(results, investment_years, drip_enabled):
    fig, ax = plt.subplots(figsize=(10, 6))
    years = list(range(investment_years + 1))
    
    for result in results:
        if not result['error']:
            if drip_enabled:
                ax.plot(years, result['portfolio_values_with_drip'], label=f"{result['ticker']} (With DRIP)")
            else:
                ax.plot(years, result['portfolio_values_no_drip'], label=f"{result['ticker']} (No DRIP)")

    ax.set_title('Portfolio Value Over Time')
    ax.set_xlabel('Years')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return plot_url

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the main HTML page (GET) and handles the dividend calculation (POST).
    """
    results = []
    plot_url = None
    comparison_table = None
    
    # Initialize variables for template context (used for persisting form inputs)
    initial_investment = request.form.get('initial_investment')
    investment_years = request.form.get('investment_years')
    drip_enabled = request.form.get('drip_enabled')
    tickers_input = request.form.get('tickers')

    # This block handles the "Calculate" button click (POST request)
    if request.method == 'POST':
        # Validate inputs
        if not initial_investment or not investment_years:
            return render_template('index.html', results=[{'error': 'Please provide investment amount and years.'}])

        try:
            initial_investment_f = float(initial_investment)
            investment_years_i = int(investment_years)
            drip_enabled_b = request.form.get('drip_enabled') == 'yes'
            if initial_investment_f <= 0 or investment_years_i <= 0:
                return render_template('index.html', results=[{'error': 'Investment amount and years must be positive.'}])
        except (TypeError, ValueError) as e:
            return render_template('index.html', results=[{'error': f"Invalid input for investment amount or years: {e}"}])

        # Initialize all_ticker_data
        all_ticker_data = {}
        
        # --- Calculation Logic ---
        tickers = request.form.get('tickers', '').split(',')
        
        for ticker_symbol in tickers:
            ticker_symbol = ticker_symbol.strip().upper()
            if not ticker_symbol:
                continue

            try:
                quote_data = finnhub_client.quote(ticker_symbol)
                current_price = quote_data.get('c')

                if current_price is None or current_price == 0:
                    raise ValueError(f"Could not get valid current price for {ticker_symbol}.")

                basic_financials = finnhub_client.company_basic_financials(ticker_symbol, 'all')
                financial_metrics = basic_financials.get('metric', {})
                dividend_yield = financial_metrics.get('dividendYieldTTM', 0)
                annual_dividend_per_share = financial_metrics.get('dividendPerShareAnnual', 0)
                
                if annual_dividend_per_share == 0 and dividend_yield > 0 and current_price > 0:
                    annual_dividend_per_share = (dividend_yield / 100) * current_price

                if annual_dividend_per_share == 0:
                    results.append({
                        'ticker': ticker_symbol,
                        'current_price': f"${current_price:,.2f}",
                        'error': f"{ticker_symbol} does not pay dividends, DRIP has no effect."
                    })
                    continue

                annual_dividend_per_share = annual_dividend_per_share if annual_dividend_per_share is not None else 0
                dividend_yield = dividend_yield if dividend_yield is not None else 0
                payout_ratio = financial_metrics.get('payoutRatioTTM', 0) if financial_metrics.get('payoutRatioTTM') is not None else 0

                initial_shares = initial_investment_f / current_price

                shares_no_drip = initial_shares
                shares_with_drip = initial_shares
                current_value_no_drip = initial_investment_f
                current_value_with_drip = initial_investment_f
                
                portfolio_values_no_drip = [initial_investment_f]
                portfolio_values_with_drip = [initial_investment_f]

                for year in range(1, investment_years_i + 1):
                    dividends_no_drip = shares_no_drip * annual_dividend_per_share
                    current_value_no_drip += dividends_no_drip
                    portfolio_values_no_drip.append(current_value_no_drip)

                    dividends_with_drip = shares_with_drip * annual_dividend_per_share
                    if current_price > 0:
                        new_shares = dividends_with_drip / current_price
                        shares_with_drip += new_shares
                    current_value_with_drip = shares_with_drip * current_price
                    portfolio_values_with_drip.append(current_value_with_drip)

                final_value_no_drip = portfolio_values_no_drip[-1]
                final_value_with_drip = portfolio_values_with_drip[-1]

                results.append({
                    'ticker': ticker_symbol,
                    'current_price': f"${current_price:,.2f}",
                    'annual_dividend_per_share': f"${annual_dividend_per_share:,.2f}",
                    'dividend_yield': f"{dividend_yield:,.2f}%",
                    'payout_ratio': f"{payout_ratio:,.2f}%",
                    'initial_shares': f"{initial_shares:,.2f}",
                    'final_value_no_drip': f"${final_value_no_drip:,.2f}",
                    'final_value_with_drip': f"${final_value_with_drip:,.2f}",
                    'portfolio_values_no_drip': portfolio_values_no_drip,
                    'portfolio_values_with_drip': portfolio_values_with_drip,
                    'error': None
                })
                all_ticker_data[ticker_symbol] = {
                    'portfolio_values_no_drip': portfolio_values_no_drip,
                    'portfolio_values_with_drip': portfolio_values_with_drip,
                    'final_value_no_drip': final_value_no_drip,
                    'final_value_with_drip': final_value_with_drip,
                    'annual_yield': dividend_yield
                }

            except Exception as e:
                results.append({
                    'ticker': ticker_symbol,
                    'error': f"Could not retrieve data for {ticker_symbol} from Finnhub. Error: {e}."
                })
            
            time.sleep(0.5) # Delay for API limits

        # Generate Comparison Table
        if len(all_ticker_data) > 1:
            comparison_data = {'Metric': ['Starting Investment']}
            comparison_data['Metric'].append(f'{investment_years_i}-Year Value ({ "With DRIP" if drip_enabled_b else "No DRIP"})')

            for ticker_symbol, data in all_ticker_data.items():
                comparison_data[ticker_symbol] = [f"${initial_investment_f:,.2f}"]
                if drip_enabled_b:
                    comparison_data[ticker_symbol].append(f"${data['final_value_with_drip']:,.2f}")
                else:
                    comparison_data[ticker_symbol].append(f"${data['final_value_no_drip']:,.2f}")

            comparison_table = pd.DataFrame(comparison_data).to_html(classes='table table-striped table-bordered', index=False)

        # Generate Chart
        if results and any(not r['error'] for r in results):
            plot_url = generate_plot(results, investment_years_i, drip_enabled_b)

    # Render the template, passing the results and input values
    return render_template('index.html', 
                           results=results, 
                           plot_url=plot_url, 
                           comparison_table=comparison_table,
                           initial_investment=initial_investment,
                           investment_years=investment_years,
                           drip_enabled=drip_enabled,
                           tickers_input=tickers_input)

@app.route('/api/stocks')
def api_stocks():
    """Provides the latest stock data for the dashboard as JSON."""
    with data_lock:
        return jsonify(latest_stock_data)
        
@app.route('/export_csv', methods=['POST'])
def export_csv():
    """Handles the CSV export logic based on calculation inputs."""
    
    try:
        tickers = request.form.get('export_tickers').split(',')
        initial_investment = float(request.form.get('export_initial_investment'))
        investment_years = int(request.form.get('export_investment_years'))
        drip_enabled = request.form.get('export_drip_enabled').lower() == 'true' 
        if initial_investment <= 0 or investment_years <= 0:
            return "Investment amount and years must be positive.", 400
    except (TypeError, ValueError) as e:
        return f"Invalid export parameters: {e}", 400

    all_data_for_export = []

    for ticker_symbol in tickers:
        ticker_symbol = ticker_symbol.strip().upper()
        if not ticker_symbol:
            continue
            
        try:
            quote_data = finnhub_client.quote(ticker_symbol)
            current_price = quote_data.get('c')
            
            basic_financials = finnhub_client.company_basic_financials(ticker_symbol, 'all')
            financial_metrics = basic_financials.get('metric', {})
            annual_dividend_per_share = financial_metrics.get('dividendPerShareAnnual', 0)

            if annual_dividend_per_share == 0 and financial_metrics.get('dividendYieldTTM', 0) > 0 and current_price > 0:
                annual_dividend_per_share = (financial_metrics.get('dividendYieldTTM', 0) / 100) * current_price

            if current_price is None or current_price == 0:
                raise ValueError(f"Could not get valid current price for {ticker_symbol}.")

            initial_shares = initial_investment / current_price
            shares = initial_shares
            current_value = initial_investment
            
            yearly_breakdown = []

            # Year 0 data
            yearly_breakdown.append({
                'Year': 0,
                'Ticker': ticker_symbol,
                'Starting Shares': f"{initial_shares:,.2f}",
                'Starting Value': f"${initial_investment:,.2f}",
                'Dividends Received': "$0.00",
                'Ending Shares': f"{initial_shares:,.2f}",
                'Ending Value': f"${initial_investment:,.2f}"
            })

            # Calculation for subsequent years (1 to investment_years)
            for year in range(1, investment_years + 1):
                starting_shares_this_year = shares
                starting_value_this_year = current_value
                
                dividends = starting_shares_this_year * annual_dividend_per_share

                if drip_enabled and current_price > 0:
                    new_shares = dividends / current_price
                    shares += new_shares
                    current_value = shares * current_price 
                else:
                    current_value += dividends 

                yearly_breakdown.append({
                    'Year': year,
                    'Ticker': ticker_symbol,
                    'Starting Shares': f"{starting_shares_this_year:,.2f}",
                    'Starting Value': f"${starting_value_this_year:,.2f}",
                    'Dividends Received': f"${dividends:,.2f}",
                    'Ending Shares': f"{shares:,.2f}",
                    'Ending Value': f"${current_value:,.2f}"
                })
            all_data_for_export.extend(yearly_breakdown)
            
        except Exception as e:
            all_data_for_export.append({'Ticker': ticker_symbol, 'Error': f"Could not export data for {ticker_symbol}: {e}"})
            
        time.sleep(0.5)

    df = pd.DataFrame(all_data_for_export)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = make_response(csv_buffer.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=dividend_calculator_results.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response







if __name__ == '__main__':
    # We set use_reloader=False because we are running a background thread, 
    # which can cause issues with Flask's auto-reloader.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)