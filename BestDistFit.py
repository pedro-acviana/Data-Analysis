import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit

def determine_best_distribution(data, col):
    """Analyzes a column and suggests the best distribution through statistical testing"""
    # Convert to numpy array and remove NaNs
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    # Special handling for SkinThickness (remove zeros)
    if col == 'SkinThickness' or col == 'Insulin':
        data = data[data > 0]
    
    # Calculate basic statistics
    n_zeros = np.sum(data == 0)
    pct_zeros = n_zeros / len(data) * 100 if len(data) > 0 else 0
    is_continuous = len(np.unique(data)) > 20 if len(data) > 0 else False
    
    # Initialize results
    result = {
        'column': col,
        'suggested_distribution': 'Unknown',
        'reason': 'No valid data points' if len(data) == 0 else '',
        'parameters': None,
        'pct_zeros': pct_zeros
    }
    
    if len(data) == 0:
        return result
    
    # Check for binary/categorical data
    unique_vals = np.unique(data)
    if len(unique_vals) == 2:
        result['suggested_distribution'] = 'Bernoulli'
        result['reason'] = 'Binary variable'
        return result
    
    # Check for count data (integers with limited unique values)
    is_count_data = all(np.mod(data[data > 0], 1) == 0) and not is_continuous
    
    # Test continuous distributions
    continuous_dists = {
        'Normal': stats.norm,
        'Log-Normal': stats.lognorm,
        'Exponential': stats.expon,
        'Gamma': stats.gamma,
        'Weibull': stats.weibull_min
    }
    
    # Test count distributions
    count_dists = {
        'Poisson': stats.poisson,
        'Negative Binomial': stats.nbinom
    }
    
    best_fit = None
    best_sse = np.inf
    dist_results = {}
    
    # Select appropriate distributions to test
    test_dists = continuous_dists if not is_count_data else count_dists
    
    for name, dist in test_dists.items():
        try:
            # Fit distribution
            params = dist.fit(data)
            
            # Generate theoretical distribution
            x = np.linspace(np.min(data), np.max(data), 100)

            if name in ['Poisson', 'Negative Binomial']:
                x_vals = np.arange(0, int(np.max(data)) + 1)
                if name == 'Poisson':
                    pmf = dist.pmf(x_vals, *params)
                else:
                    pmf = dist.pmf(x_vals, *params)
                # Compare with empirical PMF
                hist = np.histogram(data, bins=np.arange(0, int(np.max(data)) + 2), density=True)[0]
                sse = np.sum((hist[:len(pmf)] - pmf) ** 2)
            else:
                pdf = dist.pdf(x, *params)
                hist, _ = np.histogram(data, bins=30, density=True)
                hist_x = (_[:-1] + _[1:]) / 2
                pdf_hist = np.interp(hist_x, x, pdf)
                sse = np.sum((hist - pdf_hist) ** 2)
            
            dist_results[name] = sse
            if sse < best_sse:
                best_fit = name
                best_sse = sse
                result['parameters'] = params
                
        except Exception as e:
            dist_results[name] = str(e)
            continue
    
    # Determine best fit
    if best_fit:
        result['suggested_distribution'] = best_fit
        result['reason'] = f'Best fit (SSE: {best_sse:.2f}) among tested distributions'
        
        # Handle zero-inflation
        if pct_zeros > 20:
            if best_fit in ['Gamma', 'Weibull', 'Log-Normal']:
                result['suggested_distribution'] = f'Zero-Inflated {best_fit}'
                result['reason'] += f' with {pct_zeros:.1f}% zeros'
            elif best_fit in ['Poisson', 'Negative Binomial']:
                result['suggested_distribution'] = f'Zero-Inflated {best_fit}'
                result['reason'] += f' with {pct_zeros:.1f}% zeros'
    
    # Fallback to KDE if no good parametric fit
    if result['suggested_distribution'] == 'Unknown' or best_sse > 0.5:
        result['suggested_distribution'] = 'Kernel Density'
        result['reason'] = 'No good parametric fit found'
    
    return result

def plot_distribution(ax, data, dist_info):
    """Plots data histogram with fitted distribution and value-based limits"""
    data = np.array(data)
    data = data[~np.isnan(data)]

    ax.hist(data, bins=30, density=True, color='skyblue',
            edgecolor='black', alpha=0.7)

    if len(data) == 0:
        ax.set_title(f'{dist_info["column"]}\nNo valid data')
        return

    x = np.linspace(np.min(data), np.max(data), 200)

    if dist_info['parameters'] is not None:
        dist_name = dist_info['suggested_distribution']
        params = dist_info['parameters']
        
        lower, upper = None, None
        
        if dist_name == 'Normal':
            pdf = stats.norm.pdf(x, *params)
            ax.plot(x, pdf, 'r-', lw=2, label='Normal fit')
            lower, upper = stats.norm.ppf(0.025, *params), stats.norm.ppf(0.975, *params)
        
        elif dist_name == 'Gamma':
            pdf = stats.gamma.pdf(x, *params)
            ax.plot(x, pdf, 'r-', lw=2, label='Gamma fit')
            lower, upper = stats.gamma.ppf(0.025, *params), stats.gamma.ppf(0.975, *params)
        
        elif dist_name == 'Exponential':
            pdf = stats.expon.pdf(x, *params)
            ax.plot(x, pdf, 'r-', lw=2, label='Exponential fit')
            lower, upper = stats.expon.ppf(0.025, *params), stats.expon.ppf(0.975, *params)
        
        elif dist_name == 'Log-Normal':
            pdf = stats.lognorm.pdf(x, *params)
            ax.plot(x, pdf, 'r-', lw=2, label='Log-Normal fit')
            lower, upper = stats.lognorm.ppf(0.025, *params), stats.lognorm.ppf(0.975, *params)
        
        elif dist_name == 'Weibull':
            pdf = stats.weibull_min.pdf(x, *params)
            ax.plot(x, pdf, 'r-', lw=2, label='Weibull fit')
            lower, upper = stats.weibull_min.ppf(0.025, *params), stats.weibull_min.ppf(0.975, *params)
        
        elif dist_name == 'Poisson':
            x_vals = np.arange(0, int(np.max(data)) + 1)
            pmf = stats.poisson.pmf(x_vals, *params)
            ax.plot(x_vals, pmf, 'ro', ms=5, label='Poisson fit')
            lower, upper = stats.poisson.ppf(0.025, *params), stats.poisson.ppf(0.975, *params)
        
        elif dist_name.startswith('Zero-Inflated'):
            if 'Gamma' in dist_name:
                pdf = stats.gamma.pdf(x, *params)
                lower, upper = stats.gamma.ppf(0.025, *params), stats.gamma.ppf(0.975, *params)
            elif 'Weibull' in dist_name:
                pdf = stats.weibull_min.pdf(x, *params)
                lower, upper = stats.weibull_min.ppf(0.025, *params), stats.weibull_min.ppf(0.975, *params)
            elif 'Log-Normal' in dist_name:
                pdf = stats.lognorm.pdf(x, *params)
                lower, upper = stats.lognorm.ppf(0.025, *params), stats.lognorm.ppf(0.975, *params)

            ax.plot(x, pdf, 'r-', lw=2, label=dist_name)
            ax.axvline(0, color='purple', linestyle='--', alpha=0.5, 
                       label=f'Zeros ({dist_info["pct_zeros"]:.1f}%)')

        # Plot confidence bounds
        if lower is not None and upper is not None:
            ax.axvline(lower, color='green', linestyle='--', label=f'Lower: {lower:.2f}')
            ax.axvline(upper, color='orange', linestyle='--', label=f'Upper: {upper:.2f}')
    
    elif dist_info['suggested_distribution'] == 'Kernel Density':
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(data.reshape(-1, 1))
        log_pdf = kde.score_samples(x.reshape(-1, 1))
        ax.plot(x, np.exp(log_pdf), 'r-', lw=2, label='KDE fit')
        lower, upper = np.percentile(data, 2.5), np.percentile(data, 97.5)
        ax.axvline(lower, color='green', linestyle='--', label=f'Lower: {lower:.2f}')
        ax.axvline(upper, color='orange', linestyle='--', label=f'Upper: {upper:.2f}')
    
    ax.set_title(f'{dist_info["column"]}\n{dist_info["suggested_distribution"]}')
    ax.legend()



def analyze_data(df):
    """Main analysis function"""
    plt.figure(figsize=(18, 25))
    results = []
    
    for i, col in enumerate(df.columns, 1):
        ax = plt.subplot(5, 2, i)
        dist_info = determine_best_distribution(df[col], col)
        
        if col == 'Outcome':
            ax.hist(df[col], bins=[-0.5, 0.5, 1.5], rwidth=0.8, density=True)
            ax.set_xticks([0, 1])
            ax.set_title(f'{col}\nBernoulli')
        else:
            plot_distribution(ax, df[col], dist_info)
        
        results.append(dist_info)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n=== DISTRIBUTION ANALYSIS RESULTS ===")
    for res in results:
        print(f"\n{res['column']}: {res['suggested_distribution']}")
        print(f"Reason: {res['reason']}")
        if res['parameters'] is not None:
            print(f"Parameters: {res['parameters']}")
    
    return results

# Load and analyze data
csv_file_path = "Datasets/diabetes.csv"
df = pd.read_csv(csv_file_path)
analysis_results = analyze_data(df)