"""
Модуль 5: Бэктестинг VaR/ES — тест Купика и тест на независимость
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class BacktestVaR:
    """
    Бэктестинг VaR моделей
    """
    
    def __init__(self, returns, var_predictions, confidence=0.95):
        """
        Параметры:
        returns: фактический ряд доходностей
        var_predictions: прогнозы VaR (отрицательные числа)
        confidence: доверительный уровень
        """
        self.returns = np.array(returns)
        self.var = np.array(var_predictions)
        self.confidence = confidence
        self.n = len(returns)
        
        # Индикатор превышения VaR (1 если убыток > VaR)
        self.exceedances = (self.returns < self.var).astype(int)
        self.n_exceed = np.sum(self.exceedances)
        self.expected_exceed = self.n * (1 - confidence)
        
    def kupiec_test(self):
        """
        Тест Купика (Unconditional Coverage Test)
        Проверяет, соответствует ли доля превышений ожидаемой
        
        H0: доля превышений = 1 - confidence
        Статистика LR ~ χ²(1)
        """
        p_hat = self.n_exceed / self.n
        p_expected = 1 - self.confidence
        
        if p_hat == 0 or p_hat == 1:
            lr_stat = 1e6
        else:
            # LR = 2 * [ln(L(p_hat)) - ln(L(p_expected))]
            lr_stat = 2 * (self.n_exceed * np.log(p_hat / p_expected) + 
                          (self.n - self.n_exceed) * np.log((1 - p_hat) / (1 - p_expected)))
        
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'statistic': lr_stat,
            'p_value': p_value,
            'expected_exceed': self.expected_exceed,
            'actual_exceed': self.n_exceed,
            'exceedance_ratio': p_hat,
            'reject_h0': p_value < 0.05
        }
    
    def christoffersen_independence_test(self):
        """
        Тест Кристофферсена на независимость превышений
        Проверяет, не кластеризуются ли превышения
        
        H0: превышения независимы
        """
        # Матрица переходов
        n00 = np.sum((self.exceedances[:-1] == 0) & (self.exceedances[1:] == 0))
        n01 = np.sum((self.exceedances[:-1] == 0) & (self.exceedances[1:] == 1))
        n10 = np.sum((self.exceedances[:-1] == 1) & (self.exceedances[1:] == 0))
        n11 = np.sum((self.exceedances[:-1] == 1) & (self.exceedances[1:] == 1))
        
        # Вероятности переходов
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
        
        # LR статистика
        if pi == 0 or pi == 1:
            lr_stat = 0
        else:
            l1 = (n00 + n01) * np.log(1 - pi) if n00 + n01 > 0 else 0
            l2 = (n10 + n11) * np.log(1 - pi) if n10 + n11 > 0 else 0
            l3 = (n01 + n11) * np.log(pi) if n01 + n11 > 0 else 0
            
            l_hat = (n00 * np.log(1 - pi0) if n00 > 0 else 0) + \
                    (n01 * np.log(pi0) if n01 > 0 else 0) + \
                    (n10 * np.log(1 - pi1) if n10 > 0 else 0) + \
                    (n11 * np.log(pi1) if n11 > 0 else 0)
            
            l_0 = l1 + l2 + l3
            
            lr_stat = 2 * (l_hat - l_0) if l_hat > l_0 else 0
        
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'statistic': lr_stat,
            'p_value': p_value,
            'pi0': pi0,
            'pi1': pi1,
            'reject_h0': p_value < 0.05
        }
    
    def loss_function(self):
        """
        Функция потерь (Fengler & Okhrin)
        Штрафует модель за неправильные предсказания
        """
        # VaR exceedance loss
        exceed_loss = np.mean((self.returns[self.exceedances == 1] - 
                               self.var[self.exceedances == 1])**2)
        
        # Miss loss (когда превышения нет, но VaR близко)
        diff = self.returns - self.var
        miss_loss = np.mean(diff[diff > 0]**2)  # когда доходность выше VaR
        
        return exceed_loss + miss_loss

def compare_backtests(returns, var_normal, var_historical, var_evt, confidence=0.99):
    """
    Сравнение трех моделей VaR на бэктесте
    """
    results = {}
    
    models = {
        'Normal': var_normal,
        'Historical': var_historical,
        'EVT': var_evt
    }
    
    for name, var_pred in models.items():
        bt = BacktestVaR(returns, var_pred, confidence)
        results[name] = {
            'kupiec': bt.kupiec_test(),
            'independence': bt.christoffersen_independence_test(),
            'loss': bt.loss_function(),
            'exceedance_ratio': bt.n_exceed / len(returns)
        }
    
    return results

if __name__ == "__main__":
    from data_generation import generate_regime_switching_returns
    from extreme_value_theory import ExtremeValueTheory
    from scipy import stats as sp_stats
    
    np.random.seed(42)
    returns, regime = generate_regime_switching_returns(n_days=2000)
    
    print("=" * 60)
    print("Бэктестинг VaR моделей (тест Купика)")
    print("=" * 60)
    
    confidence = 0.99
    test_size = 500
    train_size = len(returns) - test_size
    
    train_returns = returns[:train_size]
    test_returns = returns[train_size:]
    
    # 1. Нормальный VaR (скользящий)
    mu_train = np.mean(train_returns)
    sigma_train = np.std(train_returns)
    var_normal = mu_train + sigma_train * sp_stats.norm.ppf(1 - confidence)
    var_normal_test = np.full(test_size, var_normal)
    
    # 2. Исторический VaR (скользящее окно)
    window = 250
    var_historical_test = np.zeros(test_size)
    for i in range(test_size):
        window_returns = returns[train_size + i - window:train_size + i]
        var_historical_test[i] = np.percentile(window_returns, (1 - confidence) * 100)
    
    # 3. EVT VaR
    evt = ExtremeValueTheory(train_returns)
    evt.fit_gpd_pot()
    var_evt_test = np.full(test_size, evt.tail_var(confidence))
    
    print(f"\nТестовый период: {test_size} дней")
    print(f"Доверительный уровень: {confidence*100:.1f}%")
    print(f"Ожидаемое количество превышений: {test_size * (1-confidence):.1f}")
    
    # Сравнение
    results = compare_backtests(test_returns, var_normal_test, var_historical_test, var_evt_test, confidence)
    
    for model_name, res in results.items():
        print(f"\n--- {model_name} модель ---")
        print(f"Фактическое превышение: {res['exceedance_ratio']:.2%}")
        print(f"Тест Купика: LR={res['kupiec']['statistic']:.2f}, p-value={res['kupiec']['p_value']:.4f}")
        print(f"  → Отклоняем H0? {res['kupiec']['reject_h0']}")
        print(f"Тест на независимость: p-value={res['independence']['p_value']:.4f}")
        print(f"Функция потерь: {res['loss']:.6f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Превышения VaR
    time = np.arange(test_size)
    axes[0, 0].plot(time, test_returns, alpha=0.7, label='Доходности', linewidth=0.7)
    axes[0, 0].plot(time, var_normal_test, 'r--', label='Normal VaR', linewidth=1)
    axes[0, 0].plot(time, var_historical_test, 'g--', label='Historical VaR', linewidth=1)
    axes[0, 0].plot(time, var_evt_test, 'b--', label='EVT VaR', linewidth=1)
    
    # Отметим превышения
    exceed_normal = test_returns < var_normal_test
    exceed_hist = test_returns < var_historical_test
    exceed_evt = test_returns < var_evt_test
    
    axes[0, 0].scatter(time[exceed_normal], test_returns[exceed_normal], c='red', s=10, alpha=0.7)
    axes[0, 0].set_title('VaR прогнозы и фактические превышения')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True)
    
    # Кумулятивное количество превышений
    axes[0, 1].plot(np.cumsum(exceed_normal), label='Normal', alpha=0.7)
    axes[0, 1].plot(np.cumsum(exceed_hist), label='Historical', alpha=0.7)
    axes[0, 1].plot(np.cumsum(exceed_evt), label='EVT', alpha=0.7)
    axes[0, 1].axhline(y=test_size * (1-confidence), color='black', linestyle='--', 
                       label=f'Ожидаемое ({test_size * (1-confidence):.0f})')
    axes[0, 1].set_title('Кумулятивное количество превышений VaR')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Сравнение p-value
    models_names = list(results.keys())
    p_values = [results[m]['kupiec']['p_value'] for m in models_names]
    colors = ['red' if p < 0.05 else 'green' for p in p_values]
    axes[1, 0].bar(models_names, p_values, color=colors)
    axes[1, 0].axhline(y=0.05, color='black', linestyle='--', label='α=0.05')
    axes[1, 0].set_ylabel('p-value')
    axes[1, 0].set_title('Тест Купика: p-value (красный = reject H0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Функция потерь
    losses = [results[m]['loss'] for m in models_names]
    axes[1, 1].bar(models_names, losses, color='steelblue')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Сравнение функций потерь (чем ниже, тем лучше)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualization/results_backtesting.png', dpi=150)
    plt.show()
    
    print("\nГрафик сохранен как 'results_backtesting.png'")