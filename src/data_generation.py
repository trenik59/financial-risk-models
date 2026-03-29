"""
Модуль 1: Генерация данных с толстыми хвостами и переключением режимов
Реализация своими руками: t-распределение, Markov switching волатильности
"""

import numpy as np
import matplotlib.pyplot as plt

def t_distribution_manual(nu, size):
    """
    Генерация t-распределения через отношение:
    t = Z / sqrt(Chi2 / nu)
    где Z ~ N(0,1), Chi2 ~ χ²(nu)
    
    Параметры:
    nu   — степени свободы (чем меньше, тем толще хвосты)
    size — количество样本
    
    Возвращает:
    массив из t-распределения
    """
    Z = np.random.normal(0, 1, size)
    # Хи-квадрат с nu степенями свободы = сумма квадратов nu нормальных величин
    Chi2 = np.sum(np.random.normal(0, 1, (nu, size))**2, axis=0)
    t = Z / np.sqrt(Chi2 / nu)
    return t

def generate_regime_switching_returns(n_days=2000, mu=0.0005, 
                                        sigma_low=0.01, sigma_high=0.04,
                                        p_stay_low=0.98, p_stay_high=0.94,
                                        nu=4, use_t_dist=True):
    """
    Генерация доходностей с переключением режимов волатильности
    
    Режим 0: спокойный рынок (низкая волатильность)
    Режим 1: кризисный рынок (высокая волатильность + толстые хвосты)
    
    Параметры:
    use_t_dist — если True, в кризисном режиме используем t-распределение
    """
    returns = np.zeros(n_days)
    regime = np.zeros(n_days, dtype=int)
    
    current_regime = 0
    
    for i in range(n_days):
        if current_regime == 0:
            # Спокойный режим: нормальное распределение
            if use_t_dist:
                # Но все равно добавим легкие хвосты
                r = np.random.normal(mu, sigma_low)
            else:
                r = np.random.normal(mu, sigma_low)
            
            if np.random.random() > p_stay_low:
                current_regime = 1
        else:
            # Кризисный режим: толстые хвосты (t-распределение)
            if use_t_dist:
                # t-распределение с nu=4 дает очень толстые хвосты
                scaled_t = t_distribution_manual(nu, 1)[0] * sigma_high + mu
                r = scaled_t
            else:
                r = np.random.normal(mu, sigma_high)
            
            if np.random.random() > p_stay_high:
                current_regime = 0
        
        returns[i] = r
        regime[i] = current_regime
    
    return returns, regime

def analyze_regimes(returns, regime):
    """Анализирует статистику по режимам"""
    regime0_returns = returns[regime == 0]
    regime1_returns = returns[regime == 1]
    
    stats = {
        'regime0_mean': np.mean(regime0_returns),
        'regime0_std': np.std(regime0_returns),
        'regime0_kurtosis': np.mean((regime0_returns - np.mean(regime0_returns))**4) / (np.std(regime0_returns)**4),
        'regime1_mean': np.mean(regime1_returns),
        'regime1_std': np.std(regime1_returns),
        'regime1_kurtosis': np.mean((regime1_returns - np.mean(regime1_returns))**4) / (np.std(regime1_returns)**4),
        'regime0_fraction': len(regime0_returns) / len(returns),
        'regime1_fraction': len(regime1_returns) / len(returns)
    }
    return stats

if __name__ == "__main__":
    # Генерация данных
    np.random.seed(42)
    returns, regime = generate_regime_switching_returns(n_days=3000)
    
    # Анализ
    stats = analyze_regimes(returns, regime)
    
    print("=" * 60)
    print("Анализ сгенерированных данных")
    print("=" * 60)
    print(f"Режим 0 (спокойный): доля = {stats['regime0_fraction']:.1%}")
    print(f"  Средняя доходность: {stats['regime0_mean']:.5f}")
    print(f"  Волатильность: {stats['regime0_std']:.5f}")
    print(f"  Эксцесс: {stats['regime0_kurtosis']:.2f} (3 = нормальное)")
    print()
    print(f"Режим 1 (кризисный): доля = {stats['regime1_fraction']:.1%}")
    print(f"  Средняя доходность: {stats['regime1_mean']:.5f}")
    print(f"  Волатильность: {stats['regime1_std']:.5f}")
    print(f"  Эксцесс: {stats['regime1_kurtosis']:.2f} ( >3 = толстые хвосты)")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Временной ряд
    axes[0, 0].plot(returns, alpha=0.7, linewidth=0.5)
    crisis_mask = regime == 1
    axes[0, 0].fill_between(range(len(returns)), -0.2, 0.2, 
                             where=crisis_mask, alpha=0.3, color='red')
    axes[0, 0].set_title('Доходности с кризисными периодами (красный)')
    axes[0, 0].set_ylabel('Доходность')
    
    # Гистограммы по режимам
    axes[0, 1].hist(returns[regime == 0], bins=50, alpha=0.5, label='Спокойный', density=True)
    axes[0, 1].hist(returns[regime == 1], bins=50, alpha=0.5, label='Кризисный', density=True)
    axes[0, 1].set_title('Распределение доходностей по режимам')
    axes[0, 1].legend()
    
    # QQ-plot для проверки хвостов
    from scipy import stats as sp_stats
    sp_stats.probplot(returns[regime == 1], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('QQ-plot кризисного режима (отклонение от нормали)')
    
    # ACF доходностей
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(returns, lags=40, ax=axes[1, 1])
    axes[1, 1].set_title('Автокорреляция доходностей')
    
    plt.tight_layout()
    plt.savefig('visualization/results_data_generation.png', dpi=150)
    plt.show()
    
    print("\nГрафик сохранен как 'results_data_generation.png'")