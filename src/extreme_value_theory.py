"""
Модуль 3: Extreme Value Theory (EVT) — моделирование хвостов распределения
Реализация своими руками: метод превышений порога (POT), GPD, оценка параметров
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GPD:
    """
    Generalized Pareto Distribution (GPD) для моделирования хвоста
    CDF: F(x) = 1 - (1 + ξ * x / β)^(-1/ξ) для ξ ≠ 0
         F(x) = 1 - exp(-x/β) для ξ = 0
    """
    
    @staticmethod
    def pdf(x, xi, beta):
        """Плотность GPD"""
        if xi == 0:
            return (1/beta) * np.exp(-x/beta)
        else:
            return (1/beta) * (1 + xi * x / beta) ** (-1/xi - 1)
    
    @staticmethod
    def cdf(x, xi, beta):
        """Функция распределения GPD"""
        if xi == 0:
            return 1 - np.exp(-x/beta)
        else:
            return 1 - (1 + xi * x / beta) ** (-1/xi)
    
    @staticmethod
    def quantile(p, xi, beta):
        """Квантиль GPD"""
        if xi == 0:
            return -beta * np.log(1 - p)
        else:
            return beta * ((1 - p)**(-xi) - 1) / xi

class ExtremeValueTheory:
    """
    EVT с методом превышений порога (POT)
    """
    
    def __init__(self, returns):
        self.returns = np.array(returns)
        self.exceedances = None
        self.threshold = None
        self.xi = None      # параметр формы (хвоста)
        self.beta = None    # параметр масштаба
        self.nu = None      # количество превышений
        
    def select_threshold(self, quantile_range=(0.85, 0.99), n_points=30):
        """
        Выбор порога с помощью Mean Excess Plot
        Для GPD среднее превышение линейно: e(u) = (β + ξ*u) / (1-ξ)
        """
        sorted_returns = np.sort(self.returns)
        thresholds = np.percentile(self.returns, np.linspace(quantile_range[0]*100, 
                                                             quantile_range[1]*100, n_points))
        
        mean_excess = []
        for u in thresholds:
            exceedances = self.returns[self.returns < u]  # Левый хвост (убытки)
            if len(exceedances) > 10:
                mean_excess.append(np.mean(u - exceedances))
            else:
                mean_excess.append(np.nan)
        
        mean_excess = np.array(mean_excess)
        valid = ~np.isnan(mean_excess)
        
        # Выбираем порог, после которого график становится линейным
        # Упрощенно: берем 90-й процентиль
        self.threshold = np.percentile(self.returns, 90)
        
        return self.threshold, thresholds[valid], mean_excess[valid]
    
    def fit_gpd_pot(self, threshold=None):
        """
        Оценка параметров GPD для превышений над порогом
        Используем метод максимального правдоподобия
        """
        if threshold is None:
            self.threshold = np.percentile(self.returns, 90)
        else:
            self.threshold = threshold
        
        # Превышения (отрицательные убытки, которые больше порога по модулю)
        exceedances = self.returns[self.returns < self.threshold]
        self.exceedances = - (exceedances - self.threshold)  # Переводим в положительные
        self.exceedances = self.exceedances[self.exceedances > 0]
        self.nu = len(self.exceedances)
        
        if self.nu < 10:
            print(f"Предупреждение: мало превышений ({self.nu})")
            self.xi, self.beta = 0.2, np.std(self.exceedances)
            return self.xi, self.beta
        
        def neg_log_likelihood_gpd(params):
            """Отрицательное логарифмическое правдоподобие для GPD"""
            xi, beta = params
            if beta <= 0:
                return 1e10
            if xi <= -0.5:  # Ограничение для существования дисперсии
                return 1e10
            
            y = self.exceedances / beta
            if xi == 0:
                ll = -np.sum(np.log(beta) + y)
            else:
                # Проверка: 1 + xi*y > 0
                if np.any(1 + xi * y <= 0):
                    return 1e10
                ll = -np.sum(np.log(beta) + (1/xi + 1) * np.log(1 + xi * y))
            
            return -ll
        
        # Начальные приближения
        xi_init = 0.2
        beta_init = np.std(self.exceedances) / 2
        
        result = minimize(neg_log_likelihood_gpd, [xi_init, beta_init], 
                         method='L-BFGS-B', 
                         bounds=[(-0.5, 1), (1e-6, None)])
        
        if result.success:
            self.xi, self.beta = result.x
        else:
            print(f"Оптимизация не сошлась: {result.message}")
            self.xi, self.beta = xi_init, beta_init
        
        return self.xi, self.beta
    
    def tail_var(self, confidence=0.99):
        """
        Расчет VaR для хвоста с помощью GPD
        VaR = threshold - beta/xi * ((n/nu * (1-confidence))^(-xi) - 1)
        """
        if self.xi is None:
            self.fit_gpd_pot()
        
        n = len(self.returns)
        p = 1 - confidence
        
        if self.xi == 0:
            var_tail = self.threshold - self.beta * np.log(n / self.nu * p)
        else:
            var_tail = self.threshold - (self.beta / self.xi) * ((n / self.nu * p) ** (-self.xi) - 1)
        
        return var_tail
    
    def tail_es(self, confidence=0.99):
        """
        Expected Shortfall для хвоста (через GPD)
        ES = VaR / (1-ξ) + (β - ξ*threshold) / (1-ξ)
        """
        if self.xi is None:
            self.fit_gpd_pot()
        
        var_tail = self.tail_var(confidence)
        
        if self.xi < 1:
            es_tail = var_tail / (1 - self.xi) + (self.beta - self.xi * self.threshold) / (1 - self.xi)
        else:
            es_tail = var_tail * 2  # fallback
        
        return es_tail

def compare_risk_measures(returns, confidence_levels=[0.95, 0.99, 0.995]):
    """
    Сравнение различных мер риска
    """
    results = []
    
    for conf in confidence_levels:
        # Нормальный VaR
        mu = np.mean(returns)
        sigma = np.std(returns)
        from scipy import stats
        normal_var = mu + sigma * stats.norm.ppf(1 - conf)
        normal_es = mu - sigma * stats.norm.pdf(stats.norm.ppf(1 - conf)) / (1 - conf)
        
        # Исторический VaR
        hist_var = np.percentile(returns, (1 - conf) * 100)
        hist_es = np.mean(returns[returns <= hist_var])
        
        # EVT VaR/ES
        evt = ExtremeValueTheory(returns)
        evt.fit_gpd_pot()
        evt_var = evt.tail_var(conf)
        evt_es = evt.tail_es(conf)
        
        results.append({
            'confidence': conf,
            'normal_var': normal_var,
            'normal_es': normal_es,
            'hist_var': hist_var,
            'hist_es': hist_es,
            'evt_var': evt_var,
            'evt_es': evt_es,
            'evt_xi': evt.xi
        })
    
    return results

if __name__ == "__main__":
    from data_generation import generate_regime_switching_returns
    
    np.random.seed(42)
    returns, regime = generate_regime_switching_returns(n_days=3000)
    
    print("=" * 60)
    print("Extreme Value Theory (EVT) - Моделирование хвостов")
    print("=" * 60)
    
    # Выбор порога
    evt = ExtremeValueTheory(returns)
    threshold, thresholds, mean_excess = evt.select_threshold()
    
    print(f"\nВыбранный порог (90-й процентиль): {threshold:.6f}")
    print(f"Доля убытков ниже порога: {np.mean(returns < threshold):.2%}")
    
    # Оценка параметров GPD
    evt.fit_gpd_pot(threshold)
    print(f"\nОцененные параметры GPD:")
    print(f"  ξ (форма хвоста): {evt.xi:.4f}")
    print(f"  β (масштаб): {evt.beta:.6f}")
    print(f"  Количество превышений: {evt.nu}")
    
    if evt.xi > 0:
        print(f"  → Хвост тяжелый (ξ > 0), распределение Фреше")
    elif evt.xi == 0:
        print(f"  → Хвост экспоненциальный")
    else:
        print(f"  → Хвост короткий (ограниченный)")
    
    # Сравнение мер риска
    print("\n" + "=" * 60)
    print("Сравнение мер риска")
    print("=" * 60)
    
    results = compare_risk_measures(returns)
    
    for res in results:
        print(f"\nДоверительный уровень: {res['confidence']*100:.1f}%")
        print(f"  Нормальный VaR: {res['normal_var']:.6f} | ES: {res['normal_es']:.6f}")
        print(f"  Исторический VaR: {res['hist_var']:.6f} | ES: {res['hist_es']:.6f}")
        print(f"  EVT VaR: {res['evt_var']:.6f} | ES: {res['evt_es']:.6f}")
        print(f"  EVT параметр ξ: {res['evt_xi']:.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean Excess Plot
    axes[0, 0].plot(thresholds, mean_excess, 'o-', markersize=3)
    axes[0, 0].axvline(x=threshold, color='red', linestyle='--', label=f'Выбранный порог: {threshold:.4f}')
    axes[0, 0].set_xlabel('Порог u')
    axes[0, 0].set_ylabel('Среднее превышение e(u)')
    axes[0, 0].set_title('Mean Excess Plot (выбор порога)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Распределение превышений
    axes[0, 1].hist(evt.exceedances, bins=30, density=True, alpha=0.7, label='Эмпирическое')
    x_gpd = np.linspace(0, np.max(evt.exceedances), 100)
    y_gpd = GPD.pdf(x_gpd, evt.xi, evt.beta)
    axes[0, 1].plot(x_gpd, y_gpd, 'r-', linewidth=2, label=f'GPD (ξ={evt.xi:.3f}, β={evt.beta:.3f})')
    axes[0, 1].set_xlabel('Превышение над порогом')
    axes[0, 1].set_ylabel('Плотность')
    axes[0, 1].set_title('Аппроксимация хвоста GPD')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # QQ-plot для GPD
    from scipy import stats as sp_stats
    sorted_exceed = np.sort(evt.exceedances)
    theoretical_quantiles = GPD.quantile(np.linspace(0.5/evt.nu, 1-0.5/evt.nu, evt.nu), evt.xi, evt.beta)
    axes[1, 0].scatter(theoretical_quantiles, sorted_exceed, s=5, alpha=0.6)
    axes[1, 0].plot([0, max(sorted_exceed)], [0, max(sorted_exceed)], 'r--', label='y=x')
    axes[1, 0].set_xlabel('Теоретические квантили GPD')
    axes[1, 0].set_ylabel('Эмпирические квантили')
    axes[1, 0].set_title('QQ-plot: GPD vs эмпирическое распределение')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Сравнение VaR/ES
    confs = [r['confidence'] for r in results]
    normal_vars = [r['normal_var'] for r in results]
    hist_vars = [r['hist_var'] for r in results]
    evt_vars = [r['evt_var'] for r in results]
    
    x = np.arange(len(confs))
    width = 0.25
    axes[1, 1].bar(x - width, normal_vars, width, label='Нормальный VaR')
    axes[1, 1].bar(x, hist_vars, width, label='Исторический VaR')
    axes[1, 1].bar(x + width, evt_vars, width, label='EVT VaR')
    axes[1, 1].set_xlabel('Доверительный уровень')
    axes[1, 1].set_ylabel('VaR')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{c*100:.0f}%' for c in confs])
    axes[1, 1].set_title('Сравнение VaR для разных методов')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualization/results_evt.png', dpi=150)
    plt.show()
    
    print("\nГрафик сохранен как 'results_evt.png'")