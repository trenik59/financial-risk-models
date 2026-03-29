"""
Модуль 4: Опционы и греки — рукописная реализация Блэка-Шоулза
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BlackScholesManual:
    """
    Реализация формулы Блэка-Шоулза своими руками
    Без использования готовых функций из scipy.stats (но для нормального CDF используем)
    """
    
    @staticmethod
    def norm_cdf(x):
        """CDF стандартного нормального распределения (аппроксимация)"""
        return stats.norm.cdf(x)
    
    @staticmethod
    def norm_pdf(x):
        """PDF стандартного нормального распределения"""
        return stats.norm.pdf(x)
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Вычисление d1 в формуле Блэка-Шоулза"""
        if T <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Вычисление d2 в формуле Блэка-Шоулза"""
        if T <= 0:
            return 0
        return BlackScholesManual.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @classmethod
    def call_price(cls, S, K, T, r, sigma):
        """Цена европейского опциона колл"""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        price = S * cls.norm_cdf(d1) - K * np.exp(-r * T) * cls.norm_cdf(d2)
        return price
    
    @classmethod
    def put_price(cls, S, K, T, r, sigma):
        """Цена европейского опциона пут"""
        if T <= 0:
            return max(K - S, 0)
        
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        price = K * np.exp(-r * T) * cls.norm_cdf(-d2) - S * cls.norm_cdf(-d1)
        return price
    
    @classmethod
    def delta_call(cls, S, K, T, r, sigma):
        """Дельта колла: ∂C/∂S = N(d1)"""
        d1 = cls.d1(S, K, T, r, sigma)
        return cls.norm_cdf(d1)
    
    @classmethod
    def delta_put(cls, S, K, T, r, sigma):
        """Дельта пута: ∂P/∂S = N(d1) - 1"""
        d1 = cls.d1(S, K, T, r, sigma)
        return cls.norm_cdf(d1) - 1
    
    @classmethod
    def gamma(cls, S, K, T, r, sigma):
        """Гамма (общая для колла и пута): ∂²V/∂S² = φ(d1)/(S·σ·√T)"""
        if T <= 0:
            return 0
        d1 = cls.d1(S, K, T, r, sigma)
        return cls.norm_pdf(d1) / (S * sigma * np.sqrt(T))
    
    @classmethod
    def vega(cls, S, K, T, r, sigma):
        """Вега: ∂V/∂σ = S·φ(d1)·√T"""
        if T <= 0:
            return 0
        d1 = cls.d1(S, K, T, r, sigma)
        return S * cls.norm_pdf(d1) * np.sqrt(T)
    
    @classmethod
    def theta_call(cls, S, K, T, r, sigma):
        """Тета колла: ∂C/∂t"""
        if T <= 0:
            return 0
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        theta = (-S * cls.norm_pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * cls.norm_cdf(d2))
        return theta
    
    @classmethod
    def theta_put(cls, S, K, T, r, sigma):
        """Тета пута"""
        if T <= 0:
            return 0
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        theta = (-S * cls.norm_pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * cls.norm_cdf(-d2))
        return theta

class OptionPortfolio:
    """
    Портфель опционов для расчета риск-метрик
    """
    
    def __init__(self):
        self.positions = []  # каждый элемент: (type, S, K, T, r, sigma, quantity)
        # type: 'call' или 'put'
    
    def add_position(self, option_type, S, K, T, r, sigma, quantity=1):
        """Добавление позиции в портфель"""
        self.positions.append((option_type, S, K, T, r, sigma, quantity))
    
    def portfolio_price(self):
        """Текущая стоимость портфеля"""
        total = 0
        for opt_type, S, K, T, r, sigma, q in self.positions:
            if opt_type == 'call':
                price = BlackScholesManual.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholesManual.put_price(S, K, T, r, sigma)
            total += price * q
        return total
    
    def portfolio_delta(self):
        """Дельта портфеля (чувствительность к цене базового актива)"""
        total_delta = 0
        for opt_type, S, K, T, r, sigma, q in self.positions:
            if opt_type == 'call':
                delta = BlackScholesManual.delta_call(S, K, T, r, sigma)
            else:
                delta = BlackScholesManual.delta_put(S, K, T, r, sigma)
            total_delta += delta * q
        return total_delta
    
    def portfolio_gamma(self):
        """Гамма портфеля (скорость изменения дельты)"""
        total_gamma = 0
        for opt_type, S, K, T, r, sigma, q in self.positions:
            gamma = BlackScholesManual.gamma(S, K, T, r, sigma)
            total_gamma += gamma * q
        return total_gamma
    
    def portfolio_vega(self):
        """Вега портфеля (чувствительность к волатильности)"""
        total_vega = 0
        for opt_type, S, K, T, r, sigma, q in self.positions:
            vega = BlackScholesManual.vega(S, K, T, r, sigma)
            total_vega += vega * q
        return total_vega
    
    def pnl_scenario(self, S_new, sigma_new=None):
        """
        P&L при изменении цены базового актива и волатильности
        """
        old_price = self.portfolio_price()
        new_price = 0
        
        for opt_type, S_old, K, T, r, sigma_old, q in self.positions:
            sigma = sigma_new if sigma_new is not None else sigma_old
            if opt_type == 'call':
                new_price += BlackScholesManual.call_price(S_new, K, T, r, sigma) * q
            else:
                new_price += BlackScholesManual.put_price(S_new, K, T, r, sigma) * q
        
        return new_price - old_price

if __name__ == "__main__":
    print("=" * 60)
    print("Опционы и греки — рукописная реализация Блэка-Шоулза")
    print("=" * 60)
    
    # Параметры
    S = 100.0      # текущая цена базового актива
    K = 105.0      # страйк
    T = 0.5        # время до экспирации (0.5 года)
    r = 0.05       # безрисковая ставка (5%)
    sigma = 0.25   # волатильность (25%)
    
    # Цены опционов
    call = BlackScholesManual.call_price(S, K, T, r, sigma)
    put = BlackScholesManual.put_price(S, K, T, r, sigma)
    
    print(f"\nБазовые параметры:")
    print(f"  S = {S}, K = {K}, T = {T}, r = {r:.2%}, σ = {sigma:.2%}")
    print(f"\nЦены опционов:")
    print(f"  Колл: {call:.4f}")
    print(f"  Пут:  {put:.4f}")
    print(f"  Паритет пут-колл: C - P = {call - put:.4f}, S - K*e^(-rT) = {S - K*np.exp(-r*T):.4f}")
    
    print(f"\nГреки для колла:")
    print(f"  Дельта: {BlackScholesManual.delta_call(S, K, T, r, sigma):.4f}")
    print(f"  Гамма:  {BlackScholesManual.gamma(S, K, T, r, sigma):.4f}")
    print(f"  Вега:   {BlackScholesManual.vega(S, K, T, r, sigma):.4f}")
    print(f"  Тета:   {BlackScholesManual.theta_call(S, K, T, r, sigma):.4f}")
    
    # Создание портфеля
    portfolio = OptionPortfolio()
    portfolio.add_position('call', S, K=100, T=0.5, r=0.05, sigma=0.25, quantity=2)
    portfolio.add_position('put', S, K=110, T=0.5, r=0.05, sigma=0.25, quantity=1)
    
    print(f"\nПортфель опционов:")
    print(f"  Стоимость портфеля: {portfolio.portfolio_price():.4f}")
    print(f"  Дельта портфеля: {portfolio.portfolio_delta():.4f}")
    print(f"  Гамма портфеля: {portfolio.portfolio_gamma():.4f}")
    print(f"  Вега портфеля: {portfolio.portfolio_vega():.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Цена колла vs цена базового актива
    S_range = np.linspace(50, 150, 100)
    call_prices = [BlackScholesManual.call_price(Si, K, T, r, sigma) for Si in S_range]
    axes[0, 0].plot(S_range, call_prices)
    axes[0, 0].axvline(x=K, color='r', linestyle='--', label=f'Страйк K={K}')
    axes[0, 0].set_xlabel('Цена базового актива S')
    axes[0, 0].set_ylabel('Цена колла')
    axes[0, 0].set_title('Цена европейского опциона колл')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Дельта vs цена базового актива
    delta_values = [BlackScholesManual.delta_call(Si, K, T, r, sigma) for Si in S_range]
    axes[0, 1].plot(S_range, delta_values)
    axes[0, 1].set_xlabel('Цена базового актива S')
    axes[0, 1].set_ylabel('Дельта')
    axes[0, 1].set_title('Дельта колла')
    axes[0, 1].grid(True)
    
    # Гамма vs цена базового актива
    gamma_values = [BlackScholesManual.gamma(Si, K, T, r, sigma) for Si in S_range]
    axes[1, 0].plot(S_range, gamma_values)
    axes[1, 0].set_xlabel('Цена базового актива S')
    axes[1, 0].set_ylabel('Гамма')
    axes[1, 0].set_title('Гамма (максимум около страйка)')
    axes[1, 0].grid(True)
    
    # P&L портфеля при изменении цены
    S_scenarios = np.linspace(80, 120, 50)
    pnl_scenarios = [portfolio.pnl_scenario(Si) for Si in S_scenarios]
    axes[1, 1].plot(S_scenarios, pnl_scenarios)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_xlabel('Новая цена базового актива')
    axes[1, 1].set_ylabel('P&L')
    axes[1, 1].set_title('P&L портфеля опционов')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualization/results_options.png', dpi=150)
    plt.show()
    
    print("\nГрафик сохранен как 'results_options.png'")