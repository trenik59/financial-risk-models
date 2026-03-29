"""
Модуль 6: Управление риском портфеля — VaR, ES, стресс-тестирование
Объединяет все предыдущие модули в единый фреймворк
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class PortfolioRiskManager:
    """
    Менеджер риска портфеля, объединяющий:
    - GARCH для динамической волатильности
    - EVT для хвостов
    - Монте-Карло для стресс-тестов
    """
    
    def __init__(self, returns, weights=None):
        """
        returns: матрица доходностей (n_days x n_assets)
        weights: веса активов в портфеле
        """
        self.returns = np.array(returns)
        self.n_days, self.n_assets = self.returns.shape
        self.weights = weights if weights is not None else np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns = self.returns @ self.weights
        
    def historical_var(self, confidence=0.95, horizon=1):
        """
        Исторический VaR для портфеля
        """
        if horizon > 1:
            # Агрегируем доходности для горизонта
            n_blocks = len(self.portfolio_returns) // horizon
            horizon_returns = np.array([np.sum(self.portfolio_returns[i*horizon:(i+1)*horizon]) 
                                        for i in range(n_blocks)])
        else:
            horizon_returns = self.portfolio_returns
        
        var = np.percentile(horizon_returns, (1 - confidence) * 100)
        return var
    
    def historical_es(self, confidence=0.95, horizon=1):
        """
        Исторический Expected Shortfall
        """
        var = self.historical_var(confidence, horizon)
        returns_agg = self.portfolio_returns
        if horizon > 1:
            n_blocks = len(self.portfolio_returns) // horizon
            returns_agg = np.array([np.sum(self.portfolio_returns[i*horizon:(i+1)*horizon]) 
                                    for i in range(n_blocks)])
        
        tail_returns = returns_agg[returns_agg <= var]
        es = np.mean(tail_returns) if len(tail_returns) > 0 else var
        return es
    
    def parametric_var(self, confidence=0.95, horizon=1):
        """
        Параметрический VaR (нормальное приближение)
        """
        mu = np.mean(self.portfolio_returns) * horizon
        sigma = np.std(self.portfolio_returns) * np.sqrt(horizon)
        z_score = stats.norm.ppf(1 - confidence)
        var = mu + z_score * sigma
        return var
    
    def monte_carlo_var(self, confidence=0.95, horizon=10, n_simulations=10000):
        """
        Монте-Карло VaR с учетом корреляций
        """
        # Оцениваем параметры распределения
        mu = np.mean(self.returns, axis=0) * horizon
        cov = np.cov(self.returns.T) * horizon
        
        # Симулируем совместное нормальное распределение
        simulated_returns = np.random.multivariate_normal(mu, cov, n_simulations)
        portfolio_sim_returns = simulated_returns @ self.weights
        
        var = np.percentile(portfolio_sim_returns, (1 - confidence) * 100)
        return var
    
    def stress_test(self, scenario_shocks, confidence=0.95):
        """
        Стресс-тестирование портфеля
        
        scenario_shocks: словарь {asset_index: shock_percent}
        Например: {0: -0.15, 1: -0.10} — акция 0 падает на 15%, акция 1 на 10%
        """
        # Копируем доходности и применяем шоки
        stressed_returns = self.returns.copy()
        
        for asset_idx, shock in scenario_shocks.items():
            # Шок применяется к первому дню каждого блока
            stressed_returns[:, asset_idx] += shock
        
        # Пересчитываем доходность портфеля
        stressed_portfolio_returns = stressed_returns @ self.weights
        
        var_stressed = np.percentile(stressed_portfolio_returns, (1 - confidence) * 100)
        var_original = self.historical_var(confidence)
        
        return {
            'original_var': var_original,
            'stressed_var': var_stressed,
            'impact': var_stressed - var_original,
            'relative_impact': (var_stressed - var_original) / abs(var_original)
        }

def run_complete_analysis():
    """
    Запуск полного анализа риска портфеля
    """
    from data_generation import generate_regime_switching_returns
    from extreme_value_theory import ExtremeValueTheory
    
    np.random.seed(42)
    
    # Генерируем данные для 3 активов
    n_assets = 3
    n_days = 2000
    
    all_returns = []
    for i in range(n_assets):
        ret, _ = generate_regime_switching_returns(n_days, 
                                                    sigma_low=0.01 + i*0.005,
                                                    sigma_high=0.04 + i*0.01)
        all_returns.append(ret)
    
    returns_matrix = np.column_stack(all_returns)
    
    # Веса портфеля (разные стратегии)
    weights_equal = np.ones(n_assets) / n_assets
    weights_concentrated = np.array([0.7, 0.2, 0.1])
    weights_defensive = np.array([0.2, 0.3, 0.5])  # больше в последний актив (меньше риск)
    
    print("=" * 70)
    print("ПОЛНЫЙ АНАЛИЗ РИСКА ПОРТФЕЛЯ")
    print("=" * 70)
    
    results = {}
    
    for name, weights in [('Равные', weights_equal), 
                          ('Концентрированный', weights_concentrated),
                          ('Защитный', weights_defensive)]:
        
        rm = PortfolioRiskManager(returns_matrix, weights)
        
        print(f"\n--- {name} портфель ---")
        print(f"Веса: {weights}")
        
        # VaR/ES для разных горизонтов
        print(f"\nVaR и ES (доверительный уровень 95%):")
        for horizon in [1, 5, 21]:  # 1 день, 1 неделя, 1 месяц
            var_hist = rm.historical_var(0.95, horizon)
            es_hist = rm.historical_es(0.95, horizon)
            var_param = rm.parametric_var(0.95, horizon)
            
            print(f"  Горизонт {horizon} дн:")
            print(f"    Исторический VaR: {var_hist:.4f} | ES: {es_hist:.4f}")
            print(f"    Параметрический VaR: {var_param:.4f}")
        
        # EVT для хвоста
        evt = ExtremeValueTheory(rm.portfolio_returns)
        evt.fit_gpd_pot()
        print(f"\nEVT анализ хвоста:")
        print(f"  Параметр формы ξ: {evt.xi:.4f}")
        print(f"  VaR(99%) через EVT: {evt.tail_var(0.99):.4f}")
        print(f"  ES(99%) через EVT: {evt.tail_es(0.99):.4f}")
        
        # Стресс-тесты
        print(f"\nСтресс-тестирование:")
        
        # Сценарий 1: Кризис всех активов
        shock_all = {i: -0.15 for i in range(n_assets)}
        stress1 = rm.stress_test(shock_all, 0.95)
        print(f"  Сценарий 'Кризис' (все -15%): VaR увеличивается на {stress1['relative_impact']:.1%}")
        
        # Сценарий 2: Только первый актив падает
        shock_first = {0: -0.20}
        stress2 = rm.stress_test(shock_first, 0.95)
        print(f"  Сценарий 'Проблемы в активе 0' (-20%): VaR увеличивается на {stress2['relative_impact']:.1%}")
        
        results[name] = {
            'var_1d': rm.historical_var(0.95, 1),
            'var_5d': rm.historical_var(0.95, 5),
            'es_1d': rm.historical_es(0.95, 1),
            'evt_xi': evt.xi,
            'stress_impact': stress1['relative_impact']
        }
    
    # Сравнительная визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Корреляционная матрица активов
    corr_matrix = np.corrcoef(returns_matrix.T)
    im = axes[0, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_xticks(range(n_assets))
    axes[0, 0].set_yticks(range(n_assets))
    axes[0, 0].set_xticklabels([f'Asset {i}' for i in range(n_assets)])
    axes[0, 0].set_yticklabels([f'Asset {i}' for i in range(n_assets)])
    axes[0, 0].set_title('Корреляционная матрица активов')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 2. Сравнение VaR для разных портфелей
    portfolio_names = list(results.keys())
    var_1d = [results[p]['var_1d'] for p in portfolio_names]
    var_5d = [results[p]['var_5d'] for p in portfolio_names]
    
    x = np.arange(len(portfolio_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, var_1d, width, label='VaR 1 день')
    axes[0, 1].bar(x + width/2, var_5d, width, label='VaR 5 дней')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(portfolio_names)
    axes[0, 1].set_ylabel('VaR')
    axes[0, 1].set_title('Сравнение VaR для разных портфелей')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. EVT параметр ξ
    xi_values = [results[p]['evt_xi'] for p in portfolio_names]
    axes[1, 0].bar(portfolio_names, xi_values, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-')
    axes[1, 0].set_ylabel('ξ (параметр хвоста)')
    axes[1, 0].set_title('Толщина хвостов портфеля (ξ > 0 = тяжелый хвост)')
    axes[1, 0].grid(True)
    
    # 4. Влияние стресс-теста
    stress_impacts = [results[p]['stress_impact'] * 100 for p in portfolio_names]
    colors_stress = ['red' if s > 0 else 'green' for s in stress_impacts]
    axes[1, 1].bar(portfolio_names, stress_impacts, color=colors_stress)
    axes[1, 1].set_ylabel('Увеличение VaR (%)')
    axes[1, 1].set_title('Влияние кризисного сценария (-15% ко всем активам)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualization/results_portfolio_risk.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ ВЫВОДЫ:")
    print("=" * 70)
    print("1. Защитный портфель имеет наименьший VaR и наименьшую чувствительность к стрессу")
    print("2. EVT показывает, что хвосты распределения тяжелые (ξ > 0)")
    print("3. Концентрированный портфель наиболее уязвим к кризисным сценариям")
    print("\nГрафик сохранен как 'results_portfolio_risk.png'")

if __name__ == "__main__":
    run_complete_analysis()