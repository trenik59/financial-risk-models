# Quant Risk Project Makefile
# Команды: make make run, make clean, make help

.PHONY: run run-all clean help

# Цвета для вывода
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[1;33m
NC := \033[0m 

# Запуск всех модулей по порядку
run-all:
	@echo "$(GREEN) Запуск всех модулей Quant Risk проекта...$(NC)"
	@echo "$(YELLOW)1/5 Генерация данных...$(NC)"
	python data_generation.py
	@echo "$(YELLOW)2/5 Extreme Value Theory...$(NC)"
	python extreme_value_theory.py
	@echo "$(YELLOW)3/5 Опционы и греки...$(NC)"
	python options_pricing.py
	@echo "$(YELLOW)4/5 Бэктестинг...$(NC)"
	python backtesting.py
	@echo "$(YELLOW)5/5 Анализ портфеля...$(NC)"
	python portfolio_risk.py
	@echo "$(GREEN) Все модули выполнены! Графики сохранены в папке проекта.$(NC)"

# Запуск отдельного модуля (make run module=data_generation)
run:
	@if [ -z "$(module)" ]; then \
		echo "$(RED) Укажи модуль: make run module=имя_файла$(NC)"; \
		echo "Доступные модули: data_generation, garch_calibration, extreme_value_theory, options_pricing, backtesting, portfolio_risk"; \
		exit 1; \
	fi
	@echo "$(GREEN)▶ Запуск $(module).py...$(NC)"
	python $(module).py

# Очистка графиков и временных файлов
clean:
	@echo "$(YELLOW)🧹 Очистка результатов...$(NC)"
	rm -f results_*.png
	rm -rf __pycache__
	rm -rf *.pyc
	@echo "$(GREEN) Очищено$(NC)"

# Показать все доступные команды
help:
	@echo "$(GREEN)📋 Доступные команды:$(NC)"
	@echo "  make run-all    - запустить все 6 модулей по порядку"
	@echo "  make run module=имя - запустить конкретный модуль"
	@echo "  make clean      - удалить все сгенерированные графики"
	@echo "  make help       - показать эту справку"
	@echo ""
	@echo "$(YELLOW)Примеры:$(NC)"
	@echo "  make run module=data_generation"
	@echo "  make run module=portfolio_risk"