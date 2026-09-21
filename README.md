# Bio-COMSOL

Ионный транспорт на графах: модель Нернста-Планка-Пуассона для синтетических ионных систем и биологической ткани.

Validated against analytical solutions for Nernst equilibrium, 1D diffusion (first-order convergence), and Boltzmann distribution.

![CI](https://github.com/andry7610/Bio_comsol/actions/workflows/python-package.yml/badge.svg)

## Что это

Модульная система для моделирования:
- Ионного транспорта в электролитах и биологических тканях
- Мембранных потенциалов через уравнение Пуассона
- Нейронов Ходжкина-Хаксли с динамическим Нернстом
- Синапсов и сетей с обучением через STDP
- Самосогласованного потенциала — phi зависит от концентраций

## Структура

Bio_comsol/
  biological/
    diffusion.py          # Нернст-Планк-Пуассон, Пикар / Ньютон
    neuron.py             # Ходжкин-Хаксли, динамический Нернст
    synapse.py            # Синапсы, STDP, детекция спайков
    ai_backends.py        # manual / mock / API
  network/
    graph.py              # Граф с пространственной метрикой
  configs/
    config.yaml           # Все параметры
  tests/
    test_biological.py    # 46 тестов: диффузия, Пуассон, Пикар, Ньютон
    test_neuron.py        # 41 тест: гейты, токи, Нернст, overflow
    test_synapse.py       # 33 теста: синапсы, STDP, интеграция
    test_main.py          # 17 тестов: конфиг, граф, солвер, запуск
    test_validation.py    # 4 теста: аналитическая валидация
  main.py                 # Точка входа: config + CLI
  config.yaml
  README.md
  LICENSE                 # MIT

## Физика

### Ионный транспорт (Нернст-Планк-Пуассон)

Поток Нернста-Планка:
J_i = -D_i * grad(c_i) - (D_i * z_i * F / RT) * c_i * grad(phi)

Уравнение Пуассона:
laplacian(phi) = -(F / eps) * sum(z_i * c_i)

Схемы:
- Пикар — линейная сходимость, 20 итераций
- Ньютон — квадратичная сходимость, 1–2 итерации (аналитический якобиан)

### Нейрон Ходжкина-Хаксли

C_m * dV/dt = I_ext - I_Na - I_K - I_L

С динамическими равновесными потенциалами Нернста:
E_i = (RT / z_i F) * ln(c_i_out / c_i_in)

Защита от overflow: np.clip(V, -100, 100) перед экспонентами.

### Синапсы и STDP

Проводимостная модель с детекцией спайков и обучением:
- LTP — pre до post, вес растёт
- LTD — post до pre, вес падает
- Клиппинг весов, экспоненциальное затухание

## Валидация

Четыре аналитических теста в tests/test_validation.py:

| Тест | Аналитика | Допуск |
|------|-----------|--------|
| Нернст | E = (RT/zF)*ln(c_out/c_in) | rtol < 1e-3 |
| 1D-диффузия | c0/2 * [1 - erf(x / 2*sqrt(Dt))] | max err < 1e-2 |
| Порядок сходимости | Ошибка пропорциональна dt | ratio 1.7..2.3 |
| Больцман | c ~ exp(-z*phi/kT) | R2 > 0.99 |

## Запуск

pip install numpy scipy pyyaml pytest

# Полная симуляция из конфига
python main.py

# С CLI-параметрами
python main.py --steps 1000 --dt 1e-4

# Тесты
python -m pytest tests/ -v

# Только валидация
python -m pytest tests/test_validation.py -v

## Конфигурация

Всё в config.yaml:

graph:
  N: 50
  topology: chain
  conductivity: 1.0
  cross_section: 1e-8

ions:
  - {name: Na, D: 0.01, z: 1, c0: 10.0}
  - {name: K,  D: 0.02, z: 1, c0: 100.0}
  - {name: Cl, D: 0.015, z: -1, c0: 110.0}
  - {name: Ca, D: 0.008, z: 2, c0: 1.0}

solver:
  method: newton
  self_consistent: true
  dt: 1e-4

neuron:
  enabled: true
  indices: [20, 25, 30]

synapse:
  enabled: true
  stdp: true

## ИИ-бэкенды

Три режима:
- manual — без ИИ, чистый расчёт
- mock — тестовый бэкенд для CI
- api — внешний API (DeepSeek, YandexGPT, локальная модель)

## CI

GitHub Actions запускает при каждом пуше:
- 156 тестов (pytest)
- Валидация против аналитики
- Проверка отсутствия NaN и отрицательных концентраций

Статус: зелёный, 0 warnings, ~9 секунд

## Статус проекта

v0.9 — полная модель: от ионной диффузии до сети нейронов с обучением.

Компонент                  | Статус
---------------------------|--------
Нернст-Планк-Пуассон       | готово
Квазинейтральность          | готово
Самосогласованный потенциал | готово
Метод Ньютона              | готово
Ходжкин-Хаксли             | готово
Динамический Нернст        | готово
Синапсы + STDP             | готово
Аналитическая валидация    | готово
ИИ-бэкенды                 | готово

В планах (v1.0):
- Нейромедиаторы и модуляция
- Глия
- 2D/3D-ткани
- Веб-интерфейс

## Лицензия

MIT — используйте, форкайте, развивайте.
