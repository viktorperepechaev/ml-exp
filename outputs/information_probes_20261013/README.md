# OptDigits: информация и топологические признаки

Полный результат: [report_ru.md](report_ru.md).

Основная находка: четыре числа отверстий позволяют использовать MLP 68 → 6 → 10 с 484 параметрами. WINDEP: 94,296% против 92,888% у геометрической MLP с 508 параметрами в этом новом протоколе. Среднее по десяти отдельным моделям. Это исследовательский результат на ранее использовавшемся датасете, а не доказательство превосходства Shannon MI или нового SOTA.

- `run_study.py`: новый полный запуск, 510 подборов и 158 итоговых моделей; четыре основных представления, три проверки состава признаков.
- `analyze_study.py`: таблицы, парные интервалы, рисунки.
- `audit_study.py`: независимая стандартизация и повторный расчёт всех 826 массивов предсказаний.
- `formal_checks.py`: двойственные свидетельства линейных моделей, экспорт 484-параметрических моделей.
- `mechanism_diagnostics.py`: анализ редких компонент и вмешательство в их значения при фиксированных весах (после основных результатов).
- `make_report.py`: сборка отчёта из результатов.
- `run/`: зафиксированный выполненный эксперимент, исходник использованного runner, протокол, признаки, модели и предсказания.
- `run/compact_holes/`: десять моделей с 484 параметрами и параметры стандартизации.
- `installation_manifest.json`: контрольные суммы комплекта.

Из корня репозитория:

```bash
uv run --locked python outputs/information_probes_20261013/run_study.py --repo . --run runs/information-probes-repeat --workers 4
uv run --locked python outputs/information_probes_20261013/analyze_study.py --run runs/information-probes-repeat
uv run --locked python outputs/information_probes_20261013/audit_study.py --repo . --run runs/information-probes-repeat
uv run --locked python outputs/information_probes_20261013/formal_checks.py --run runs/information-probes-repeat
uv run --locked python outputs/information_probes_20261013/mechanism_diagnostics.py --run runs/information-probes-repeat
```

Проверить только сохранённый эксперимент:

```bash
uv run --locked python outputs/information_probes_20261013/audit_study.py --repo . --run outputs/information_probes_20261013/run
```

Нужен существующий датасет `work/real_data/optdigits.zip` и исходные вычислители признаков репозитория. Код ничего не скачивает. Версии зависимостей закреплены в корневых pyproject.toml и uv.lock.
