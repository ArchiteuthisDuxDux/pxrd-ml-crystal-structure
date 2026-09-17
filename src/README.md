# Что лежит в `src/`

Ноутбуки содержат исследовательскую логику, графики и выводы.

## Пути и общие функции

- `project_paths.py` — единственная конфигурация новой иерархии.
- `config.py` — параметры подготовки спектров и целевых голов.
- `utils_spectra.py` — загрузка, очистка и статистики `.npy`.

## Real-данные

- `step1_clean_and_masks.py` — очистка RRUFF/opXRD и дедупликация.
- `step2_ft_pool.py` — маски голов и FT-пулы.
- `step3_modes.py` — эмпирические режимы съёмки.
- `step4_fix_wavelengths.py` — корректный порядок Kα1/Kα2.
- `check_wavelengths.py` и `resplit_ft.py` — проверки и историческое исправление split.

## Synthetic

- `synth_config.py` — замороженная конфигурация генератора v4.
- `synth_physics.py` — отражения, structure factors и лабораторные искажения.
- `synth_run.py` — возобновляемая параллельная генерация.
- `synth_finalize.py` — объединение шардов и проверка схемы.
- `validate_engine.py`, `validate_random.py`, `make_previews.py` — контроль генератора.
- `precompute_mp_sg.py`, `mp_sg_worker.py` — безопасный расчёт SG для crystalDB.

## Общая сетка и domain shift

- `step5_preprocess_grid.py` и `validate_stageB.py` — сетка 0–90° × 4096.
- `step6_domain_features.py`, `step6_domain_check.py` — первый набор физических признаков и классификатор.
- `domain_features.py`, `domain_diag.py`, `compare_iterations.py` — финальная wavelength-matched диагностика.

## Модели и разметка

- `compare_v1_v2.py`, `diag_pretrain.py` — единая оценка pretrain.
- `pseudo_labeling.py`, `pseudo_labeling_stage2.py` — воспроизводимость отклонённого опыта.
- `enrich_rruff_spacegroups.py` — консервативное RRUFF–IMA сопоставление.
- `build_rruff_sg_ft_pools.py` — новые пулы с суффиксом `with_rruff_sg`.
