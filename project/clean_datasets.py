"""
Удаление ненужных датасетов из папки datasets/.

Оставляет только датасеты из списка KEEP_PATTERNS.
Сопоставление нечёткое: учитывает префиксы (pmlb_, openml_123_, keel_ и т.д.),
регистр, спецсимволы и дефисы.

ВНИМАНИЕ: скрипт сначала показывает что будет удалено (--dry-run по умолчанию).
Для реального удаления добавь флаг --delete.

Запуск:
    python clean_datasets.py                       # показать список (без удаления)
    python clean_datasets.py --delete              # реально удалить
    python clean_datasets.py --datasets my_path/  # другая папка
    python clean_datasets.py --delete --verbose    # удалять с подробным выводом
"""

import argparse
import re
import shutil
from pathlib import Path

# ── Список датасетов для СОХРАНЕНИЯ ──────────────────────────────────────
KEEP_PATTERNS = [
    "allbp", "allhypo", "ann_thyroid", "balance_scale", "cars", "schizo",
    "new_thyroid", "penguins", "analcatdata_germangss", "vehicle", "nursery",
    "allhyper", "car_evaluation", "analcatdata_authorship", "allrep",
    "ecoli", "calendarDOW", "page_blocks", "wine_quality_red",
    "analcatdata_dmft", "dermatology", "wine_quality_white", "yeast", "fars",
    "Corporate_Credit", "Corporate_Credit_Ratings", "Corporate_Credit_Rating",
    "Corporate_Credit_Rating_Classification", "Multclass_Classification_for_Corpo",
    "artificial-characters", "Midwest_Survey", "Midwest_Survey_nominal",
    "melbourne_airbnb", "mental_health_detection", "PriceRunner",
    "UNIX_user_data", "cnae-9", "Otto-Group-Product-Classification-C",
    "diggle_table_a2", "MiceProtein", "Ecoli", "ipums_la_97-small",
    "autoUniv-au6-750", "autoUniv-au6-400", "autoUniv-au6-1000", "MIC",
    "EDA-Home-Mortgage-NY", "ipums_la_98-small", "covertype", "braziltourism",
    "BNG(glass,nominal,137781)", "BNG(glass)", "steel-plates-fault",
    "drug-directory", "fabert", "glass_clean", "data_scientist_salary",
    "analcatdata_marketing", "prnn_fglass", "cjs", "autos", "Flare",
    "Success-Rates-2", "EDA-Home-Mortgage-NY-2",
    "company_quality_and_valuation_finan", "EDA-Home-Mortgage-NY-Sampled",
    "wine-quality-red", "dgf_96f4164d", "autos_clean", "epiparo_extract",
    "eucalyptus", "anneal", "Cervical_Cancer_Risk_Factors", "thyroid-allbp",
    "thyroid-allhyper", "credit_risk_china", "autoUniv-au7-1100",
    "autoUniv-au7-500", "thyroid-dis", "heart-long-beach", "user-knowledge",
    "HCV_data", "heart-h", "mushroom", "dilbert",
    "mabbob_ela_as_2d_classify", "microaggregation2",
    "MIP-2016-classification", "anneal_clean",
    "MIP-2016-PAR10-classification", "thyroid-allrep", "Products",
    "page-blocks", "regensburg_pediatric_appendicitis", "WBCAtt",
    "football-player-position", "car", "odds", "total_score",
    "Student_Performance_on_an_Entrance_", "dataset_credit_risk_file_2",
    "Credit_Risk_Modeling", "hypothyroid", "rmftsa_sleepdata",
    "colleges_aaup", "Mobile_Price", "internet_firewall",
    "hepatitis_c_virus_hcv_for_egyptian_", "SOCC",
    "QSAR_Bioconcentration_classificatio", "SDSS17",
    "students_dropout_and_academic_succe", "website_phishing",
    "Diabetes130US", "stress", "Traffic_violations",
    "credit-score-classification-Hzl", "Risk_Level_Classification",
    "Credit_Score_Classification", "Financial-Risk-Assessment",
    "Credit_Score", "cars1", "thyroid-ann", "thyroid-new",
    "autoUniv-au4-2500", "autoUniv-au7-700", "abalone", "Engine1",
    "PopularKids", "seeds", "seismic-bumps", "vertebra-column",
    "analcatdata_halloffame", "baseball", "BNG(cmc,nominal,55296)",
    "maternal_health_risk", "Cirrhosis_Patient_Survival_Predicti",
    "OSMI_Mental_Health_in_Tech_Survey", "student_lifestyle_dataset",
    "Indicators_of_Anxiety_or_Depression", "balance-scale",
]


# ── Нормализация и сопоставление ─────────────────────────────────────────

def normalize(s: str) -> str:
    """Нижний регистр, все не-алфавитно-цифровые → '_', убираем краевые '_'."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def matches_any(folder_name: str, patterns: list[str]) -> tuple[bool, str]:
    """
    Проверяет совпадает ли имя папки с любым паттерном.

    Алгоритм:
      1. Нормализуем имя папки и паттерн.
      2. Разбиваем на токены (по '_').
      3. Ищем вхождение токенов паттерна как подпоследовательность токенов папки.
         Это корректно обрабатывает префиксы (pmlb_, openml_123_) и постфиксы.
      4. Для длинных паттернов (≥ 6 символов) также проверяем подстроку —
         это покрывает случаи вроде BNG_glass_nominal_137781.

    Возвращает (совпало: bool, совпавший паттерн: str).
    """
    fn_norm   = normalize(folder_name)
    fn_tokens = fn_norm.split("_")

    for pat in patterns:
        pat_norm   = normalize(pat)
        pat_tokens = pat_norm.split("_")
        n, m = len(fn_tokens), len(pat_tokens)

        # Точное совпадение нормализованных строк
        if pat_norm == fn_norm:
            return True, pat

        # Токены паттерна встречаются подряд в токенах папки
        for i in range(n - m + 1):
            if fn_tokens[i : i + m] == pat_tokens:
                return True, pat

        # Подстрока (для паттернов ≥ 6 символов — снижает риск ложных срабатываний)
        if len(pat_norm) >= 6 and pat_norm in fn_norm:
            return True, pat

    return False, ""


# ── Основная логика ───────────────────────────────────────────────────────

def clean_datasets(
    datasets_dir: Path,
    delete: bool,
    verbose: bool,
) -> None:
    if not datasets_dir.exists():
        print(f"Папка не найдена: {datasets_dir}")
        return

    # Собираем все папки датасетов (содержат X_train.npy или meta.json)
    all_dirs = sorted([
        d for d in datasets_dir.iterdir()
        if d.is_dir()
        and (
            (d / "X_train.npy").exists()
            or (d / "meta.json").exists()
        )
    ])

    if not all_dirs:
        print(f"Датасетов не найдено в {datasets_dir}")
        return

    keep_dirs   = []
    delete_dirs = []

    for ds_dir in all_dirs:
        matched, pat = matches_any(ds_dir.name, KEEP_PATTERNS)
        if matched:
            keep_dirs.append((ds_dir, pat))
        else:
            delete_dirs.append(ds_dir)

    # ── Вывод ─────────────────────────────────────────────────────────
    print(f"\nПапка:      {datasets_dir}")
    print(f"Всего:      {len(all_dirs)} датасетов")
    print(f"Оставить:   {len(keep_dirs)}")
    print(f"Удалить:    {len(delete_dirs)}")
    print(f"Режим:      {'РЕАЛЬНОЕ УДАЛЕНИЕ' if delete else 'DRY RUN (без удаления)'}")

    if verbose or not delete:
        if keep_dirs:
            print(f"\n{'='*55}")
            print(f"ОСТАВИТЬ ({len(keep_dirs)}):")
            for ds_dir, pat in keep_dirs:
                print(f"  ✓  {ds_dir.name:<60}  ← '{pat}'")

    if delete_dirs:
        print(f"\n{'='*55}")
        label = "УДАЛЯЕМ" if delete else "БУДЕТ УДАЛЕНО (dry-run)"
        print(f"{label} ({len(delete_dirs)}):")
        for ds_dir in delete_dirs:
            print(f"  ✗  {ds_dir.name}")

    # ── Удаление ──────────────────────────────────────────────────────
    if not delete:
        print(f"\n{'='*55}")
        print("Это DRY RUN — ничего не удалено.")
        print("Для реального удаления запусти с флагом --delete:")
        print(f"  python clean_datasets.py --delete --datasets {datasets_dir}")
        return

    if not delete_dirs:
        print("\nНечего удалять — все датасеты из списка сохранения.")
        return

    # Подтверждение
    print(f"\n{'='*55}")
    print(f"Будет УДАЛЕНО {len(delete_dirs)} папок.")
    answer = input("Подтвердить удаление? [yes/no]: ").strip().lower()
    if answer not in ("yes", "y", "да"):
        print("Отменено.")
        return

    deleted = 0
    errors  = 0
    for ds_dir in delete_dirs:
        try:
            shutil.rmtree(ds_dir)
            if verbose:
                print(f"  Удалена: {ds_dir.name}")
            deleted += 1
        except Exception as e:
            print(f"  ОШИБКА при удалении {ds_dir.name}: {e}")
            errors += 1

    print(f"\n{'='*55}")
    print(f"Готово!")
    print(f"  Удалено:        {deleted}")
    print(f"  Ошибок:         {errors}")
    print(f"  Осталось:       {len(keep_dirs)}")

    # Обновляем all_meta.json если он есть
    meta_path = datasets_dir / "all_meta.json"
    if meta_path.exists():
        import json
        try:
            with open(meta_path, encoding="utf-8") as f:
                all_meta = json.load(f)

            remaining_names = {d.name for d, _ in keep_dirs}
            all_meta_filtered = [
                m for m in all_meta
                if m.get("name", "") in remaining_names
            ]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(all_meta_filtered, f, indent=2, ensure_ascii=False)
            print(f"  all_meta.json обновлён: {len(all_meta_filtered)} записей")
        except Exception as e:
            print(f"  Не удалось обновить all_meta.json: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Удаление ненужных датасетов из папки datasets/",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    pa.add_argument(
        "--datasets", default="datasets",
        help="Папка с датасетами (default: datasets)",
    )
    pa.add_argument(
        "--delete", action="store_true",
        help="Реально удалить (без этого флага — только показать список)",
    )
    pa.add_argument(
        "--verbose", action="store_true",
        help="Показывать каждую удалённую папку",
    )
    pa.add_argument(
        "--no-confirm", action="store_true",
        help="Не спрашивать подтверждение перед удалением",
    )
    args = pa.parse_args()

    # Патч: если --no-confirm, подменяем input
    if args.no_confirm and args.delete:
        import builtins
        builtins.input = lambda _: "yes"

    clean_datasets(
        datasets_dir=Path(args.datasets),
        delete=args.delete,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()