"""
loc_bureau.py

Процедурная реализация алгоритмов автоматизации основного бизнес-процесса
специализированного бюро локализации видеоигр.

Реализованные алгоритмы:
    А1. Разбор файла строковых ресурсов и экранирование плейсхолдеров.
    А2. Оценка проекта по памяти переводов (нечёткое сравнение, смета).
    А3. Распределение заданий между переводчиками (жадный алгоритм с кучей).
    А4. Автоматическая проверка качества перевода (QA-проверки).

Обозначения в оценках сложности:
    n - число строк (сегментов), m - число записей памяти переводов,
    L - средняя длина строки в символах, k - число исполнителей,
    g - число терминов глоссария, p - число целевых языков.
"""

import heapq
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Общие константы
# ---------------------------------------------------------------------------

# Служебные элементы, которые нельзя переводить: {0}, {name}, %s, %d, %1$s,
# теги разметки <b>, </b>, <color=#ff0000>, а также литерал \n.
PLACEHOLDER_RE = re.compile(
    r"\{[A-Za-z0-9_]+\}"
    r"|%(?:\d+\$)?[sdf]"
    r"|</?[A-Za-z][^<>]*>"
    r"|\\n"
)
TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
MASK = "\u27e6\u27e7"  # символ-заменитель плейсхолдера

# Полосы совпадения: (нижняя граница сходства, вес при расчёте оплаты, название)
MATCH_BANDS = [
    (1.00, 0.1, "100%"),
    (0.95, 0.3, "95-99%"),
    (0.85, 0.5, "85-94%"),
    (0.75, 0.7, "75-84%"),
]


# ---------------------------------------------------------------------------
# А1. Разбор файла строковых ресурсов
# ---------------------------------------------------------------------------

def count_words(masked_text):
    """Число слов в строке без учёта плейсхолдеров. O(L)."""
    return len(re.findall(r"[^\W_]+", masked_text))


def parse_resource(text):
    """
    Разбирает файл формата КЛЮЧ=Текст.
    Возвращает (список сегментов, список ошибок).
    Сложность: O(N), где N - общее число символов файла (n*L).
    """
    segments, errors, seen = [], [], set()
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append((line_no, "отсутствует разделитель '='"))
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            errors.append((line_no, "пустой ключ"))
            continue
        if not value:
            errors.append((line_no, f"пустое значение у ключа {key}"))
            continue
        if key in seen:
            errors.append((line_no, f"повторяющийся ключ {key}"))
            continue
        seen.add(key)
        masked = PLACEHOLDER_RE.sub(MASK, value)
        segments.append({
            "key": key,
            "text": value,
            "masked": masked,
            "placeholders": PLACEHOLDER_RE.findall(value),
            "words": count_words(masked),
        })
    return segments, errors


# ---------------------------------------------------------------------------
# А2. Оценка проекта по памяти переводов
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    """Расстояние Левенштейна, две строки матрицы. Время O(|a|*|b|), память O(min)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a, b):
    """Сходство строк в диапазоне [0; 1]."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


def find_match(masked, tm_index, tm_list, threshold=0.75):
    """
    Поиск лучшего совпадения в памяти переводов.
    Точное совпадение ищется через хеш-таблицу за O(L); для нечёткого
    применяется отсев по длине, что отбрасывает заведомо непохожие записи.
    Худший случай: O(m*L^2).
    """
    if masked in tm_index:
        return 1.0, tm_index[masked]
    best_score, best_target = 0.0, None
    for source, target in tm_list:
        longest = max(len(source), len(masked), 1)
        if abs(len(source) - len(masked)) / longest > 1.0 - threshold:
            continue
        score = similarity(masked, source)
        if score > best_score:
            best_score, best_target = score, target
    if best_score >= threshold:
        return best_score, best_target
    return best_score, None


def match_weight(score):
    """Возвращает (вес, название полосы) по значению сходства. O(1)."""
    for low, weight, name in MATCH_BANDS:
        if score >= low:
            return weight, name
    return 1.0, "новое"


def estimate_project(segments, tm_list, rate_per_word):
    """
    Классифицирует сегменты по памяти переводов и считает смету.
    В каждый сегмент добавляются поля match, tm_target, weighted_words.
    Сложность: O(n*m*L^2) в худшем случае, O(n*L) при точных совпадениях.
    """
    tm_index = dict(tm_list)
    by_band = Counter()
    total_words, weighted_total = 0, 0.0
    for seg in segments:
        score, target = find_match(seg["masked"], tm_index, tm_list)
        if target is None:
            weight, band = 1.0, "новое"
        else:
            weight, band = match_weight(score)
        seg["match"], seg["tm_target"] = band, target
        seg["weighted_words"] = seg["words"] * weight
        by_band[band] += seg["words"]
        total_words += seg["words"]
        weighted_total += seg["weighted_words"]
    return {
        "total_words": total_words,
        "weighted_words": round(weighted_total, 2),
        "by_band": dict(by_band),
        "cost": round(weighted_total * rate_per_word, 2),
    }


# ---------------------------------------------------------------------------
# А3. Распределение заданий между переводчиками
# ---------------------------------------------------------------------------

def split_into_batches(segments, batch_limit):
    """Делит сегменты на пакеты по взвешенному объёму. O(n)."""
    batches, current, volume = [], [], 0.0
    for seg in segments:
        if current and volume + seg["weighted_words"] > batch_limit:
            batches.append({"id": len(batches) + 1, "keys": current, "volume": volume})
            current, volume = [], 0.0
        current.append(seg["key"])
        volume += seg["weighted_words"]
    if current:
        batches.append({"id": len(batches) + 1, "keys": current, "volume": volume})
    return batches


def assign_batches(batches, languages, translators, deadline_days):
    """
    Жадное распределение пакетов (правило LPT): крупные пакеты раздаются
    первыми, каждый - исполнителю, который освободится раньше всех (куча).
    Сложность: O(n + b log b + p*(k + b log k)), b <= n - число пакетов.
    """
    ordered = sorted(batches, key=lambda b: -b["volume"])
    assignments, problems = [], []
    for lang in languages:
        staff = [t for t in translators if lang in t["langs"]]
        if not staff:
            problems.append(f"нет исполнителей для языка {lang}")
            continue
        loads = [0.0] * len(staff)
        last = 0.0  # максимальное время готовности по языку
        heap = [(0.0, i) for i in range(len(staff))]
        heapq.heapify(heap)
        for batch in ordered:
            _, i = heapq.heappop(heap)
            loads[i] += batch["volume"]
            finish = loads[i] / staff[i]["speed"]
            heapq.heappush(heap, (finish, i))
            last = max(last, finish)
            assignments.append({
                "batch": batch["id"], "lang": lang, "translator": staff[i]["name"],
                "volume": round(batch["volume"], 2), "finish_day": round(finish, 2),
            })
        if last > deadline_days:
            problems.append(f"язык {lang}: срок {last:.1f} дн. превышает {deadline_days} дн.")
    return assignments, problems


# ---------------------------------------------------------------------------
# А4. Автоматическая проверка качества перевода
# ---------------------------------------------------------------------------

def _tag_name(tag):
    return re.split(r"[\s=/>]", tag.lstrip("</"), maxsplit=1)[0]


def tags_balanced(text):
    """Проверка вложенности парных тегов стеком. O(L)."""
    stack = []
    for tag in TAG_RE.findall(text):
        if tag.endswith("/>"):
            continue
        name = _tag_name(tag)
        if tag.startswith("</"):
            if not stack or stack.pop() != name:
                return False
        else:
            stack.append(name)
    return not stack


def validate_translation(seg, target, glossary, expansion=1.5):
    """Проверки одного сегмента. Возвращает список (код, сообщение). O(L*g)."""
    if not target or not target.strip():
        return [("MISSING", "перевод отсутствует")]
    defects = []
    if Counter(PLACEHOLDER_RE.findall(target)) != Counter(seg["placeholders"]):
        defects.append(("PLACEHOLDER", f"набор плейсхолдеров отличается от {seg['placeholders']}"))
    if not tags_balanced(target):
        defects.append(("TAGS", "нарушена вложенность тегов"))
    limit = seg.get("max_len") or int(len(seg["text"]) * expansion) + 5
    if len(target) > limit:
        defects.append(("LENGTH", f"длина {len(target)} превышает лимит {limit}"))
    for src_term, tgt_term in glossary.items():
        if re.search(r"\b" + re.escape(src_term) + r"\b", seg["text"], re.IGNORECASE):
            stem = tgt_term.lower()[:max(3, len(tgt_term) - 2)]  # грубый учёт падежей
            if stem not in target.lower():
                defects.append(("TERM", f"термин «{src_term}» должен переводиться как «{tgt_term}»"))
    return defects


def validate_project(segments, translations, glossary):
    """Проверка всех сегментов. O(n*L*g)."""
    report, bad_segments = [], set()
    for seg in segments:
        for code, message in validate_translation(seg, translations.get(seg["key"], ""), glossary):
            report.append({"key": seg["key"], "code": code, "message": message})
            bad_segments.add(seg["key"])
    total = len(segments)
    rate = round(100.0 * len(bad_segments) / total, 1) if total else 0.0
    return report, {"checked": total, "with_defects": len(bad_segments), "defect_rate_%": rate}


# ---------------------------------------------------------------------------
# Тестовые данные и демонстрация
# ---------------------------------------------------------------------------

SAMPLE_RESOURCE = r"""
# Фрагмент строковых ресурсов игры Aether Quest
MENU_START=Start game
MENU_EXIT=Exit
HUD_HEALTH=Health: {0}/{1}
HUD_MANA=Mana cost: {0}
QUEST_NEW=New quest: <b>{name}</b> added to the journal
ITEM_POTION=Healing Potion restores %d points of health
ITEM_POTION_BIG=Healing Potion restores %d points of health
DLG_01=Welcome, traveler!\nThe Crystal Tower awaits you.
DLG_02=The Crystal Tower is guarded by ancient spirits.
SPELL_FIRE=Fireball deals {0} damage to all enemies in range
BAD LINE WITHOUT SEPARATOR
MENU_START=Duplicate key
"""

SAMPLE_TM = [
    ("Start game", "Начать игру"),
    ("Exit", "Выход"),
    (f"Health: {MASK}/{MASK}", "Здоровье: {0}/{1}"),
    (f"Welcome, traveler!{MASK}The Ancient Tower awaits you.", "Добро пожаловать, путник!\\nДревняя башня ждёт тебя."),
]

SAMPLE_TRANSLATORS = [
    {"name": "Иванова А.", "langs": {"ru"}, "speed": 6.0},
    {"name": "Петров Б.", "langs": {"ru", "de"}, "speed": 5.0},
    {"name": "Schmidt K.", "langs": {"de"}, "speed": 7.0},
]

SAMPLE_GLOSSARY = {"Crystal Tower": "Хрустальная башня", "Mana": "мана"}

SAMPLE_TRANSLATIONS_RU = {
    "MENU_START": "Начать игру",
    "MENU_EXIT": "Выход",
    "HUD_HEALTH": "Здоровье: {0}",  # потерян плейсхолдер
    "HUD_MANA": "Затраты маны: {0}",
    "QUEST_NEW": "Новое задание: <b>{name}</b> добавлено в журнал",
    "ITEM_POTION": "Зелье лечения восстанавливает %d здоровья",
    "ITEM_POTION_BIG": "",  # нет перевода
    "DLG_01": "Добро пожаловать, путник!\\nБашня из хрусталя ждёт тебя.",  # нарушен термин
    "DLG_02": "Хрустальная башня охраняется древними духами.",
    "SPELL_FIRE": "<i>Огненный шар наносит {0} урона всем врагам в радиусе действия",  # тег не закрыт
}


def run_tests():
    """Автоматические тесты алгоритмов."""
    segs, errs = parse_resource(SAMPLE_RESOURCE)
    assert len(segs) == 10 and len(errs) == 2
    assert segs[2]["placeholders"] == ["{0}", "{1}"] and segs[2]["words"] == 1
    assert levenshtein("kitten", "sitting") == 3
    assert similarity("abc", "abc") == 1.0
    assert tags_balanced("<b>a</b>") and not tags_balanced("<b>a<i></b></i>")
    rep = estimate_project(segs, SAMPLE_TM, 2.0)
    assert segs[0]["match"] == "100%" and rep["weighted_words"] <= rep["total_words"]
    batches = split_into_batches(segs, 10.0)
    assert abs(sum(b["volume"] for b in batches) - rep["weighted_words"]) < 1e-6
    plan, _ = assign_batches(batches, ["ru"], SAMPLE_TRANSLATORS, 10)
    assert {a["batch"] for a in plan} == {b["id"] for b in batches}
    defects, stat = validate_project(segs, SAMPLE_TRANSLATIONS_RU, SAMPLE_GLOSSARY)
    assert {d["code"] for d in defects} == {"PLACEHOLDER", "MISSING", "TERM", "TAGS"}
    assert stat["with_defects"] == 4
    print("Все тесты пройдены.")


def main():
    print("=== А1. Разбор ресурсов ===")
    segments, errors = parse_resource(SAMPLE_RESOURCE)
    print(f"Загружено сегментов: {len(segments)}")
    for line_no, message in errors:
        print(f"  Ошибка в строке {line_no}: {message}")

    print("\n=== А2. Оценка по памяти переводов ===")
    report = estimate_project(segments, SAMPLE_TM, rate_per_word=2.0)
    print(f"Всего слов: {report['total_words']}, к оплате (взвешенно): {report['weighted_words']}")
    print(f"Распределение по совпадениям: {report['by_band']}")
    print(f"Стоимость перевода на один язык: {report['cost']} у.е.")

    print("\n=== А3. Распределение заданий ===")
    batches = split_into_batches(segments, batch_limit=10.0)
    print(f"Сформировано пакетов: {len(batches)}")
    plan, problems = assign_batches(batches, ["ru", "de", "fr"], SAMPLE_TRANSLATORS, deadline_days=3)
    for a in plan:
        print(f"  [{a['lang']}] пакет {a['batch']}: {a['translator']:<11} "
              f"объём {a['volume']:>5}, готово на {a['finish_day']} дн.")
    for problem in problems:
        print("  Предупреждение:", problem)

    print("\n=== А4. Проверка качества (ru) ===")
    defects, stat = validate_project(segments, SAMPLE_TRANSLATIONS_RU, SAMPLE_GLOSSARY)
    for d in defects:
        print(f"  {d['key']:<16} {d['code']:<12} {d['message']}")
    print(f"Проверено: {stat['checked']}, с дефектами: {stat['with_defects']} "
          f"({stat['defect_rate_%']} %)")


if __name__ == "__main__":
    run_tests()
    print()
    main()