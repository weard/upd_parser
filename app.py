"""
Демон разбора пакетных сканов УПД на Synology NAS (Intel, Docker).

Логика:
  1. Каждые POLL_INTERVAL секунд сканирует /app/input (polling вместо watchdog —
     надёжно работает на любых сетевых шарах и не зависит от inotify).
  2. Новые PDF ждёт, пока размер перестанет расти (сканер дописал файл).
  3. Для каждой страницы сначала пытается достать текстовый слой PDF.
     Если текста нет — OCR через Tesseract с автоповоротом страницы (OSD),
     бинаризацией и распознаванием только верхней зоны для поиска заголовка.
  4. Пакет нарезается на отдельные УПД в /app/output, ведётся реестр Excel.
  5. Обработанные файлы запоминаются по SHA-256 в SQLite — повторная
     обработка и дубликаты исключены, оригиналы в input не изменяются.
"""

import hashlib
import io
import logging
import os
import re
import sqlite3
import time
from logging.handlers import RotatingFileHandler

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pypdf import PdfReader, PdfWriter

# ----------------------------- Конфигурация ---------------------------------

INPUT_DIR = os.getenv("INPUT_DIR", "/app/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DB_PATH = os.getenv("DB_PATH", os.path.join(LOG_DIR, "processed.db"))

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))   # секунды между проходами
OCR_DPI = int(os.getenv("OCR_DPI", "200"))              # dpi рендера для OCR
OCR_LANG = os.getenv("OCR_LANG", "rus")                 # язык Tesseract
HEADER_ZONE = float(os.getenv("HEADER_ZONE", "0.45"))   # доля страницы сверху для поиска заголовка

# ------------------------------ Логирование ---------------------------------

os.makedirs(LOG_DIR, exist_ok=True)

log = logging.getLogger("upd-parser")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
_fh = RotatingFileHandler(os.path.join(LOG_DIR, "processor.log"),
                          maxBytes=2_000_000, backupCount=3, encoding="utf-8")
_fh.setFormatter(_fmt)
log.addHandler(_sh)
log.addHandler(_fh)

# --------------------------- Журнал обработанных ----------------------------

def db_init():
    # timeout + busy_timeout: при кратковременной блокировке файла БД
    # (антивирус Synology, бэкап, второй процесс) ждём, а не падаем
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed ("
        "  hash TEXT PRIMARY KEY,"
        "  filename TEXT,"
        "  status TEXT,"           # ok / error
        "  processed_at TEXT"
        ")"
    )
    return conn


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already_done(conn, file_hash):
    row = conn.execute("SELECT status FROM processed WHERE hash = ?",
                       (file_hash,)).fetchone()
    return row is not None  # ok и error — оба не трогаем повторно


def mark_done(conn, file_hash, filename, status):
    conn.execute(
        "INSERT OR REPLACE INTO processed VALUES (?, ?, ?, datetime('now'))",
        (file_hash, filename, status))
    conn.commit()

# --------------------------- Ожидание файла ---------------------------------

def wait_for_file_stable(path, checks=3, interval=1.5, timeout=180):
    """Файл считается готовым, когда размер не меняется `checks` замеров подряд."""
    last_size, stable_count, waited = -1, 0, 0.0
    while waited < timeout:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size > 0 and size == last_size:
            stable_count += 1
            if stable_count >= checks:
                return True
        else:
            stable_count = 0
            last_size = size
        time.sleep(interval)
        waited += interval
    return False

# ------------------------- Подготовка изображений ---------------------------

def _render_pil(page, dpi):
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _detect_rotation(pil_img):
    """Угол, на который нужно повернуть изображение (по данным OSD Tesseract)."""
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate: (\d+)", osd)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def _binarize(pil_img):
    """Бинаризация по Отцу + статистика яркости (для детекта чёрных/белых листов)."""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Доля ячеек сетки 12x12, залитых чёрным > 90%: ловит полу-чёрные листы
    # (сбой дуплекса: верх чёрный, низ белый), которые по средней яркости
    # неотличимы от нормальной страницы.
    gs = 12
    h, w = gray.shape
    ch, cw = h // gs, w // gs
    crop = gray[:ch * gs, :cw * gs]
    cells = (crop.reshape(gs, ch, gs, cw)
             .transpose(0, 2, 1, 3).reshape(gs * gs, ch * cw))
    black_cells = float(np.mean((cells < 128).mean(axis=1) > 0.9))
    stats = {"mean": float(gray.mean()),
             "dark": float(np.count_nonzero(gray < 128) / gray.size),
             "black_cells": black_cells}
    return bw, stats


def _ocr(bw_img, psm=6):
    return pytesseract.image_to_string(
        Image.fromarray(bw_img), lang=OCR_LANG, config=f"--psm {psm}")


def prepare_page(page):
    """
    Возвращает dict:
      text      — текстовый слой PDF (может быть пустым),
      bw        — бинаризованное изображение страницы в правильной ориентации,
      rotated   — PIL-изображение в правильной ориентации (для проверки печати),
      angle     — угол поворота страницы по OSD (0/90/180/270, по часовой).
    bw/rotated = None, если текстовый слой достаточен и OCR не нужен.
    """
    text = page.get_text("text").strip()
    if len(text) >= 40:
        return {"text": text, "bw": None, "rotated": None, "angle": 0}
    img = _render_pil(page, OCR_DPI)
    angle = _detect_rotation(img)
    if angle:
        img = img.rotate(-angle, expand=True)
    bw, stats = _binarize(img)
    return {"text": None, "bw": bw, "rotated": img, "stats": stats,
            "angle": angle}


def page_text(prep, header_only=False):
    """Текст страницы (или её верхней зоны) — из текстового слоя или OCR."""
    if prep["text"] is not None:
        return prep["text"]
    bw = prep["bw"]
    if header_only:
        bw = bw[:int(bw.shape[0] * HEADER_ZONE), :]
    return _ocr(bw)

# ------------------------------ Парсинг УПД ---------------------------------

MONTHS = ("января|февраля|марта|апреля|мая|июня|июля|августа|"
          "сентября|октября|ноября|декабря")
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}

# Номер извлекаем только в паре «№ X от <дата>» — это отсекает номера
# постановлений (№ 1137) и прочие «№» из шапки формы. Подчёркивания —
# это линии бланка между полями, OCR их рисует между № и датой.
NUM_DATE_RE = re.compile(
    rf"№[\s_]*([0-9A-Za-zА-Яа-яЁё][\w\-/]{{0,24}}?)[\s_]*от[\s_]*"
    rf"(\d{{2}}[./]\d{{2}}[./]\d{{2,4}}|\d{{1,2}}\s*(?:{MONTHS})\s*\d{{4}})",
    re.IGNORECASE)
INN_PATTERNS = [
    r"инн.{0,15}продавца[^\d]{0,20}(\d{10,12})",
    r"инн[^\d]{0,20}(\d{10,12})",
]
# Берём сумму только из строки «Всего к оплате»: «итого» в табличной части —
# это промежуточные подытоги/НДС, подставлять их в реестр опасно.
SUM_PATTERNS = [
    r"всего\s+к\s+оплате[^\d]{0,25}([\d\s\u00a0]+[.,]\d{2})",
]
ATTACHMENT_RE = re.compile(r"доверенност")   # признак приложения, а не тела УПД


def normalize_ws(text):
    return " ".join(text.split())


def _first_match(patterns, text):
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return None


def _date_to_iso(ds):
    """'28.09.2026' или '28 сентября 2026' -> '2026-09-28' (None, если не дата)."""
    m = re.match(r"(\d{2})[./](\d{2})[./](\d{2,4})", ds)
    if m:
        d, mo, y = m.groups()
        y = "20" + y if len(y) == 2 else y
        return f"{y}-{mo}-{d}"
    m = re.match(rf"(\d{{1,2}})\s*({MONTHS})\s*(\d{{4}})", ds.lower())
    if m:
        d, mon, y = m.groups()
        return f"{y}-{MONTH_NUM[mon]:02d}-{int(d):02d}"
    return None


def parse_number_and_date(top_norm):
    """Номер и дата УПД — как единая пара «№ X от <дата>». При нескольких
    парах выбирается та, перед которой стоит «счет-фактура»/«документ»,
    а пары возле слов «постановление» (номер закона) отбрасываются."""
    pairs = []
    for m in NUM_DATE_RE.finditer(top_norm):
        iso = _date_to_iso(m.group(2))
        if not iso:
            continue
        ctx = top_norm[max(0, m.start() - 60):m.start()].lower()
        score = 0
        if "передаточн" in ctx:
            score += 120   # строка «документ об отгрузке: УПД № X от ...»
        if ("счет-фактура" in ctx or "счетфактура" in ctx
                or "счет фактура" in ctx):
            score += 100
        if "документ" in ctx:
            score += 50
        if "платежно" in ctx or "платёжн" in ctx:
            score -= 100   # платёжное поручение — НЕ номер УПД
        if "постановлен" in ctx or "положени" in ctx:
            score -= 100   # номер постановления — НЕ номер УПД
        pairs.append((score, m.start(), m.group(1).strip("_"), iso))
    if pairs:
        pairs.sort(key=lambda p: (-p[0], p[1]))
        return pairs[0][2].upper(), pairs[0][3]
    # fallback: номер без даты
    for p in (r"счет-фактура\s*№\s*([0-9A-Za-zА-Яа-яЁё][\w\-/]*)",
              r"документ\s*№\s*([0-9A-Za-zА-Яа-яЁё][\w\-/]*)"):
        m = re.search(p, top_norm, re.IGNORECASE)
        if m:
            return m.group(1).upper(), "Без_даты"
    return "Неизвестный", "Без_даты"


def make_filename(num, date_iso):
    """Формат: И030201-2.3.26.pdf"""
    if date_iso != "Без_даты":
        y, mo, d = date_iso.split("-")
        ds = f"{int(d)}.{int(mo)}.{y[2:]}"
    else:
        ds = "без_даты"
    return f"{sanitize(num)}-{ds}.pdf"


AMOUNT_RE = re.compile(r"\d[\d\s\u00a0]*[.,]\d{2}")


def _sum_from_text(text_lower):
    """Сумма после метки «Всего к оплате». В строке итогов УПД несколько
    чисел (кол-во, без НДС, НДС, с НДС) — итог с НДС идёт ПОСЛЕДНИМ."""
    m = re.search(r"всего\s+к\s+оплате(.{0,100})", text_lower)
    if m:
        amounts = AMOUNT_RE.findall(m.group(1))
        if amounts:
            return (amounts[-1].replace(" ", "").replace("\u00a0", "")
                    .replace(",", "."))
        # «5729830» -> «57298.30»: склеенные цифры после метки
        groups = re.findall(r"\d{4,}", m.group(1).replace(" ", ""))
        if groups:
            g = groups[-1]
            return f"{g[:-2]}.{g[-2:]}"
    return None


def _sum_from_band(bw):
    """Сумма из полосы итогов с удалёнными линиями таблицы."""
    h = bw.shape[0]
    band = bw[int(h * 0.40):int(h * 0.65), :]
    inv = 255 - band
    lines = cv2.add(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))))
    return _sum_from_text(normalize_ws(_ocr(255 - cv2.subtract(inv, lines))).lower())


def parse_sum(full_lower, bw):
    """Сумма «Всего к оплате» из двух независимых источников (полный текст
    страницы и полоса итогов с удалёнными линиями).
    Возвращает (значение, подтверждено):
      - оба источника согласны  -> (значение, True)
      - источник один           -> (значение, False) — пометить на проверку
      - источники расходятся    -> ("Проверить вручную (X / Y)", False)
    """
    candidates = []
    s_text = _sum_from_text(full_lower)
    if s_text:
        candidates.append(s_text)
    if bw is not None:
        s_band = _sum_from_band(bw)
        if s_band:
            candidates.append(s_band)
    if not candidates:
        return "Проверить вручную", False
    if len(set(candidates)) == 1:
        return candidates[0], len(candidates) == 2
    return ("Проверить вручную (" + " / ".join(dict.fromkeys(candidates)) + ")",
            False)


def is_upd_header(top_text):
    """Заголовок УПД: основная фраза или связка реквизитов формы
    (устойчиво к шуму OCR в отдельных словах)."""
    t = normalize_ws(top_text).lower()
    return ("передаточн" in t
            or ("счет-фактура" in t and "продавец" in t))


def sheet_number(top_text):
    """Номер листа УПД («Лист 2», «Лист 3»...). На первой странице — 1 или
    отсутствует; на страницах-продолжениях — > 1."""
    m = re.search(r"лист\s*[:№]?\s*(\d{1,2})", normalize_ws(top_text).lower())
    return int(m.group(1)) if m else None


def parse_upd(top_text, full_text, bw):
    """Извлекает поля УПД из текста первой страницы документа."""
    top_norm = normalize_ws(top_text)
    full_lower = normalize_ws(full_text).lower()

    doc_num, doc_date = parse_number_and_date(top_norm)
    inn = _first_match(INN_PATTERNS, full_lower) or "Не найден"
    total, sum_confirmed = parse_sum(full_lower, bw)

    return {"num": doc_num, "date": doc_date, "inn": inn,
            "sum": total, "sum_confirmed": sum_confirmed}


def same_document(prev_num, new_num):
    """Страница-продолжение УПД повторяет заголовок формы. Считаем её частью
    открытого документа, если цифровые части номеров совпадают (или одна —
    суффикс другой, т.к. OCR теряет/добавляет первый символ)."""
    a, b = re.sub(r"\D", "", prev_num), re.sub(r"\D", "", new_num)
    if not a or not b:
        return False
    return a == b or a.endswith(b) or b.endswith(a)


def date_warning(date_iso):
    """Проверка правдоподобия даты (OCR путает цифры года/месяца)."""
    from datetime import date as _date
    try:
        y, m, d = map(int, date_iso.split("-"))
        _date(y, m, d)
    except ValueError:
        return "дата не распознана — проверить"
    this_year = _date.today().year
    if y > this_year or y < this_year - 5:
        return "подозрительный год — проверить"
    return ""


OCR_LETTER_DIGITS = "0179"   # цифры, которые OCR чаще всего ставит вместо буквы (0→О, 7/1→И, 9→g/У)


def postprocess_packet(rows, outputs, packet_dir):
    """Пакетная нормализация после обработки всех страниц.

    1. Номера: если большинство номеров пакета имеют вид «буква + N цифр»,
       у остальных восстанавливается префикс («092505» -> «И092505»,
       «7092215» -> «И092215»).
    2. Годы: неправдоподобный год (будущее, «2926», древность) заменяется
       доминантным годом пакета с пометкой в «Примечании».
    Файлы переименовываются, если номер или дата изменились.
    """
    nums = [r["Номер УПД"] for r in rows]
    known = [n for n in nums if n and n != "Неизвестный"]

    # --- номера ---
    dom_letter, dom_len = None, None
    letters = [n[0] for n in known if n[0].isalpha()]
    if len(letters) >= 2:
        dom_letter = max(set(letters), key=letters.count)
        lens = [len(n) - 1 for n in known
                if n.startswith(dom_letter) and n[1:].isdigit()]
        if lens:
            dom_len = max(set(lens), key=lens.count)

    # --- годы ---
    from datetime import date as _date
    this_year = _date.today().year
    plausible = [int(r["Дата УПД"][:4]) for r in rows
                 if r["Дата УПД"] != "Без_даты"
                 and this_year - 5 <= int(r["Дата УПД"][:4]) <= this_year]
    dom_year = max(set(plausible), key=plausible.count) if plausible else this_year

    for r, o in zip(rows, outputs):
        # номер
        n = r["Номер УПД"]
        new = None
        if dom_letter and dom_len:
            if n.isdigit() and len(n) == dom_len:
                new = dom_letter + n                    # потерянная буква
            elif (len(n) == dom_len + 1 and n[0] in OCR_LETTER_DIGITS
                    and n[1:].isdigit()):
                new = dom_letter + n[1:]                # буква заменена цифрой
        if new and new != n:
            log.info("  нормализация номера: %s -> %s", n, new)
            r["Номер УПД"] = new

        # год
        d = r["Дата УПД"]
        if d != "Без_даты":
            y = int(d[:4])
            if not (this_year - 5 <= y <= this_year):
                r["Дата УПД"] = f"{dom_year}{d[4:]}"
                prev_note = r["Примечание"].replace(
                    "подозрительный год — проверить", "").strip("; ")
                r["Примечание"] = "; ".join(filter(None, [
                    prev_note, f"год исправлен автоматически (был {y})"]))
                log.info("  нормализация года: %s -> %s (№%s)",
                         d, r["Дата УПД"], r["Номер УПД"])

        # переименование при изменениях
        want = os.path.join(packet_dir,
                            make_filename(r["Номер УПД"], r["Дата УПД"]))
        if os.path.basename(want) != os.path.basename(o["path"]):
            new_path = unique_path(want)
            try:
                os.rename(o["path"], new_path)
                o["path"] = new_path
            except OSError as e:
                log.error("  не удалось переименовать %s: %s", o["path"], e)


def sanitize(name):
    return re.sub(r'[/\\:*?"<>|]', "_", name)


def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

# --------------------------- Пустые страницы --------------------------------

def page_state(prep, page):
    """'content' / 'blank' (белый лист) / 'black' (чёрный лист — сбой сканера:
    оборот в дуплексе, неоткрытая крышка и т.п.)."""
    if prep["text"] is not None:
        if len(prep["text"]) < 5 and not page.get_images():
            return "blank"
        return "content"
    st = prep["stats"]
    # целиком чёрная ИЛИ пятая часть+ ячеек залита чёрным (полу-чёрный лист)
    if st["mean"] < 80 or st["black_cells"] > 0.2:
        return "black"
    if st["dark"] < 0.001:   # < 0.1% тёмных пикселей — белый лист
        return "blank"
    return "content"

# --------------------------- Проверка печати --------------------------------

def has_stamp(pil_img):
    """Сине-фиолетовые оттенки печати в нижней части страницы."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img[int(h * 0.5):h, 0:w], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
    return cv2.countNonZero(mask) > 500

# ----------------------------- Excel-реестр ---------------------------------

def append_excel(rows, excel_path):
    df = pd.DataFrame(rows)
    if os.path.exists(excel_path):
        try:
            df = pd.concat([pd.read_excel(excel_path), df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["Исходный файл", "Номер УПД", "Дата УПД"], keep="last")
        except Exception as e:
            log.error("Не удалось прочитать реестр (%s) — пишу новый", e)
    tmp = excel_path + ".tmp.xlsx"
    df.to_excel(tmp, index=False)
    os.replace(tmp, excel_path)   # атомарная замена — реестр не побьётся

# --------------------------- Обработка пакета -------------------------------

def _add_page(writer, reader, idx, angle=0):
    """Добавляет страницу в выходной PDF, ставя ей правильную ориентацию
    (угол от OSD). Без этого повёрнутые сканы остаются «лежать на боку»."""
    p = reader.pages[idx]
    if angle:
        p.rotate(angle)  # pypdf крутит по часовой — как и угол OSD
    writer.add_page(p)


def process_pdf(path, conn):
    fname = os.path.basename(path)
    file_hash = sha256_of_file(path)

    if already_done(conn, file_hash):
        log.info("Пропуск (уже обработан): %s", fname)
        return
    if not wait_for_file_stable(path):
        log.warning("Файл ещё пишется, повторим позже: %s", fname)
        return

    log.info("Обработка: %s", fname)
    rows = []
    try:
        # папка пакета: output/<имя_пакета>/{сканы, реестр.xlsx, разобрать/}
        stem = sanitize(os.path.splitext(fname)[0])
        packet_dir = os.path.join(OUTPUT_DIR, stem)
        manual_dir = os.path.join(packet_dir, "разобрать")
        os.makedirs(packet_dir, exist_ok=True)

        doc = fitz.open(path)
        reader = PdfReader(path)

        writer, misc_writer = None, PdfWriter()
        misc_count = 0
        first_img = None          # PIL первой страницы текущего УПД (для печати)
        current_out = None
        has_attachment = False
        outputs = []              # пути выходных файлов для пост-нормализации

        def close_document(last_img):
            nonlocal writer, first_img, current_out, has_attachment
            if writer is None:
                return
            with open(current_out, "wb") as f:
                writer.write(f)
            stamp = (first_img is not None and has_stamp(first_img)) or \
                    (last_img is not None and has_stamp(last_img))
            rows[-1]["Наличие печати"] = "Да" if stamp else "Нет"
            rows[-1]["Наличие доверенности/приложений"] = \
                "Да" if has_attachment else "Нет"
            log.info("  -> %s", os.path.basename(current_out))

        last_img = None
        blank_count = 0
        black_count = 0
        for i in range(len(doc)):
            page = doc.load_page(i)
            prep = prepare_page(page)

            # пустые/чёрные листы пропускаем, никуда не включаем
            state = page_state(prep, page)
            if state != "content":
                if state == "black":
                    black_count += 1
                    log.warning("  стр. %d: чёрная (сбой сканера?), пропущена", i + 1)
                else:
                    blank_count += 1
                    log.info("  стр. %d: пустая, пропущена", i + 1)
                continue

            top_text = page_text(prep, header_only=True)

            if is_upd_header(top_text):
                data = parse_upd(top_text, page_text(prep), prep["bw"])
                sheet_no = sheet_number(top_text)
                is_continuation = (
                    writer is not None and rows and (
                        (sheet_no is not None and sheet_no > 1) or
                        same_document(rows[-1]["Номер УПД"], data["num"])))
                if is_continuation:
                    # страница-продолжение УПД (лист 2+ или тот же номер)
                    _add_page(writer, reader, i, prep["angle"])
                    last_img = prep["rotated"]
                    continue
                close_document(last_img)
                writer, has_attachment = PdfWriter(), False
                current_out = unique_path(os.path.join(
                    packet_dir, make_filename(data["num"], data["date"])))
                outputs.append({"path": current_out})
                first_img = prep["rotated"]
                notes = [n for n in (
                    date_warning(data["date"]),
                    "" if data["sum_confirmed"]
                    else ("проверить сумму" if data["sum"].startswith("Проверить")
                          else "сумма не подтверждена — проверить"),
                    "ИНН не найден" if data["inn"] == "Не найден" else "") if n]
                rows.append({
                    "Исходный файл": fname,
                    "Номер УПД": data["num"],
                    "Дата УПД": data["date"],
                    "ИНН Поставщика": data["inn"],
                    "Сумма": data["sum"],
                    "Наличие печати": "Нет",
                    "Наличие доверенности/приложений": "Нет",
                    "Примечание": "; ".join(notes),
                })
                _add_page(writer, reader, i, prep["angle"])
            elif writer is not None:
                # приложение (доверенность и т.п.) — по ключевому слову,
                # а не любая страница без заголовка
                if ATTACHMENT_RE.search(normalize_ws(top_text).lower()):
                    has_attachment = True
                _add_page(writer, reader, i, prep["angle"])
            else:
                # страницы до первого УПД — в отдельный файл, не теряем
                _add_page(misc_writer, reader, i, prep["angle"])
                misc_count += 1
            last_img = prep["rotated"]

        close_document(last_img)

        if rows:
            postprocess_packet(rows, outputs, packet_dir)

        if misc_count:
            os.makedirs(manual_dir, exist_ok=True)
            misc_path = unique_path(os.path.join(
                manual_dir, f"без_заголовка_{sanitize(fname)}"))
            with open(misc_path, "wb") as f:
                misc_writer.write(f)
            log.warning("  %d стр. без заголовка УПД -> %s/разобрать/%s",
                        misc_count, stem, os.path.basename(misc_path))

        if rows:
            append_excel(rows, os.path.join(packet_dir, "реестр.xlsx"))
        doc.close()
        mark_done(conn, file_hash, fname, "ok")
        log.info("Готово: %s (документов: %d, пустых: %d, чёрных: %d)",
                 fname, len(rows), blank_count, black_count)
        if black_count:
            log.warning("В пакете %s чёрных страниц: %d — проверьте сканер "
                        "(дуплекс/крышка/калибровка)", fname, black_count)

    except Exception:
        log.exception("Ошибка при обработке %s", fname)
        mark_done(conn, file_hash, fname, "error")

# ------------------------------ Главный цикл --------------------------------

def scan_input(conn):
    try:
        names = sorted(os.listdir(INPUT_DIR))
    except OSError as e:
        log.error("Папка %s недоступна: %s", INPUT_DIR, e)
        return
    for name in names:
        # отсекаем скрытые и служебные файлы (AppleDouble ._, @eaDir и т.п.)
        if name.startswith(".") or not name.lower().endswith(".pdf"):
            continue
        process_pdf(os.path.join(INPUT_DIR, name), conn)


def main():
    log.info("Демон запущен. Вход: %s | Выход: %s | Интервал: %d с",
             INPUT_DIR, OUTPUT_DIR, POLL_INTERVAL)
    conn = db_init()
    while True:
        scan_input(conn)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
