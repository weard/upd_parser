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
MANUAL_DIR = os.path.join(OUTPUT_DIR, "_разобрать_вручную")
EXCEL_PATH = os.path.join(OUTPUT_DIR, "реестр_упд.xlsx")
DB_PATH = os.path.join(OUTPUT_DIR, "processed.db")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))   # секунды между проходами
OCR_DPI = int(os.getenv("OCR_DPI", "200"))              # dpi рендера для OCR
OCR_LANG = os.getenv("OCR_LANG", "rus")                 # язык Tesseract
HEADER_ZONE = float(os.getenv("HEADER_ZONE", "0.45"))   # доля страницы сверху для поиска заголовка

# ------------------------------ Логирование ---------------------------------

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MANUAL_DIR, exist_ok=True)

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
    conn = sqlite3.connect(DB_PATH)
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
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _ocr(bw_img, psm=6):
    return pytesseract.image_to_string(
        Image.fromarray(bw_img), lang=OCR_LANG, config=f"--psm {psm}")


def prepare_page(page):
    """
    Возвращает dict:
      text      — текстовый слой PDF (может быть пустым),
      bw        — бинаризованное изображение страницы в правильной ориентации,
      rotated   — PIL-изображение в правильной ориентации (для проверки печати).
    bw/rotated = None, если текстовый слой достаточен и OCR не нужен.
    """
    text = page.get_text("text").strip()
    if len(text) >= 40:
        return {"text": text, "bw": None, "rotated": None}
    img = _render_pil(page, OCR_DPI)
    angle = _detect_rotation(img)
    if angle:
        img = img.rotate(-angle, expand=True)
    return {"text": None, "bw": _binarize(img), "rotated": img}


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

NUM_PATTERNS = [
    r"счет-фактура\s*№\s*([0-9A-Za-zА-Яа-яЁё][\w\-/]*)",
    r"документ\s*№\s*([0-9A-Za-zА-Яа-яЁё][\w\-/]*)",
    r"№\s*([0-9A-Za-zА-Яа-яЁё][\w\-/]*)\s+от\s+\d",   # № рядом с датой
]
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


def parse_date(top_lower):
    m = re.search(r"от\s*(\d{2})[./](\d{2})[./](\d{2,4})", top_lower)
    if m:
        d, mo, y = m.groups()
        y = "20" + y if len(y) == 2 else y
        return f"{y}-{mo}-{d}"
    m = re.search(rf"от\s*(\d{{1,2}})\s*({MONTHS})\s*(\d{{4}})", top_lower)
    if m:
        d, mon, y = m.groups()
        return f"{y}-{MONTH_NUM[mon]:02d}-{int(d):02d}"
    return "Без_даты"


def parse_sum(full_lower, bw):
    """Сумма «Всего к оплате»: сначала regex по тексту, затем по полосе
    итогов с удалёнными линиями таблицы (в т.ч. склеенные цифры)."""
    raw = _first_match(SUM_PATTERNS, full_lower)
    if raw:
        return raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    if bw is not None:
        h = bw.shape[0]
        band = bw[int(h * 0.40):int(h * 0.65), :]
        inv = 255 - band
        lines = cv2.add(
            cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))),
            cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))))
        band_text = normalize_ws(_ocr(255 - cv2.subtract(inv, lines))).lower()
        raw = _first_match(SUM_PATTERNS, band_text)
        if raw:
            return raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        # «5729830» -> «57298.30»: последняя длинная группа цифр после метки
        m = re.search(r"всего\s+к\s+оплате((?:[^\d]{0,15}[\d\s]+){1,6})", band_text)
        if m:
            groups = re.findall(r"\d{4,}", m.group(1).replace(" ", ""))
            if groups:
                g = groups[-1]
                return f"{g[:-2]}.{g[-2:]}"
    return "Проверить вручную"


def is_upd_header(top_text):
    return "универсальный передаточный" in normalize_ws(top_text).lower()


def parse_upd(top_text, full_text, bw):
    """Извлекает поля УПД из текста первой страницы документа."""
    top_norm = normalize_ws(top_text)
    top_lower = top_norm.lower()
    full_lower = normalize_ws(full_text).lower()

    doc_num = _first_match(NUM_PATTERNS, top_norm) or "Неизвестный"
    doc_date = parse_date(top_lower)
    inn = _first_match(INN_PATTERNS, full_lower) or "Не найден"
    total = parse_sum(full_lower, bw)

    return {"num": doc_num, "date": doc_date, "inn": inn, "sum": total}


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

# --------------------------- Проверка печати --------------------------------

def has_stamp(pil_img):
    """Сине-фиолетовые оттенки печати в нижней части страницы."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img[int(h * 0.5):h, 0:w], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
    return cv2.countNonZero(mask) > 500

# ----------------------------- Excel-реестр ---------------------------------

def append_excel(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(EXCEL_PATH):
        try:
            df = pd.concat([pd.read_excel(EXCEL_PATH), df], ignore_index=True)
        except Exception as e:
            log.error("Не удалось прочитать реестр (%s) — пишу новый", e)
    tmp = EXCEL_PATH + ".tmp.xlsx"
    df.to_excel(tmp, index=False)
    os.replace(tmp, EXCEL_PATH)   # атомарная замена — реестр не побьётся

# --------------------------- Обработка пакета -------------------------------

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
        doc = fitz.open(path)
        reader = PdfReader(path)

        writer, misc_writer = None, PdfWriter()
        misc_count = 0
        first_img = None          # PIL первой страницы текущего УПД (для печати)
        current_out = None
        has_attachment = False

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
        for i in range(len(doc)):
            page = doc.load_page(i)
            prep = prepare_page(page)
            top_text = page_text(prep, header_only=True)

            if is_upd_header(top_text):
                data = parse_upd(top_text, page_text(prep), prep["bw"])
                if writer is not None and rows and same_document(
                        rows[-1]["Номер УПД"], data["num"]):
                    # страница-продолжение УПД с повторным заголовком формы
                    writer.add_page(reader.pages[i])
                    last_img = prep["rotated"]
                    continue
                close_document(last_img)
                writer, has_attachment = PdfWriter(), False
                current_out = unique_path(os.path.join(
                    OUTPUT_DIR,
                    f"УПД_№{sanitize(data['num'])}_от_{data['date']}.pdf"))
                first_img = prep["rotated"]
                notes = [n for n in (date_warning(data["date"]),
                                     "проверить сумму" if data["sum"] == "Проверить вручную" else "",
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
                writer.add_page(reader.pages[i])
            elif writer is not None:
                # приложение (доверенность и т.п.) — по ключевому слову,
                # а не любая страница без заголовка
                if ATTACHMENT_RE.search(normalize_ws(top_text).lower()):
                    has_attachment = True
                writer.add_page(reader.pages[i])
            else:
                # страницы до первого УПД — в отдельный файл, не теряем
                misc_writer.add_page(reader.pages[i])
                misc_count += 1
            last_img = prep["rotated"]

        close_document(last_img)

        if misc_count:
            misc_path = unique_path(os.path.join(
                MANUAL_DIR, f"без_заголовка_{sanitize(fname)}"))
            with open(misc_path, "wb") as f:
                misc_writer.write(f)
            log.warning("  %d стр. без заголовка УПД -> %s",
                        misc_count, os.path.basename(misc_path))

        if rows:
            append_excel(rows)
        doc.close()
        mark_done(conn, file_hash, fname, "ok")
        log.info("Готово: %s (документов: %d)", fname, len(rows))

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
