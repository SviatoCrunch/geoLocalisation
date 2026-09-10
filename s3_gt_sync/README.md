# s3_gt_sync

Двонаправлена синхронізація UAV-кадрів GT між **S3** і **локальною директорією**,
на тих самих правилах іменування, що й `main_try/labels/rename_gt.py`.

Правила імен (per-folder):

| На S3                         | Роль                          | У локальному виході            |
|-------------------------------|-------------------------------|--------------------------------|
| `N.jpg`                       | кадр UAV                      | `N_lat_lon.jpg` (перейменований) |
| `N_lat,lon.jpg` (кома)        | скриншот-джерело координат     | **не копіюється, не видаляється** |
| `N_lat_lon.jpg` (підкреслення)| вже готовий GT-кадр            | копіюється як є                |

Ключова відмінність від ноутбука: **на S3 нічого не видаляється** — скриншот
`N_lat,lon` читається лише як джерело координат і лишається на місці.

## Крос-конекшн вимог

1. **Нічого не видаляти з S3** — обидва напрямки не викликають `delete_object`.
2. **Додавати лише відсутнє** — pull завантажує тільки ті імена, яких ще нема
   локально (той самий розмір = вважається наявним); push вантажить лише те,
   чого ще нема під префіксом.
3. **Обійти всі піддиректорії, крім зазначеної** — `--exclude <name> [...]`
   пропускає піддиректорію з такою назвою будь-де в дереві.
4. **Відправка локальне → S3** — окремий скрипт `push_to_s3.py`.
5. **Власний вихід ігнорується автоматично** — `pull` завжди пропускає
   піддиректорії `GT_flat` **і** `GT_flat_mask` (`ALWAYS_EXCLUDE_SUBDIRS`), навіть
   без `--exclude`, щоб синхронізація не затягувала назад те, що сама ж запушила.

## Встановлення / креденшели

Використовує стандартний ланцюжок AWS (env-змінні / `~/.aws` / IAM-роль).
Для S3-сумісного сховища (MinIO тощо) — `--endpoint-url` або `AWS_ENDPOINT_URL`.
Залежність `boto3` вже є в `pyproject.toml`.

## Pull: S3 → локальна директорія

```bash
# dry-run — показати, що завантажиться, нічого не чіпати
python -m s3_gt_sync.pull_from_s3 \
    --s3-uri s3://my-bucket/uav/Kramatorsk \
    --dest  "C:/data/GT_flat" \
    --exclude raw --dry-run

# реальна синхронізація
python -m s3_gt_sync.pull_from_s3 \
    --s3-uri s3://my-bucket/uav/Kramatorsk \
    --dest  "C:/data/GT_flat" \
    --exclude raw
```

Звіт наприкінці: `downloaded / skipped / unpaired / conflicts`.
- **unpaired** — кадри `N.jpg` без парного скриншота `N_lat,lon` (нема координат
  → неможливо назвати, пропущено; ключі виводяться списком).
- **conflicts** — два різні S3-ключі дають одне вихідне ім'я (напр. однаковий
  номер кадру в двох містах): лишається перший, решта пропускається з попередженням.

## Push: локальна директорія → S3

```bash
# dry-run
python -m s3_gt_sync.push_to_s3 \
    --src   "C:/data/GT_flat" \
    --s3-uri s3://my-bucket/uav/Kramatorsk_GT --dry-run

# реальне вивантаження
python -m s3_gt_sync.push_to_s3 \
    --src   "C:/data/GT_flat" \
    --s3-uri s3://my-bucket/uav/Kramatorsk_GT
```

Ключі формуються як `<prefix>/<шлях-відносно-src>` (POSIX-роздільники).
`--no-recursive` — тільки файли верхнього рівня. За замовчуванням наявні об'єкти
того ж розміру пропускаються.

## Маски: COCO `result.json` → побітові маски по класах → S3

Скрипт `make_masks.py` читає COCO-файл (`images` + `annotations` + `categories`)
і рендерить для кожної пари **(зображення, клас)** одну **бінарну** PNG-маску
(передній план = 255, фон = 0). Імена йдуть у парі з кадрами:

| Кадр (GT_flat)        | Маска (GT_flat_mask)               |
|-----------------------|------------------------------------|
| `…_lat_lon.jpg`       | `lat_lon__<class>.png` (по одній на клас у кадрі) |

Ім'я маски будується з пари координат `lat_lon` у COCO `file_name` (стабільний
GT-ключ). Експорт Label-Studio часто несе повний `..\..\label-studio\…` Windows-шлях
із хеш-префіксом (`0d0aeb21-102_…`) — він відкидається, лишається лише
`lat_lon__<class>.png`. Усі анотації одного класу в одному зображенні
об'єднуються (union) в одну маску.
Підтримка сегментації: полігони (PIL), нестиснене RLE (нативно), стиснене RLE
(через `pycocotools`, якщо встановлено), а без сегментації — заливається bbox.

```bash
# 1) лише зібрати маски (спершу dry-run)
uv run python -m s3_gt_sync.make_masks \
    --coco /home/ubuntu/work/gt_kup/result.json \
    --dest /home/ubuntu/work/gt_kup/GT_flat_mask --dry-run

# 2) зібрати + запушити на S3 (той самий non-destructive push)
uv run python -m s3_gt_sync.make_masks \
    --coco /home/ubuntu/work/gt_kup/result.json \
    --dest /home/ubuntu/work/gt_kup/GT_flat_mask \
    --push-s3-uri s3://geo-reference/gt/raw/kup/GT_flat_mask
```

Звіт: `written / skipped / empty / images / annotations / classes`
(`empty` = анотація без придатної сегментації чи bbox). `pull` тепер завжди
пропускає і `GT_flat`, і `GT_flat_mask` на S3 — маски не затягуються назад.
Програмно: `from s3_gt_sync import masks_from_coco; masks_from_coco(coco, dest)`.

## Приклад: Lyman (AWS S3)

S3: `s3://geo-reference/gt/raw/lyman` (готові кадри у підпапці `GT_flat/`).
Локально на сервері: `/home/ubuntu/work/gt_liman`. Креденшели — стандартний AWS-ланцюжок
(`~/.aws` / env / IAM-роль).

### Двонаправлений пайплайн (raw → локально → GT_flat)

`pull` бере з `lyman/` сирі пари `N.jpg` + `N_lat,lon.jpg` і перейменовує локально в
`N_lat_lon.jpg` (`--exclude GT_flat` — щоб не тягнути ще раз уже готові з підпапки);
`push` вивантажує перейменовані назад у `lyman/GT_flat/`. Нічого не видаляється,
вантажиться лише відсутнє.

```bash
# 1) dry-run — переконатися в плані, нічого не пише
uv run python -m s3_gt_sync.pull_from_s3 \
    --s3-uri s3://geo-reference/gt/raw/lyman \
    --dest /home/ubuntu/work/gt_liman \
    --exclude GT_flat --dry-run

uv run python -m s3_gt_sync.push_to_s3 \
    --src /home/ubuntu/work/gt_liman \
    --s3-uri s3://geo-reference/gt/raw/lyman/GT_flat --dry-run

# 2) реальний запуск (прибрати --dry-run)
uv run python -m s3_gt_sync.pull_from_s3 \
    --s3-uri s3://geo-reference/gt/raw/lyman \
    --dest /home/ubuntu/work/gt_liman \
    --exclude GT_flat

uv run python -m s3_gt_sync.push_to_s3 \
    --src /home/ubuntu/work/gt_liman \
    --s3-uri s3://geo-reference/gt/raw/lyman/GT_flat
```

### Лише стягнути готові кадри (без обробки сирих)

Якщо в `GT_flat/` уже лежать готові `N_lat_lon.jpg` і потрібно просто отримати їх
на сервер для тренування (push не потрібен):

```bash
uv run python -m s3_gt_sync.pull_from_s3 \
    --s3-uri s3://geo-reference/gt/raw/lyman/GT_flat \
    --dest /home/ubuntu/work/gt_liman --dry-run   # прибрати --dry-run для реального запуску
```

Підказки: `-v` — детальний per-file лог; звіт `pull` наприкінці —
`downloaded / skipped / unpaired / conflicts` (`unpaired` = кадр `N.jpg` без
скриншота з координатами). Не-AWS сховище (MinIO тощо) — додай `--endpoint-url https://...`.

## Програмний API

```python
from s3_gt_sync import pull, push, plan_pull

# сухий план без завантаження
targets, unpaired = plan_pull("s3://bkt/city", exclude={"raw"})

pull("s3://bkt/city", "C:/data/GT_flat", exclude={"raw"})
push("C:/data/GT_flat", "s3://bkt/city_GT")
```

## Тести

```bash
python -m pytest s3_gt_sync/tests -q
```

Тести використовують in-memory fake-S3 (без мережі, без `moto`).
