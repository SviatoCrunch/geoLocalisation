# Відео-геолокалізація: motion-keyframes → kup-прод fine → frame-voting (детальна памятка)

Локалізація всього **відео** (chunk) до клітинки прямої ~1-км² шахівниці: відбір опорних кадрів
продовою SegProp-рецептурою, per-кадровий **kup-вузловий** fine-рерank (byte-identical до прод-сервісу),
і **агрегація голосуванням кадрів** за одну клітинку. Метрика успіху — **top-10 у радіусі 1 км**.

## Вхід / артефакти
- Відео: `mapforml_public_Detection.csv` (зона `0-17-34-138` = Куп'янськ-вузол, 99 відео), GT = col 27/28 (verified).
- Карта: пряма шахівниця `kup` = 221 неперекривна ~1-км клітинка (`kup_prodaction.h5`); packed-пірамід-стор
  `kup_newformat` (221 cell × 25 позицій × 8 рівнів × 32×32 × d1024, fp16, ~86 GiB).
- Прод-еталон (kup-вузол): Hetzner `62.238.88.50` `/root/work/vision_project` (гілка `server/e2c-live-snapshot`),
  env `service/kup.env`: `SIAM_RERANK_VERIFY=torch`, `SIAM_RERANK_FORMAT=packed`, `K=100`, `REPROJ=2.0`,
  `MPP_HYPS=256`, `IRLS=1`, `closed_form`. **md5 reranker/matcher/magsacpp == `single_frame_locate/rerank`** (звірено).

## Алгоритм (повний конвеєр)

**1. Відбір опорних кадрів** — `pipeline1_interframe/build_label_set.py` (`place_in_motion`):
декод → DINOv2 quality-gate (`good`) → моушн good→good (Farneback, медіана) → розрив сегментів на `max_gap` →
`place_in_motion`: кадри рівномірно в кумулятивному русі, `budget = frames_per_key × median_step`;
`--target-k 20` підбирає budget (фактична кількість **8–20**, адаптивно до руху). → **K опорних кадрів**.
(Обгортка: `clip_search/sample_motion.py` — CSV-driven, тегує кадри GT, видає `clips_manifest.json`.)

**2. Кодування кадру** — `map_extract` / e2c backend: DINOv2 ViT-g/14 (p448, 32×32, d1024, `proj_seed 0`, sky-filter) → токени.

**3. Coarse (e2c SuperVLAD)** — скоринг проти 221 клітинки → **top-100 кандидатів на кадр**.

**4. Fine rerank (kup-прод, torch GPU MAGSAC++)** — `single_frame_locate/rerank`:
для кожної клітинки, кожної (позиція `i`∈25, рівень `L`∈8): mutual-NN (fp16, batched) → MAGSAC++ (256 гіп.) →
`score(i,L) = inliers / N_query` ∈ [0,1]. Згортки:
- піраміда: `pyr(i) = Σ_L score(i,L)` (сума 8 рівнів);
- **скор клітинки (на кадр): `cell_score = mean_i pyr(i)`** (середнє по 25 пірамідах);
- точка = позиція `argmax_i pyr(i)` (її lat/lon), рівень = `argmax_L score(bi,L)`.

**5. Ранжування в кадрі** — клітинки сортуються за `cell_score` → `rank` (1 = найкраща).

**6. Агрегація по кадрах (нова схема)** — `clip_search/vote_aggregate.py`:
- юніон = ∪ top-100 усіх кадрів;
- **reciprocal-rank голоси** (масштаб-інваріантні): `V(cell) = Σ_f 1/rank_f(cell)`;
- **гео-NMS**: будуючи top-K, пропускаємо клітинку, якщо вона ближче `nms_m=900 м` до вже взятої
  (злиття near-GT сусідів, бо ~1-км сітка ділить сигнал між 2–3 клітинками);
- вихід: top-K клітинок (lat/lon = найкраща позиція).

**Успіх:** top-10 @ 1 км — хоч одна з 10 клітинок ≤1 км від GT.

## Чому так (ключові знахідки)
- **Покриття вирішене, вузьке місце — ранжування.** GT досяжний у кандидатах у **100%** відео (@1км),
  але top-1 витягує лише ~44%. Різниця — не у формулі, а в контрасті скорів.
- **Скори низькоконтрастні й різномасштабні** (0.0–4.7 між відео/кадрами), доміновані **burst/hub**-клітинками.
  → `sum`/`mean`/`pooled-mean` закопують GT на rank 20–136. **Reciprocal-rank** (ранг, не величина) стійкіший.
- **Near-GT сигнал ділиться** між сусідніми прямими клітинками → **гео-NMS** консолідує зони, щоб top-K
  покривав K різних місць, а не дублі.
- `max`-агрегація небезпечна (один burst-кадр перемагає); поріг per-frame `N` впливає per-video.

## Результати (ФІНАЛ, 99/99 відео) — rr + гео-NMS 900 м
| | @250 | @500 | @1000 |
|---|---|---|---|
| CEILING (GT досяжний) | 96/99 | 99/99 | **99/99 (100%)** |
| top-1 | 20 | 28 | 37 (37%) |
| top-5 | 31 | 43 | 62 (63%) |
| **top-10** | 38 | 53 | **76/99 (77%)** |

Порівняння агрегацій (top-10 @1 км): `rr+nms900=76` > `rr(no-nms)=75` > `votes=72` > `sum=mean=66` > `max=0`.
Масштаб/час: keyframes/відео mean 17.6 (1–26, total 1743); rerank total 10.8 год; сирих скорів 58 351 000.
Висновок: GT досяжний у 100%; практичний top-10 @1 км = 77%; вузьке місце — ранжування (ранг+гео-NMS виграє).

## Як запустити
```bash
# батч (сервер AWS T4): добір + kup-fine + сирий дамп per-video, resume-safe
bash ~/work/kup_batch.sh 9999            # STORE=/opt/dlami/nvme/kup (локальний NVMe)
# офлайн: вивести per-frame cell-скори, потім схему
python3 ~/work/derive_cellscores.py                      # tile_scores_full.csv -> cell_scores_per_frame.csv
python3 clip_search/vote_aggregate.py <root> rr 900      # top-{1,5,10}@{250,500,1000}
```
Артефакти per-video: `tile_scores_full.csv` (кадр×клітинка×позиція×рівень), `cell_scores_per_frame.csv`
(кадр×клітинка, mean-по-пірамідах, з рейтингом), `locate.json`; майстер `kup_batch/timing.csv`.

## Параметри
`K_coarse=100`, позицій=25, рівнів=8, MAGSAC `hyps=256`/`irls=1`/`closed_form`/`reproj=2.0`, fp16-матч,
голос=`1/rank`, `nms_m=900 м`, метрика `top-10@1000 м`. Лівери: `nms_m`, rank-ваги, поріг 1.5 км, більше keyframes.
