# ADHD Assist — Setup Guide

אפליקציה מלאה: frontend (`static/index.html`) + backend אמיתי ב-Flask (`app.py`)
שמריץ ניתוח אמיתי — מעלים **הקלטת EEG (למשל 2 שניות = חלון אחד, או הקלטה
ארוכה יותר = כמה חלונות), 19 ערוצים**, השרת מחלץ ממנה בדיוק את אותם 1,501
מאפיינים שהקוד המקורי שלכן מחשב (`features.py` הוא שכפול מדויק של
`CODE_MRMR_19_ELEC.py`), ומריץ עליהם את מודל ה-KNN המאומן.

## מבנה התיקיות
```
neurofocus_app/
├── app.py                   ← שרת Flask (מריץ את הניתוח)
├── features.py               ← חילוץ מאפיינים מ-EEG גולמי — שכפול מדויק
│                               של הפרה-פרוססינג האמיתי, לשימוש השרת בזמן אמת
├── CODE_MRMR_19_ELEC.py       ← הקוד המקורי שלכן (רק KNN) — הפרה-פרוססינג
│                               עצמו רץ כאן, מקומית, פעם אחת
├── code_knn_only.py           ← הקוד המקורי שלכן (רק KNN) — האימון הסופי
│                               ושמירת המודל, רץ אחרי הקוד הקודם
├── requirements.txt
├── static/
│   └── index.html             ← ה-frontend
└── model/                     ← כאן שמים את 5 הקבצים שהאימון הסופי ייצר
    ├── knn_model.joblib
    ├── selected_indices.npy
    ├── feat_mean.npy
    ├── feat_std.npy
    └── meta.json
```

## שלב 1 — פרה-פרוססינג (מקומית, איפה שה-121 קבצי mat נמצאים)

ב-`CODE_MRMR_19_ELEC.py`:
1. עורכים את `save_folder` ואת הנתיבים בתוך `folders` (בראש הקובץ) כך
   שיצביעו לתיקיות ה-mat האמיתיות אצלך (`ADHD_IDS = 1-61`,
   `CONTROL_IDS = 62-121` — זה כבר מוגדר נכון בקובץ)
2. משנים `RUN_PREPROCESSING = True` (כרגע `False`)
3. מריצים:
   ```
   pip install numpy scipy PyWavelets antropy fooof scikit-learn joblib openpyxl
   python CODE_MRMR_19_ELEC.py
   ```
   זה ייצור `features_subj{i}.npy` + `labels_subj{i}.npy` לכל 121 הנבדקים
   בתוך `save_folder`.

⚠️ **שימו לב:** בגרסת antropy הציבורית (`pip install antropy`) אין פונקציה
בשם `fuzzy_entropy` — אם `ant.fuzzy_entropy` לא קיים אצלכן, זה עלול לזרוק
שגיאה בקוד המקורי. ב-`features.py` (הגרסה ששרת ה-web משתמש בה) כבר טיפלתי
בזה כך שהתכונה הזו פשוט מקבלת ערך 0 אם היא לא קיימת — ייתכן שכדאי לעדכן
גם את `CODE_MRMR_19_ELEC.py` באותו אופן לפני ההרצה המקומית, אם תיתקלו בשגיאה.

## שלב 2 — אימון סופי ושמירת המודל

ב-`code_knn_only.py`:
1. עורכים את `FOLDER` כך שיצביע לאותה `save_folder` משלב 1
2. מריצים:
   ```
   python code_knn_only.py
   ```
   זה טוען את כל ה-`features_subj{i}.npy`, מאמן KNN סופי על 240 המאפיינים
   המדורגים הכי גבוה, ושומר 5 קבצים לתוך `model_export/`.
3. מעתיקים את 5 הקבצים מתוך `model_export/` לתוך `neurofocus_app/model/`.

## שלב 3 — בדיקה מקומית

```
pip install -r requirements.txt
python app.py
```
פותחים `http://localhost:5000`, גוררים הקלטת EEG (למשל 2 שניות, 19 ערוצים),
ובודקים שמתקבלת תוצאה. חלון בודד (2 שניות) לוקח כמה שניות לניתוח, כי
חילוץ המאפיינים (בעיקר FOOOF וקוהרנטיות) איטי יחסית.

## שלב 4 — GitHub

מעלים את **כל** תיקיית `neurofocus_app` (כולל `model/` עם 5 הקבצים) לריפו חדש.

## שלב 5 — Render (Web Service, לא Static Site!)

1. render.com → **New +** → **Web Service**
2. מחברים לריפו
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app --timeout 120`
   (הזמן המוארך חשוב — חילוץ המאפיינים לוקח יותר מ-30 שניות הדיפולטיות
   של gunicorn עבור הקלטות ארוכות)
5. מקבלים כתובת ציבורית — זו הכתובת ששולחים הלאה

## שלב 6 — גוגל סייט

Insert → Embed → By URL → מדביקים את הכתובת מ-Render.
