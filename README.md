# Streamlit приложение для верификации лиц с ArtifactAwareEncoder в рамках соревнования Kryptonite ML Challenge

[![Ссылка на Streamlit App](https://img.shields.io/badge/Streamlit%20App-Live-success?style=for-the-badge&logo=streamlit)](https://app-face-verification-app-58ey6ulk8m4uswbrygzwha.streamlit.app/)

**[Запустить онлайн приложение Streamlit](https://app-face-verification-app-58ey6ulk8m4uswbrygzwha.streamlit.app/)**

---

## 🚀 Описание

Это Streamlit приложение позволяет:

* **Верифицировать лица:** загрузите два изображения и получите оценку схожести, показывающую, насколько вероятно, что это один и тот же человек.
* **Оценить EER на своих данных:** загрузите CSV файл с парами изображений и метками для оценки Equal Error Rate (EER) модели `ArtifactAwareEncoder`.

Приложение использует модель `ArtifactAwareEncoder`, основанную на EfficientNet B3.

---

## 🧠 О модели

Модель `ArtifactAwareEncoder`:

* **Бэкбон:** EfficientNet B3 (ImageNet pre-trained).
* **Лоссы:** ArcFace + Contrastive Loss.
* **Метрика:** EER.

**Результаты**:

* Public test EER: **0.1545**.
* Локальный тест: **EER 0.1017**.

---

## 💻 Использование

**Запуск локально:**

1. Клонируйте репозиторий.
2. Установите библиотеки: `pip install -r requirements.txt`.
3. Запустите приложение: `streamlit run stream.py`.

**Онлайн использование:**

1. Перейдите по ссылке: [https://app-face-verification-app-58ey6ulk8m4uswbrygzwha.streamlit.app/](https://app-face-verification-app-58ey6ulk8m4uswbrygzwha.streamlit.app/)
2. **Режим "Face Verification":** загрузите два изображения для проверки.
3. **Режим "Evaluate EER on Custom Dataset":** загрузите CSV файл (`path1`, `path2`, `label`) для оценки EER.

---

## ⚠️ Важно

* Файл `model.pth` содержит веса модели.
* Проверьте пути к изображениям в CSV файле для оценки EER.
* Производительность зависит от качества изображений.

#Computer vision
#Deep Learning
