import streamlit as st
st.write("Начало файла: проверка вывода")
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# --- Функция маппинга путей (адаптируйте под свои данные) ---
def map_path(original_path: str) -> str:
    # Пример: если путь начинается с "data/train", заменяем на другой каталог
    if original_path.startswith("data/train"):
        return original_path.replace("data/train", "/kaggle/input/tain-krip/train")
    elif original_path.startswith("data/test_public"):
        return original_path.replace("data/test_public", "/kaggle/input/test-krip/test_public")
    return original_path

# --- Трансформация для инференса ---
def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# --- Предобработка изображения ---
def preprocess_image(image):
    image_np = np.array(image)
    transform = get_inference_transform()
    transformed = transform(image=image_np)
    tensor = transformed["image"]
    return tensor.unsqueeze(0)  # добавить batch dimension

# First, make sure you have timm installed:
# pip install timm

class ArtifactAwareEncoder(nn.Module):
    def __init__(self, num_classes=1000, embedding_size=512, device='cpu'):
        super().__init__()
        self.device = device

        # Use timm directly as in your training code
        import timm
        self.backbone = timm.create_model(
            'tf_efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.in_channels = features.shape[1]

        self.artifact_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )

        self.embedding_proj = nn.Linear(self.in_channels + 128, embedding_size)

    def forward(self, x):
        features = self.backbone(x)
        artifact = self.artifact_detector(features)
        pooled = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        combined = torch.cat([pooled, artifact], dim=1)
        embedding = F.normalize(self.embedding_proj(combined), p=2, dim=1)
        return embedding


@st.cache(allow_output_mutation=True)
def load_model():
    st.write("Функция load_model() вызвана")
    model = ArtifactAwareEncoder(num_classes=1000, embedding_size=512)
    st.write("Модель ArtifactAwareEncoder создана")
    state_dict = torch.load("model.pth", map_location=torch.device("cpu"))

    # Diagnostic prints
    #st.write("Model's state_dict keys:")
    model_keys = set(model.state_dict().keys())
    #st.write(model_keys)

    #st.write("Loaded state_dict keys:")
    loaded_keys = set(state_dict.keys())
    #st.write(loaded_keys)

    st.write("Missing keys:", model_keys - loaded_keys)
    st.write("Unexpected keys:", loaded_keys - model_keys)

    try:
        model.load_state_dict(state_dict, strict=False)  # Try with strict=False
        st.write("Loaded with strict=False")
    except Exception as e:
        st.write(f"Error even with strict=False: {e}")

    model.eval()
    return model

st.write("Перед загрузкой модели")
model = load_model()
st.write("Модель загружена успешно")

# --- Функция вычисления EER для пользовательского датасета ---
def calculate_eer(csv_data, model, device):
    transform = get_inference_transform()
    similarities = [] # Список для хранения значений косинусного сходства
    true_labels = []  # Список для хранения истинных меток (из CSV)

    st.write("Начинаем обработку пар изображений из CSV для расчета EER...") # Отладочное сообщение

    # Проходим по строкам CSV; ожидается, что в файле есть колонки: 'path1', 'path2', 'label'
    for index, row in csv_data.iterrows():
        try:
            img1 = Image.open(map_path(row['path1'])).convert("RGB")
            img2 = Image.open(map_path(row['path2'])).convert("RGB")
        except Exception as e:
            st.warning(f"Ошибка загрузки изображений на строке {index}: {e}")
            continue
        tensor1 = preprocess_image(img1)
        tensor2 = preprocess_image(img2)
        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)

        # Получаем эмбеддинги от модели
        with torch.no_grad():
            emb1 = model(tensor1)
            emb2 = model(tensor2)

        # Вычисляем косинусное сходство между эмбеддингами
        sim = F.cosine_similarity(emb1, emb2).item()

        st.write(f"Строка {index}: Косинусное сходство = {sim:.4f}, Метка = {row['label']}") # Отладочное сообщение

        similarities.append(sim) # Добавляем вычисленное сходство в список
        true_labels.append(int(row['label'])) # Добавляем метку в список

    st.write("Закончили вычисление сходства для всех пар. Начинаем расчет EER...") # Отладочное сообщение

    # Вычисляем ROC-кривую на основе накопленных сходств и меток
    fpr, tpr, thresholds = roc_curve(true_labels, similarities)
    idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[idx]

    st.write(f"EER успешно вычислен: {eer:.4f}") # Отладочное сообщение
    return eer, similarities, true_labels


def load_and_evaluate_user_data(uploaded_csv, model, device): # Эта функция не используется для EER, можно удалить, если не нужна для другого режима
    try:
        # Добавляем диагностический вывод
        st.write(f"Размер файла: {uploaded_csv.size} байт")

        # Пробуем разные параметры для чтения CSV
        try:
            # Сначала пробуем стандартный метод
            csv_data = pd.read_csv(uploaded_csv)
        except pd.errors.EmptyDataError:
            st.error("CSV файл пуст или неправильно отформатирован.")
            return None
        except Exception as e1:
            st.warning(f"Стандартное чтение не удалось: {e1}")
            # Пробуем с разными разделителями
            try:
                csv_data = pd.read_csv(uploaded_csv, sep=';')
                st.info("Файл прочитан с разделителем ';'")
            except Exception:
                try:
                    csv_data = pd.read_csv(uploaded_csv, sep='\t')
                    st.info("Файл прочитан с разделителем '\\t' (табуляция)")
                except Exception as e2:
                    st.error(f"Не удалось прочитать файл с разными разделителями: {e2}")
                    return None

        # Проверяем структуру CSV
        if csv_data.empty:
            st.error("CSV файл не содержит данных.")
            return None

        st.write("Loaded data:", csv_data.head())
        st.write("Columns in CSV:", list(csv_data.columns))

        # Проверяем наличие необходимых колонок
        required_columns = ['label', 'path'] #  ОШИБКА: здесь неверные колонки, должны быть path1 и path2 для EER
        missing_columns = [col for col in required_columns if col not in csv_data.columns]

        if missing_columns:
            st.error(f"В CSV отсутствуют обязательные колонки: {', '.join(missing_columns)}")
            return None

        # Продолжаем обработку как раньше
        transform = get_inference_transform()
        results = []

        # Group by label to process pairs
        for label, group in csv_data.groupby('label'):
            # Проверяем, достаточно ли строк в группе
            if len(group) < 2:
                st.warning(f"Недостаточно изображений для label {label}, пропускаем.")
                continue

            # Определяем query и gallery изображения
            if 'is_query' in group.columns and 'is_gallery' in group.columns:
                query_rows = group[group['is_query'] == True]
                gallery_rows = group[group['is_gallery'] == True]

                if len(query_rows) == 0 or len(gallery_rows) == 0:
                    st.warning(f"Для label {label} не найдены query или gallery изображения, используем первые две строки.")
                    query_row = group.iloc[0]
                    gallery_row = group.iloc[1]
                else:
                    query_row = query_rows.iloc[0]
                    gallery_row = gallery_rows.iloc[0]
            else:
                # Если колонок is_query/is_gallery нет, используем первые две строки
                query_row = group.iloc[0]
                gallery_row = group.iloc[1]

            # Загружаем и обрабатываем изображения
            try:
                query_img = Image.open(map_path(query_row['path'])).convert("RGB")
                gallery_img = Image.open(map_path(gallery_row['path'])).convert("RGB")
            except Exception as e:
                st.warning(f"Ошибка загрузки изображений для label {label}: {e}")
                continue

            # Трансформируем изображения
            query_tensor = preprocess_image(query_img).to(device)
            gallery_tensor = preprocess_image(gallery_img).to(device)

            # Извлекаем эмбеддинги
            with torch.no_grad():
                query_emb = model(query_tensor)
                gallery_emb = model(gallery_tensor)

            # Вычисляем сходство
            similarity = F.cosine_similarity(query_emb, gallery_emb).item()

            # Сохраняем результат
            results.append({
                'label': label,
                'query_path': query_row['path'],
                'gallery_path': gallery_row['path'],
                'similarity': similarity,
                'match': similarity >= 0.5  # Используем выбранный порог
            })

        # Если нет результатов, возвращаем None
        if not results:
            st.error("Не удалось обработать ни одну пару изображений.")
            return None

        # Создаем DataFrame с результатами
        results_df = pd.DataFrame(results)
        return results_df

    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
        import traceback
        st.error(f"Подробности ошибки: {traceback.format_exc()}")
        return None

# --- Streamlit интерфейс ---
st.title("Face Verification and EER Evaluation App")

# Выбор режима работы
mode = st.sidebar.selectbox("Выберите режим", ["Face Verification", "Evaluate EER on Custom Dataset", "Evaluate Custom Dataset"]) # Убрал просто "Evaluate EER" и уточнил названия

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if mode == "Face Verification":
    st.header("Face Verification")
    st.write("Загрузите два изображения для проверки, принадлежат ли они одному человеку.")
    uploaded_file1 = st.file_uploader("Загрузите первое изображение", type=["jpg", "jpeg", "png"], key="fv1")
    uploaded_file2 = st.file_uploader("Загрузите второе изображение", type=["jpg", "jpeg", "png"], key="fv2")
    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = Image.open(uploaded_file1).convert("RGB")
        image2 = Image.open(uploaded_file2).convert("RGB")
        st.image([image1, image2], caption=["Изображение 1", "Изображение 2"], width=200)
        tensor1 = preprocess_image(image1).to(device)
        tensor2 = preprocess_image(image2).to(device)
        with torch.no_grad():
            emb1 = model(tensor1)
            emb2 = model(tensor2)
        cosine_similarity = F.cosine_similarity(emb1, emb2).item()
        st.write(f"**Cosine Similarity:** {cosine_similarity:.4f}")
        # Простой порог – настройте его под вашу задачу
        threshold = 0.7
        if cosine_similarity >= threshold:
            st.success("Позитивный запрос: изображения, вероятно, принадлежат одному человеку.")
        else:
            st.error("Негативный запрос: изображения, вероятно, принадлежат разным людям или одно из них синтетическое.")

elif mode == "Evaluate EER on Custom Dataset": # Переименованный режим для ясности
    st.header("Evaluate EER on Custom Dataset") # Заголовок режима соответствует названию в selectbox
    st.write("Загрузите CSV-файл с пользовательским датасетом. Файл должен содержать колонки: `path1`, `path2` и `label` (1 — если изображения принадлежат одному человеку, 0 — если разным).")
    uploaded_csv = st.file_uploader("Загрузите CSV-файл", type=["csv"], key="eer")
    if uploaded_csv is not None:
        try:
            csv_data = pd.read_csv(uploaded_csv)
            st.write("Загруженные данные:", csv_data.head())
            if st.button("Вычислить EER"):
                with st.spinner("Вычисление EER..."):
                    eer, sims, labels = calculate_eer(csv_data, model, device) # Используем функцию calculate_eer для вычисления EER
                st.write(f"**EER:** {eer:.4f}")
                # Построение ROC-кривой
                import matplotlib.pyplot as plt
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(labels, sims)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="ROC")
                ax.plot([0,1], [1,0], linestyle="--", label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

elif mode == "Evaluate Custom Dataset": # Режим для оценки пар, как был раньше
    st.header("Evaluate on Custom Dataset")
    st.write("Загрузите CSV-файл с парами лиц. Файл должен содержать колонки: `label`, `path`, и опционально `is_query` и `is_gallery`.")

    # Добавляем опции для пользователя
    st.write("Опции загрузки CSV:")
    csv_separator = st.selectbox("Разделитель CSV", [",", ";", "\\t"], index=0)

    uploaded_csv = st.file_uploader("Загрузите CSV-файл", type=["csv", "txt"], key="custom")

    if uploaded_csv is not None:
        try:
            # Предварительный просмотр содержимого файла
            uploaded_csv.seek(0)
            file_bytes = uploaded_csv.read(min(uploaded_csv.size, 5000))
            uploaded_csv.seek(0)  # Сбрасываем указатель

            try:
                file_content = file_bytes.decode('utf-8')
                st.subheader("Предварительный просмотр содержимого файла:")
                st.text(file_content[:1000] + ("..." if len(file_content) > 1000 else ""))
            except UnicodeDecodeError:
                st.warning("Не удалось декодировать содержимое файла как текст UTF-8")

            # Пробуем прочитать с выбранным разделителем
            sep_map = {"\\t": "\t"}  # Маппинг для спецсимволов
            separator = sep_map.get(csv_separator, csv_separator)

            try:
                # Предварительно просматриваем файл, используя выбранный разделитель
                preview_data = pd.read_csv(uploaded_csv, sep=separator, nrows=5)
                st.write("Предварительный просмотр данных:", preview_data)
                uploaded_csv.seek(0)  # Сбрасываем указатель
            except Exception as e:
                st.warning(f"Не удалось прочитать предварительные данные: {e}")

            if st.button("Оценить пары"):
                with st.spinner("Обработка пар изображений..."):
                    results_df = load_and_evaluate_user_data(uploaded_csv, model, device)

                if results_df is not None:
                    st.write("Результаты:")
                    st.dataframe(results_df)

                    # Вычисляем статистику
                    avg_similarity = results_df['similarity'].mean()
                    match_rate = results_df['match'].mean() * 100

                    st.write(f"Среднее сходство: {avg_similarity:.4f}")
                    st.write(f"Процент совпадений: {match_rate:.2f}%")

                    # Визуализируем распределение
                    fig, ax = plt.subplots()
                    ax.hist(results_df['similarity'], bins=20)
                    ax.set_xlabel("Оценка сходства")
                    ax.set_ylabel("Количество")
                    ax.set_title("Распределение оценок сходства")
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Ошибка обработки файла: {e}")
            import traceback
            st.error(f"Подробности ошибки: {traceback.format_exc()}")
