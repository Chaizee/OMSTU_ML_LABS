import pickle
import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import xgboost as xgb

# Настройка многостраничности через боковую панель
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Перейти на страницу:",
    ["Разработчик", "О наборе данных", "Визуализация данных", "Инференс моделей"],
)

# --- СТРАНИЦА 1: РАЗРАБОТЧИК ---
if page == "Разработчик":
    st.title("👨‍💻 Информация о разработчике")
    st.subheader(
        "РГР по дисциплине «Машинное обучение и большие данные»"
    )  #
    st.write("**Тема:** Разработка Web-приложения для инференса моделей ML")

    col1, col2 = st.columns([1, 2])
    with col1:
        # Укажите путь к вашему фото или заглушке
        st.image(
            "https://placeholder.com",
            caption="Фото студента",
            use_container_width=True,
        )  #
    with col2:
        st.write("**ФИО:** Иванов Иван Иванович")  # Укажите свои данные
        st.write("**Группа:** МО-221")  # Укажите свою группу

# --- СТРАНИЦА 2: О НАБОРЕ ДАННЫХ ---
elif page == "О наборе данных":
    st.title("📊 Описание набора данных")
    st.write(
        "### Предметная область: [Укажите вашу тему, например: Предсказание цен на квартиры]"
    )  #
    st.write(
        "Здесь приводится подробное описание датасета, его происхождения и бизнес-задачи."
    )

    st.write("### Описание признаков (Данные):")  #
    st.markdown(
        """
    * **Feature_1** — [Описание и единицы измерения]
    * **Feature_2** — [Описание и единицы измерения]
    * **Target** — Целевой признак ([Единица измерения])
    """
    )

    st.write("### Особенности предобработки и EDA:")  #
    st.info(
        "Удалены пропуски, обработаны выбросы методом IQR, категориальные признаки закодированы с помощью OneHotEncoder."
    )

# --- СТРАНИЦА 3: ВИЗУАЛИЗАЦИЯ ЗАВИСИМОСТЕЙ ---
elif page == "Визуализация данных":
    st.title("📈 Визуализация зависимостей")
    st.write(
        "Минимум 4 различных вида визуализации данных (Matplotlib/Seaborn):"
    )  #

    # Генерация демонстрационных данных (замените на загрузку вашего реального датасета)
    np.random.seed(42)
    df_demo = pd.DataFrame(
        {
            "Feature_1": np.random.randn(100),
            "Feature_2": np.random.rand(100) * 100,
            "Target": np.random.randn(100) * 10 + 50,
            "Category": np.random.choice(["Тип А", "Тип Б"], size=100),
        }
    )

    # Визуализация 1: Матрица корреляции
    st.write("#### 1. Тепловая карта корреляции (Correlation Heatmap)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_demo.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # Визуализация 2: Распределение целевой переменной
    st.write("#### 2. Распределение целевого признака (Histogram)")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.histplot(df_demo["Target"], kde=True, color="skyblue", ax=ax2)
    st.pyplot(fig2)

    # Визуализация 3: Диаграмма рассеяния
    st.write("#### 3. Диаграмма рассеяния (Scatter Plot)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df_demo, x="Feature_1", y="Target", hue="Category", ax=ax3)
    st.pyplot(fig3)

    # Визуализация 4: Ящик с усами
    st.write("#### 4. Диаграмма размаха (Boxplot)")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=df_demo, x="Category", y="Target", ax=ax4)
    st.pyplot(fig4)

# --- СТРАНИЦА 4: ИНФЕРЕНС (ПРЕДСКАЗАНИЕ) ---
elif page == "Инференс моделей":
    st.title("🤖 Получение предсказаний моделей ML")  #

    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель для инференса:",
        [
            "ML1: Классическая модель",
            "ML2: Ансамбль (XGBoost)",
            "ML3: Advanced Бустинг (CatBoost)",
            "ML4: Бэггинг",
            "ML5: Стэкинг",
            "ML6: Нейросеть",
        ],
    )

    # Функции ленивой загрузки моделей (чтобы не падали, если файлов еще нет)
    @st.cache_resource
    def load_ml_model(choice):
        try:
            if choice == "ML1: Классическая модель":
                with open("model_ml1.pkl", "rb") as f:
                    return pickle.load(f)
            elif choice == "ML2: Ансамбль (XGBoost)":
                model = xgb.Booster()
                model.load_model("model_ml2.json")
                return model
            elif choice == "ML3: Advanced Бустинг (CatBoost)":
                model = cb.CatBoostRegressor()  # или CatBoostClassifier
                model.load_model("model_ml3.cbm")
                return model
            elif choice == "ML4: Бэггинг":
                with open("model_ml4.pkl", "rb") as f:
                    return pickle.load(f)
            elif choice == "ML5: Стэкинг":
                with open("model_ml5.pkl", "rb") as f:
                    return pickle.load(f)
            elif choice == "ML6: Нейросеть":
                return tf.keras.models.load_model("model_ml6.keras")
        except Exception as e:
            return f"Ошибка загрузки: файл модели не найден. ({str(e)})"

    # Вариант 1: Загрузка через CSV
    st.write("### Вариант А: Загрузка данных из CSV-файла")
    uploaded_file = st.file_uploader(
        "Выберите файл .csv для пакетного прогноза", type="csv"
    )
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:", input_df.head())
        # Здесь должна быть логика предобработки input_df и model.predict()
        st.success("Файл успешно обработан!")

    st.write("---")

    # Вариант 2: Ручной ввод параметров
    st.write("### Вариант Б: Ручной ввод параметров")

    # Интерактивные виджеты ввода (примеры с валидацией и единицами измерения)
    f1 = st.number_input(
        "Введите признак 1 (например, Площадь в кв.м.):",
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=0.1,
    )
    f2 = st.slider("Укажите признак 2 (Возраст здания в годах):", 0, 100, 5)

    # Кнопка запуска инференса
    if st.button("Рассчитать прогноз"):
        model = load_ml_model(model_choice)

        if isinstance(model, str):
            st.error(model)
        else:
            # Формируем вектор для предсказания (подставьте вашу логику)
            features = np.array([[f1, f2]])

            # Пример вывода заглушки прогноза (замените на реальный predict)
            # pred = model.predict(features)
            dummy_pred = float(f1 * 1200 + (100 - f2) * 50)

            st.write("### Результат прогнозирования:")
            # Вывод в понятной интерпретации для пользователя (валюта, денежный формат)
            st.metric(
                label=f"Прогноз стоимости по модели ({model_choice})",
                value=f"{dummy_pred:,.2f} ₽",
            )
