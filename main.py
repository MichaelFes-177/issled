import streamlit as st
import pandas as pd
import requests
from io import StringIO

file_url = 'https://drive.google.com/file/d/1HppWeYN230iazekdJq5GQvr0hB7n3t9t/view'
file_url1 = 'https://drive.google.com/file/d/1TO8fdG8VH3v4wZnroX2t_Wdo00kF6pFS/view?usp=sharing'
file_id = file_url.split('/')[-2]
file_id1 = file_url1.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
dwn_url1 = 'https://drive.google.com/uc?export=download&id=' + file_id1
file_content = requests.get(dwn_url).text
file_content1 = requests.get(dwn_url1).text
data = pd.read_csv(StringIO(file_content))
data1 = pd.read_csv(StringIO(file_content1))

st.title('Исследование "предсказание изменения акций компании"')
st.subheader('Проблема')
st.text('Была поставлена задача построения модели для предсказания\n'
        ' изменений акций крупной компаниив образовательных целях.\n '
        'с точки зрения технической части мы поставили задачу\n'
        ' построить модель временных рядов, которая способна\n'
        '  валидировать данные котировок Сбербанка с MOEX и предсказывать\n'
        '  движение котировок на временном промежутке в 1 час.')
st.subheader('Актуальность')
st.text('В настоящее время существует большое количество людей, которые\n'
        ' являются новичками в сфере финансовой аналитики, но хотят \n'
        'глубже погрузиться в эту область знаний. Наше решение \n'
        'призвано помочь такой аудитории. ')
st.subheader('Выбор категории рынка')
st.markdown('Мы выбрали банковскую сферу (Сбербанк) для нашего исследования\n'
            ' - она является ключевым сегментом экономики\n'
            ' - банки регулярно публикуют достоверные отчеты.\n'
            ' - большое количество доступных данных для анализа.'
            )
st.subheader('Подготовка данных')
st.text('В рамках кейса у нас имелся датасет с новостями(news_processed.csv)  \n'
        'и с котировками(train.csv)')
st.caption('news_processed.csv')
st.dataframe(data, width=800, height=200)
st.caption('train.csv')
st.dataframe(data1, width=800, height=200)
st.text('Для большей эффективности заголовки и контент новостей были\n'
        ' преобразованы в эмбеддинги. В результате мы получили датасеты\n'
        ' news_embedding_content.csv и News_embedding_title.csv')
st.subheader('Бейзлайн')
st.text('В качестве бейзлайна мы использовали модель линейной регрессии.')
st.markdown('Преимущества:\n'
            ' - Простая интерпретация результатов\n'
            ' - Эффективна для линейно разделимых классов\n'
            ' - Низкие вычислительные затраты\n')
st.markdown('Недостатки:\n '
            '- Не подходит для сложных нелинейных зависимостей\n '
            '- Может давать неточные вероятности для редких событий')
st.text('Mean Squared Error: 1.81\n'
        'Mean Absolute Percentage Error: 0.0035\n'
        'Root Mean Squared Error: 1.3452\n'
        'R-squared: 0.96')
st.image('dd.png')
st.subheader('Аналогичные решения')
st.markdown('- Citadel\n'
            ' - D. E. Shaw & Co\n'
            ' - Numerai\n'
            ' - BlackRock'
            ' - AlphaSense')
st.header('Методы глубокого обучения')
st.subheader('XGBoost')
st.text('XGBoost -  алгоритм градиентного бустинга на деревьях решений.\n '
        'Он обучает ансамбль слабых моделей, улучшая предсказания на\n'
        ' ошибках предыдущих моделей.')
st.markdown('Преимущества: \n'
            ' - высокая точность\n'
            ' - работает с нелинейностями\n'
            ' - предотвращает переобучение\n'
            ' - эффективно обрабатывает пропуски.')
st.markdown('Недостатки:\n'
            ' - может переобучаться на малых выборках\n'
            ' - требует настройки гиперпараметров\n'
            ' - менее интерпретируем.')
st.text('Mean Squared Error: 0.51\n'
        'Mean Absolute Percentage Error: 0.0015698279158068258\n'
        'Root Mean Squared Error: 0.7108751767929241\n'
        'R-squared: 0.9847553615392751\n')
st.image('kek.png')
st.subheader('CatBoost')
st.text('CatBoost - алгоритм градиентного бустинга от Яндекса,\n'
        'эффективный для категориальных признаков.\n'
        'Bспользует симметричные деревья и упорядоченный бустинг.')
st.markdown('Преимущества:\n'
            ' - отличная работа с категориальными признаками\n'
            ' - борьба с переобучением \n'
            ' - высокая точность\n'
            ' - быстрое обучение')
st.markdown('Недостатки:\n'
            ' - может быть медленнее XGBoost\n'
            ' - меньше возможностей для настройки')
st.text('Mean Squared Error: 0.53\n'
        'Mean Absolute Percentage Error: 0.001602423727587821\n'
        'Root Mean Squared Error: 0.7300961157967322\n'
        'R-squared: 0.9839198347869493')
st.image('dddd.png')

st.subheader('Выбор модели')
st.text("XGBoost показал лучшие результаты с точностью ~99%. \n"
        "Его точности хватает для успешной торговли на бирже.")
st.subheader('Дальнейшие перспективы')
st.markdown(' - Применение обучения с подкреплением для выработки стратегии торговли.\n'
        ' - Улучшение качества моделей с помощью более современных методов.\n'
        ' - Сделать возможность предсказания для большого количества тикеров (уже есть наработки)\n'
        ' - Создать стартап на основе данного проекта.\n')
