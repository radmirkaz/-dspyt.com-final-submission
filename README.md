# RaifHack solution [dspyt.com]

### Описание

1. Обучение на данных price_type=1
2. Чистка данных (floor, categorical columns)
3. 10 фолдов
4. Предикт домножаем на 0.94
5. Ансамбль LGBMRegressor (0.75) + XGBRegressor (0.05) + CatBoostRegressor (0.2)
6. Публичный лидерборд - 1.4062123483345823

Файл работы - <code>raifhack-dspyt-com-final-solution.ipynb</code>
Файо для отправки - <code>submission_final_raif</code>

### Запуск
<ol>
    <li> убедиться, что у вас стоит python3.6 или выше </li>
    <li> установить зависимости:
    
    pip install -r requirements.txt 
</li>
    <li> запустить обучение, предикт
        Обучение и предсказание моделей происходит в ноутбуке <code>raifhack-dspyt-com-final-solution.ipynb</code>
</li>
    <li> загрузить полученные результаты в систему</li>
</ol>

### Анализ

1. Пробовалось аггрегировать фичи, но это не дало результат.
2. В колонке city хранятся данные не только о городах, но и о улицах, районах, метро, АССР....
3. Очищение колонки floor дало наиболее ощутимый результат.
4. Кросс валидация улучшает результат в среднем на 0.02
5. Пробовались разные варианты ансамблирования, оптимальный представлен в описании (п.5)
6. Пост процессинг, для удаления выбросов умножаем предикты соло моделей на 0.9, финального на 0.94

Распределение недвижимости по России:

![dist_of_estate]()

Важность дефолтных фич:

![feature_importances](https://github.com/RadmirZ/-dspyt.com-final-submission/blob/main/feature_importances.PNG?raw=true)

Распределение предсказаний:

![dist](https://github.com/RadmirZ/-dspyt.com-final-submission/blob/main/distribution.PNG?raw=true)



