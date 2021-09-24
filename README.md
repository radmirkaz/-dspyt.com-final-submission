# Описание
Это бенчмарк скрипт для хакатона от Раййфайзенбанка по оценке коммерческой недвижимости
Бенчмарк состоит из:
* pyproject.toml - конфигурационный файл для менеджера пакетов poetry (https://python-poetry.org/) - в интернете есть много статей, посвященных ему (например https://habr.com/ru/post/455335/ и https://khashtamov.com/ru/python-poetry-dependency-management/)
* requirements.txt - стандартный requirements для pip
* train.py - скрипт, который обучает модель и сохраняет ее
* predict.py - скрипт, который делает предсказание на отложенной тестовой выборке

# Запуск
## Вариант с poetry
**Крайне рекомендую именно установку с poetry** - poetry это новый packet manager для питона, и он гораздо круче чем pip. Разобравшись с ним (а это очень-очень просто), думаю, вы будете необычайно счастливы.
Для запуска необходимо:
<ol>
    <li> убедиться, что у вас стоит python3.6 или выше </li>
    <li> установить poetry:

     pip install poetry 
</li>
    <li> установить все нужные пакеты из poetry.lock:
    <ol>
        <li> по умолчанию poetry создает виртуальное окружение - это лучше для изоляции от вашей системы и рекомендуем именно такой способ установи пакетов:
            
         poetry  install  
</li>
        <li> если хочется установить без виртуального окружения, то установить нужно с помощью следующей команды:
            
        poetry config virtualenvs.create false && poetry  install
</li>
    </ol> 
    </li>
    <li> запустить обучение

    poetry run python3 train.py --train_data <path_to_train_data> --model_path <path_to_pickle_ml_model>
</li>
    <li> запустить предикт

    poetry run python3 predict.py --model_path <path_to_pickled_model> --test_data <path_to_test_data> --output <path_to_output_csv_file>
</li>
    <li> загрузить полученные результаты в систему </li>
</ol>
## Вариант с requirements.txt
<ol>
    <li> убедиться, что у вас стоит python3.6 или выше </li>
    <li> установить зависимости:
    
    pip install -r requirements.txt 
</li>
    <li> запустить обучение

    python3 train.py --train_data <path_to_train_data> --model_path <path_to_pickle_ml_model>
</li>
    <li> запустить предикт
    
    python3 predict.py --model_path <path_to_pickled_model> --test_data <path_to_test_data> --output <path_to_output_csv_file>
</li>
    <li> загрузить полученные результаты в систему</li>
</ol>

В репозитории есть своя реализация регуляризованного target encoding (SmoothedTargetEncoding). можно поэкспериментировать с ним