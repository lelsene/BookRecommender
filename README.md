# BookRecommender
Рекомендательная система по подбору книг, реализующая следующие функции с использованием набора данных (https://www.kaggle.com/zygmunt/goodbooks-10k) :
1.	Просмотр диаграммы наиболее популярных жанров книг;
2.	Просмотр диаграммы с книгами, которые пользователи чаще всего оставляют к прочтению;
3.	Подбор книг с помощью алгоритма content-based, основанного на схожести в содержании книг;
4.	Подбор книг с помощью коллаборативного алгоритма, основанного на оценках других пользователей

## Алгоритм коллаборативной фильтрации (item-based)
Основная идея - предложение новых элементов для конкретного пользователя на основе предыдущих предпочтений пользователя или мнения других единомышленников. Этот подход еще называют методом ближайших соседей.

#### Преимущества:
1. Является достаточно универсальным подходом, поэтому часто дает высокие результаты.
2. Для работы данного метода не нужна детальная информация о продуктах. Вместо этого используется как история оценок самого пользователя, так и других пользователей.

#### Недостатки:
1. Нельзя работать с новыми пользователями, для которых еще нет истории (задача холодного старта).
2. Неизвестно, что делать с новыми объектами, которые еще никто не оценил.
3. Ресурсоемкость вычислений, которая замедляет время работы системы.
4. Необходим большой объем данных для высокой точности предсказаний.

#### Схема работы алгоритма в системе подбора:
- выполнение подготовки данных, а именно: удаление повторов в наборе данных;
- группировка книг по количеству их оценок;
- отбор тех книг, количество оценок которых больше 60;
- группировка пользователей по количеству оценок;
- отбор тех пользователей, у которых количество оценок книг более 50;
- cоздание сводной таблицы, которая преобразуется к разреженной матрице;
- создание модели для алгоритма ближайших соседей, при подгоне которой в качестве данных обучения используется разреженная матрица оценок пользователей – книг;
- поиск выбранной книги;
- нахождение соседних книг с использованием созданной модели;
- сортировка по расстоянию до соседних книг по убыванию;
- отбор списка книг по полученным индексам;
- вывод рекомендованных книг пользователю.

## Алгоритм content-based
Основная идея - рекомендации основаны на схожести элементов. При данном алгоритме по истории действий пользователя формируется вектор его предпочтений, что бы в дальнейшем рекомендовались элементы близкие к этому вектору.

#### Преимущества:
1. Не требует большого количества пользователей для достижения высокой точности рекомендаций;
2. Новые элементы можно рекомендовать сразу, как только у них появляются заполненные характеристики.

#### Недостатки:
1. Сильная зависимость от предметной области, полезность рекомендаций ограничена;
2. Вектор предпочтений пользователя и вектор остальных элементов должен состоять из одинакового набора характеристик, чтобы их можно было сравнивать и составлять рекомендации.

#### Схема работы алгоритма в системе подбора:
- выполнение подготовки данных, а именно: удаление повторов в наборе данных;
- удаление пробелов в необходимых полях;
- приведение полей к нижнему регистру и строчному типу;
- получение парной схожести между выбранной книгой и всеми остальными;
- сортировка по убыванию степеней схожести;
- выборка двадцати самых похожих на указанную пользователем книгу;
- получение индексов отобранных книг;
- получение списка названий книг.

## Технологии
- Python 3.7.8
- Flask 1.1.2
- Bootstrap 4.1.3
- Pandas 1.1.2
- Scikit-learn 0.23.2
- SciPy 1.4.1
- FuzzyWuzzy 0.18.0
- Chart JS 2.9.4
- Bootstrap-select 1.14.0

## Пример работы
![1](https://user-images.githubusercontent.com/43280704/124367205-3a6e1580-dc66-11eb-902e-2aa54e091980.png)
![2](https://user-images.githubusercontent.com/43280704/124367208-3e019c80-dc66-11eb-83c6-d8e6517e7a94.png)
![3](https://user-images.githubusercontent.com/43280704/124367210-4063f680-dc66-11eb-8560-3b263524e7f5.png)
![4](https://user-images.githubusercontent.com/43280704/124367211-42c65080-dc66-11eb-961c-36208b6196d3.png)

