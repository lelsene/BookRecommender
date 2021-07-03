import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# импорт данных о книгах из .csv в dataframe,
# error_bad_lines - удаление строк со слишком большим количеством полей
books = pd.read_csv(r'books.csv', error_bad_lines=False)
# удаление строк с пустыми полями
books = books.dropna()
# удаление строк с повторяющимися названиями книг,
# keep - сохранение последнего дубликата, inplace - изменение в текущем dataframe
books.drop_duplicates(subset='original_title', keep='last', inplace=True)

# импорт данных о книгах, которые хотят прочитать пользователи из .csv в dataframe
to_read = pd.read_csv("to_read.csv")
# удаление строк с повторяющимися парами - пользователь, книга
to_read.drop_duplicates(subset=["user_id", "book_id"], keep='last', inplace=True)

# импорт данных об оценках пользователей из .csv в dataframe
ratings = pd.read_csv('ratings.csv')
# сортировка оценок по пользователям
ratings = ratings.sort_values("user_id")
# удаление строк с повторяющимися парами - пользователь, книга
ratings.drop_duplicates(subset=["user_id", "book_id"], keep='last', inplace=True)

# импорт данных о книгах и отнесенным к ним тегам из .csv в dataframe
btags = pd.read_csv('book_tags.csv')
# удаление строк с повторяющимися парами - тег, книга
btags.drop_duplicates(subset=['tag_id', 'goodreads_book_id'], keep='last', inplace=True)

# импорт данных о тегах из .csv в dataframe
tags = pd.read_csv('tags.csv')
# удаление строк с повторяющимися тегами
tags.drop_duplicates(subset='tag_id', keep='last', inplace=True)

# добавление к данным о книгах и отнесенным к ним тегам полей из данных о тегах (имя тега),
# left_on - имя столбца для присоединения в левом dataframe, идентификатор тега
# right_on - имя столбца для присоединения в правом dataframe, идентификатор тега
# how - тип слияния, использование пересечения ключей
joint_tags = pd.merge(tags, btags, left_on='tag_id', right_on='tag_id', how='inner')

# группировка по названию тега, подсчет количества значений в каждой группе
group_tags = joint_tags.groupby('tag_name').count()
# сортировка тегов по убыванию количества использования
group_tags = group_tags.sort_values(by='count', ascending=False)

# стандартный набор жанров
genres = ["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics", "Comics", "Contemporary",
          "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction", "Gay and Lesbian", "Graphic Novels",
          "Historical Fiction", "History", "Horror", "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery",
          "Nonfiction", "Paranormal", "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science",
          "Science Fiction", "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]

# привидение жанров к нижнему регистру
for i in range(len(genres)):
    genres[i] = genres[i].lower()

# отбор тех сгруппированных тегов, которые встречаются в стандартных жанрах
new_tags = group_tags[group_tags.index.isin(genres)]

# построение круговой диаграммы по популярным жанрам
fig = go.Figure(data=[go.Pie(labels=new_tags.index, values=new_tags['count'], textinfo="none", title='Top Genres')])
# fig.show()

# создание визуального представления списка жанров с помощью облака слов
wordcloud = WordCloud().generate(str(new_tags.index.values))
# определение размера графика и фона
plt.figure(figsize=(8, 8), facecolor=None)
# представление данных графика как изображение
plt.imshow(wordcloud)
# скрытие системы координат
plt.axis("off")
# plt.show()

# добавление к данным о книгах, которые хотят прочитать пользователи полей из данных о книгах
books_to_read = pd.merge(to_read, books, left_on='book_id', right_on='book_id', how='inner')
# группировка по названию книги, подсчет количества значений в каждой группе
books_to_read = books_to_read.groupby('original_title').count()
# сортировка книг по убыванию количества желающих прочесть
books_to_read = books_to_read.sort_values(by='id', ascending=False)
# выборка 40 самых желаемых к прочтению книг
books_to_read_top = books_to_read.head(20)
# построение гистограммы по 20 самым желаемым к прочтению книгам
fig = px.bar(books_to_read_top, x=books_to_read_top.index, y='id', color='id')


# fig.show()


# метод для привидения к нижнему регистру без пробелов
def lower_case(x):
    return str.lower(x.replace(" ", ""))


# необходимые поля для контентной фильтрации
features = ['original_title', 'authors', 'average_rating']
# отбор книг по полям
features_books = books[features]
# привидение значений полей к строчному типу
features_books = features_books.astype(str)
# приминения метода привидения к нижнему регистру к каждой книге
for feature in features:
    features_books[feature] = features_books[feature].apply(lower_case)


# создание строки из значений всех полей
def create_soup(x):
    return x['original_title'] + ' ' + x['authors'] + ' ' + x['average_rating']


# создание поля со строчным представлением всех полей каждой книги
features_books['soup'] = features_books.apply(create_soup, axis=1)

# иницилизация модуля CountVectorizer для анализа количественного вхождения слов
# stop_words='english' - использование встроенного списка стоп-слов для английского языка
vectorizer = CountVectorizer(stop_words='english')
# подсчет количества вхождения слов в значении поля со строчным представлением всех полей каждой книги
count_matrix = vectorizer.fit_transform(features_books['soup'])
# вычисление косинусного сходства между книгами
cosine_sim = cosine_similarity(count_matrix)

# введение стандартных индексов
features_books = features_books.reset_index()

# создание одномерного помеченного массива, содержащего соответствия между индексом книги и ее названием
indices = pd.Series(features_books.index, index=features_books['original_title'])


# метод получения рекомендаций, основанных на схожести книг (content-based)
def get_content_based_recommendations(title, cosine_similarity=cosine_sim):
    # привидения названия в нижний регистр без пробелов
    book_title = title.replace(' ', '').lower()
    book_index = indices[book_title]
    # получение парной схожести между выбранной и всеми книгами
    sim_scores = list(enumerate(cosine_similarity[book_index]))
    # сортировка книг на основе степени схожести по убыванию
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # отбор десяти самых похожих книг
    sim_scores = sim_scores[1:11]
    # получение индексов отобранных книг
    books_indices = [i[0] for i in sim_scores]
    # отбор списка названий книг по полученным индексам книг
    return list(books['original_title'].iloc[books_indices])


# необходимые поля для коллаборативной фильтрации (collaborative filtering)
collaborative_columns = ['book_id', 'original_title']
# отбор книг по полям
collaborative_books = books[collaborative_columns]

# минимальное количество оценок книги
rating_count = 60
# группировка книг по количеству оценок
books_ratings = pd.DataFrame(ratings.groupby('book_id').size(), columns=['count'])
# список книг с количеством оценок >= минимального
popular_books = list(set(books_ratings.query('count >= @rating_count').index))
# рейтинги книг, количество оценок которых >= минимального
ratings_ = ratings[ratings.book_id.isin(popular_books)]

# минимальное количество оценок пользователя
rating_user = 50
# группировка пользователей по количеству оценок
users_ratings = pd.DataFrame(ratings_.groupby('user_id').size(), columns=['count'])
# список пользователей с количеством оценок >= минимального
active_users = list(set(users_ratings.query('count >= @rating_user').index))
# рейтинги книг пользователей, количество оценок которых >= минимального
ratings_ = ratings_[ratings_.user_id.isin(active_users)]

# создание сводной таблицы (колонки - пользователи, строки - книги, значение - рейтинг)
df_collaborative_books = ratings_.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
# преобразование сводной таблицы к разреженной матрице
sparse_collaborative_books = csr_matrix(df_collaborative_books.values)
# создание модели для алгоритма ближайших соседей
# косинусная метрика, алгоритм - поиск метододом полного перебора, количество соседей,
# использование всех процессоров при поиске соседей
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# подгон модели, используя в качестве данных обучения разреженную матрицу оценок-пользователей-книг
model_knn.fit(sparse_collaborative_books)


# метод поиска индекса книги с помощью нечетких сравнений
def fuzzy_matching(fav_book, mapper=indices):
    # список похожих книг
    match_tuple = []
    for title, index in mapper.items():
        # нечетное простое сравнение названий книг
        ratio = fuzz.ratio(title.lower(), fav_book.lower())
        # если коэффициент сравнения >= 60, то добавляем книгу в список
        if ratio >= 60:
            match_tuple.append((title, index, ratio))
    # сортировка книг по коэффициенту сравнения по убыванию
    match_tuple = sorted(match_tuple, key=lambda x: x[2], reverse=True)
    # возвращение индекса fav_book
    return match_tuple[0][1]


def get_collaborative_recommendations(fav_book, data=sparse_collaborative_books, n_recommendations=20, mapper=indices,
                                      model_knn=model_knn):
    # индекс выбранной книги (или похожей)
    fav_book_index = fuzzy_matching(fav_book)
    # нахождение соседних книг
    book_distances, book_indices = model_knn.kneighbors(data[fav_book_index], n_neighbors=n_recommendations + 1)
    # сортировка по расстоянию до соседних книг
    raw_recommends = sorted(list(zip(book_indices.squeeze().tolist(), book_distances.squeeze().tolist())),
                            key=lambda x: x[1], reverse=True)[:-1]
    # изменение обращения к массиву соответствий индексов и названий книг (раньше - по названию, сейчас - по индексу)
    reverse_mapper = {v: k for k, v in mapper.items()}
    # список рекомендаций
    rec_indices = []
    for index, distance in enumerate(raw_recommends):
        # если индекса нет в списке книг - пропускаем
        if index not in reverse_mapper.keys():
            continue
        # добавление индекса книги в список
        rec_indices.append(index)
    # отбор списка названий книг по полученным индексам книг
    return list(books['original_title'].iloc[rec_indices])


name = "Harry Potter and the Order of the Phoenix"
cont = get_content_based_recommendations(name)
collab = get_collaborative_recommendations(name)
# print(f"Content-based recommendations for '{name}'")
# for x in cont:
#     print("->", x)
# print()
print(f"Collaborative recommendations for '{name}'")
for x in collab:
    print("->", x)
