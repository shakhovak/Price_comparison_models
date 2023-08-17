# Постороение моделей для сравнения цен на похожие товары.
![image](https://github.com/shakhovak/Price_comparison_models/assets/89096305/0b0a2fa7-e389-4162-8cf8-ce9d2da8dc83)

## Цель проекта.
Основная цель проекта - находить похожие/аналогичные товары для заданной категории товаров по их описанию (+- характеристикам) и/или изображению для позиционирование по цене. 
В качестве **аналогов** рассматриваем товары, которые на 80-90% сходжи по характеристикам, **похожие** - менее 80%, но являющиеся заменителем и могут быть приобретены пользователем вместо сравниваемого товара.
## Этапы проекта.
Этапы проекта включают:
<li> Получение исходных данных для анализа - будем использовать уже опубликованные данные, поэтому репозиторий сделаю публичным :)
<li> Сбор данных конкурентов - для каждой группы категории он будет свой, но в большинство из них будут входить курпные маркетплейсы и оналйн магазины, будем на них тренироваться. Идеально, чтобы на каждый артикул исходного датасета было по 5 артикулов разных конкурентов.
<li> Векторизация собранной информации - буду пробовать разные модели, только на тексте/кратинках или мультимодальность
<li> Поиск похожих товаров - воспользуюсь косинусной близостью для определения похожих товаров. Уровень скора похожести поможет определить качество спарсенной выборки и/или уникальноститовара.
Определение рекумендуемой цены и сравнение ее с фактической/планируемой - можно использовать разные модели для регрессии на основе векторов, полученных на предыдущем этапе. Регрессоры будут обучаться на спарсенных данных (либо на всех для категории, либо отобранных по пороговому индексу похожести для категории) в зависимости от кол-ва собранных данных и индекса похожести.

</li>

### Этап 1. Получение исходных данных для анализа.
Для экспериментов и моделирования возьмем данные для 2-х категорий товаров:
<li> беговые дорожки - на этой категории будем проводить эксперименты. В категории почти 600 артикулов. Пример полученных данных на рисунке ниже. Дополнительно по каждому артикулу есть фотографии в формате jpeg.
  
![image](https://github.com/shakhovak/Price_comparison_models/assets/89096305/cb09e720-a9c0-456e-bc17-12c290b0aa9a)
  
<li> степперы - это будет валидационная категория, на ней будем смотреть как все работает. Категория включает 80 артикулов, данные получены в аналогином виде, есть фотографии

  ![image](https://github.com/shakhovak/Price_comparison_models/assets/89096305/d4d1b80f-ff56-45de-ad88-8386df9c6083)
</li>

Данные собраны в файл - добавить ссылку на файл

### Этап 2.Сбор данных конкурентов.
В качестве кокурентов выбраны следующие онлайн магазины и маркетплейсы:
<li> www.mvideo.ru - с помощью парсера (добавить ссылку) получено примерно 200 артикулов (добавить ссылку), не густо, но они нужны для полноты выборки. 
<li> озон + WB + спортмастер - пока работаю

  
</li>
