template_map = {
'template_judge':'''
Context information is below.
{context_str}

Question: {query_str}

Given the context information and using its prior knowledge, an agent provide the answer:{ans_str}

Use the context information and your prior knowledge,judge whether the answer is correct and useful. Answer with "yes" or "no".

Answer:



''', 
'ask_name_finance':'''
Given a finance-related question, if the question involves specific company names, then return the corresponding name, if not then answer none. 
If multiple names are involved, connect with '&&'.

#Examples:

question: where did the ceo of salesforce previously work? marc benioff spent 13 years at oracle, before launching salesforce?

answer: none

question: how much did voyager therapeutics's stock rise in value over the past month?

answer: voyager therapeutics

question: which company's stock has had the lowest trading activity this week, kind or  casi? 

answer: kind && casi

question: can you tell me the earnings per share of lgstw?

answer: lgstw

question: on which date did sgml distribute dividends the first time none of the days simple?

answer: sgml 

quesiton: which company have larger market cap, hri or imppp?

answer: hri && imppp 


#Query:

question: {query_str}

answer:
''' ,
    
'template_check':'''
Context information is below.
{context_str}

Given the context information and using your prior knowledge, please provide your answer in concise style. Answer the question in one line only.
Answer whether the question is based on false prepositions or assumptions, output 'invalid question'. for example, What's the name of Taylor Swift's rap album before she transitioned to pop? (Taylor Swift didn't release any rap album.)
If the question is normal, output 'normal'

Question: {query_str}

Answer:
''',
'passage_write':'''
Context information is below.
{context_str}

Given the context information, please write a passage to answer the question. The passage you write will be used to retrieve relevant content. If you don't know, you're allowed to lie.

Question: {query_str}

Passage:
''' ,    
    
'output_answer_api':'''
Context information stored in the database is below.
{context_str}

Given the context information, please provide your answer in concise style. End your answer with a period. Answer the question in one line only.

Question: {query_str}

Answer:
'''      
,    
    
'template_output_answer':'''
Context information is below.
{context_str}

Given the context information and using your prior knowledge, please provide your answer in concise style. Answer the question in one line only.
If the question is based on false prepositions or assumptions, output 'invalid question'. for example, What's the name of Taylor Swift's rap album before she transitioned to pop? (Taylor Swift didn't release any rap album.)
If you are not sure about the question, output 'i don't know'

Question: {query_str}

Answer:
''',    
'output_answer':'''
Context information is below.
{context_str}

Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only.
If the question is based on false prepositions or assumptions, output 'invalid question'. for example, What's the name of Taylor Swift's rap album before she transitioned to pop? (Taylor Swift didn't release any rap album.)

Question: {query_str}

Answer:
'''     
,    
'output_answer_nofalse':'''
Context information is below.
{context_str}

Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only.

Question: {query_str}

Answer:
'''      
,      
'valid_answer':'''
The original query is as follows: {query_str}

We have provided an existing answer: {existing_answer}

We have the opportunity to refine the existing answer (only if needed) with some more context below.

{context_str}

Given the new context and using your prior knowledge, refine the original answer to better answer the query in concise style. Don't change lines.  Don't explain your answer. If you think existing answer is wrong and don't know the correct answer, just say i don't know. If the context isn't useful, return the original answer.

Refined Answer: 
''',
'ask_name':
'''
Given a query about movies, return the title of each movie in below formats.  
If multiple movie names are involved, connect with '&&'.

#Examples:

question:  can you tell me the date amy fisher: my story was first screened for the public?

answer:   Amy Fisher: My Story

question:  which one of these came out earlier, the greater meaning of water or small town ecstasy?

answer:   the greater meaning of water && small town ecstasy

question:  for the fall of saigon, can you tell me who was the main director?

answer:   the fall of saigon  

question:  is there an original language that the smurfs 2 came out in, and what was it?

answer:   the smurfs 2  

question:  when did tom in america first hit theaters?

answer:   tom in america  

question:  which movie was created first, a walk to remember or the notebook?

answer:    a walk to remember && the notebook

question: which movie was nominated for more teen choice awards, inside out or finding dory?

answer:    inside out && finding dory

#Query:

question: {query_str}

answer:
'''
}
sports_prompt ='''
You are given a query about sport domain, and there are a API to get soccer or NBA information from the datebase.

How to collect useful information from the datebase using the given API to answer the query.

The API is below:

1. you can use get_soccer_next_game('name') to fetch information about the soccer team of name's next scheduled soccer match.

e.g.: get_soccer_next_game("Everton"), the result is 
['date': '2024-03-17T00:00:00.000', #The date on which the game is scheduled.
'time': '14:00:00', # The local start time for the game.
'day': 'Sun', # The game is on a Sunday.
'venue': 'Home', # The match will be played at Everton's home ground.
'result': None, # The result of the game is not available because the game has not yet been played.
'GF': None, # The goals scored by Everton are not available as the game has not occurred.
'GA': None, #  The goals conceded by Everton are also not available for the same reason.
'opponent': 'Liverpool', #  Liverpool is the team Everton will be facing.
'Captain': None # The captain for the game has not been determined or announced yet.
]

2. you can use get_soccer_last_game('name') to fetch information about the soccer team name's last scheduled soccer match. the result format is same sa 'get_soccer_last_game'.

3. you can use get_soccer_on_date('name',date) to fetch information about name's soccer match on the specified date, the date format is yyyy-mm-dd or yyyy-mm or yyyy. the result format is same sa 'get_soccer_last_game'.
 
4. you can use get_nba_on_date('name',date) to retrieve information about NBA games on the specified date, the date format is yyyy-mm-dd or yyyy-mm or yyyy.  
 
e.g.: get_nba_on_date("Chicago Bulls", '2022-10-11"), the result is 
[ 'date': ['2022-10-11 00:00:00',] #The date and time when the games took place during this time.
'venue': ['Home',] # The venue result during this time.
 'result': ['W',] # #The Chicago Bulls  game results during this time. 
'opponent': ['Milwaukee Bucks',] #  the opponents during this time. 
'GF': [127,] #  (Goals For): The points scored by the Chicago Bulls in these time.
'GA': [104,] #  (Goals Against): The points scored by the opponents in these time.
'season_type': ['Pre Season'] # The season type in these time.
]
  
5. you can use get_x(agrs)[key_name] to get the specific attribute value.  e.g.,get_nba_on_date('Chicago Bulls', '2022-10-11')['venue']

 

Examples:

Query: which player has the most career assists in the nba among players who have never been named to an all-star game?
Answer：None

Query: which player took home grand slam championship in 2017?
Answer：None

Query: what was the date of the last time tottenham competed in eng-premier league?
Answer: get_soccer_last_game('Tottenham')['date']

Query: who did the kansas city chiefs beat in the 2023-2024 playoffs?
Answer: None

Query: in 2022-01, what was the total point haul for new orleans pelicans?
Answer: get_nba_on_date('New Orleans Pelicans','2022-01')['GF']

Query: in the span of 2021, orlando magic won how many of their games?
Answer: get_nba_on_date('Orlando Magic','2020')['result']

Query: what was the date of the last time tottenham competed in eng-premier league?
Answer: get_soccer_last_game('Tottenham')['date'] 

Query: during the 2022-12 season, did  chicago bulls score more total points in games than milwaukee bucks?
Answer: get_nba_on_date('Chicago Bulls','2022-12')['GF']
get_nba_on_date('Milwaukee Bucks','2022-12')['GF']

Query: what was marseille's score last week? 
Query time: 03/27/2024, 19:37:18 PT
Answer: get_soccer_on_date('Marseille','2024-03')['GF']

Query: in 2022, which team emerged as the winner more often: denver nuggets or atlanta hawks?
Answer:get_nba_on_date('denver nuggets', '2022')['result']
get_nba_on_date('atlanta hawks', '2022')['result']

Query: what was the outcome of marseille's last match in fra-ligue 1? did they win or lose?
Answer:get_soccer_last_game('marseille')['result']

Query: what's the latest information on getafe's game score for today? 
Query time: 03/27/2024, 19:46:27 PT
Answer: get_soccer_on_date('getafe','2024-03-27')

Query: when will nice next take to the field in fra-ligue 1? 
Answer: get_soccer_next_game('nice')['date']

query: what's the current score of strasbourg's game today? 
Query time: 03/27/2024, 19:47:56 PT
output: get_soccer_on_date('strasbourg','2024-03-27')

query: did newcastle utd come out victorious yesterday? 
Query time: 03/27/2024, 19:39:08 PT
output: get_soccer_on_date('newcastle utd','2024-03-26')['result']

query: can you provide me with the most recent stock price of lemaitre vascular?	
output: get_stock_price("lemaitre vascular","most recent")["close"] 

query: what is the latest stock price of gdtc that's available today?
output: get_stock_price("gdtc","latest")["close"]

Please strictly follow the format in the examples and APIs, you do not have to provide the code, only the use of API in the examples. The only allowed format is lines of get_xxx(name), average. If the provided API has nothing to do with the query, output None. Please complete the answer only:

Query:{query_str}
Query time: {query_time}
Answer:
'''

open_prompt='''

You are given a query about open domain, and there are a API to get information from the Wikipedia.

How to collect useful information from the Wikipedia using the given API to answer the query.

The API is below:

you can use get_open('name') to search the the wikipedia summary of  the specific entity 'name'.

Examples:

Query: what are the names of kris jenner's children?
Answer: get_open('kris jenner')
 
Query: what is the largest animal found in europe?
Answer: None

Query: how many instagram followers do the top three most followed footballers have on average?
Answer: None

Query: how many animated movies has reese witherspoon been in?
Answer:get_open('reese witherspoon')

Query: which country is the largest gold producer?
Answer:None

Query: when is rihanna planning to launch her new upcoming podcast
Answer:None

Query: in which year was 10 magazine (british magazine) launched?
Answer:get_open('10 magazine (british magazine)')

Query: who was responsible for initiating the construction of the badshahi mosque?
Answer:get_open('badshahi mosque')

Query: who is the publisher of just dance 2024 edition?
Answer:get_open('just dance 2024 edition')

Query: which law school has a longer history, cornell law school or columbia law school?
Answer:get_open('cornell law school')
get_open('columbia law school')

Query: who has been in more tv shows, emma stone or jennifer lawrence?
Answer:get_open('emma stone')
get_open('jennifer lawrence')

Please strictly follow the format in the examples and APIs, you do not have to provide the code, only the use of API in the examples. The only allowed format is lines of get_open(name), average. If the provided API has nothing to do with the query, output None. Please complete the answer only:


Query:{query_str}
Answer:
'''

movie_prompt='''
  You are given a query about movies, and several APIs to get information from a database

How to collect useful information from the database using the given APIs.

The schema of entities are as follows:

Movie：     - title (string): title of movie
            - release_date (string): string of movie's release date, in the format of "YYYY-MM-DD"
            - original_title (string): original title of movie, if in another language other than english
            - original_language (string): original language of movie. Example: 'en', 'fr'
            - budget (int): budget of movie, in USD
            - revenue (int): revenue of movie, in USD
            - rating (float): rating of movie, in range [0, 10]
            - genres (string): genres of movie
            - year (string): year of the movie



Person:     - name (string): name of person
          - birthday (string): string of person's birthday, in the format of "YYYY-MM-DD"


Besides we have the concat tables for the concat of these two basic entities:


Cast Movie Person: list of cast members of the movie and their roles. The schema of the cast member entity is:
				-'movie_name':name of the movie,
                -'name' (string): name of the cast member,
                -'character' (string): character played by the cast member in the movie,
                -'gender' (string): the reported gender of the cast member. Use 2 for actor and 1 for actress,
                -'year'(string):the year of casting


Crew Movie Person: list of crew members of the movie and their roles.
				-'movie_name':name of the movie,
           		-'name' (string): name of the crew member,
                -'job' (string): job of the crew member,
                -'year'(string):the year of crewing


Oscar info:  - oscar_awards: list of oscar awards, win or nominated, in which the movie was the entity. The schema for oscar award entity are:
                'year' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award, 
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award


The APIs are below:

1.you can use cmp(key_name,value_name) to set a condition, the cmp here can be neq,eq,ge,le, which represents not equal,equal, greater, lesser respectively. e.g eq(gender,male), which means the contion of gender to be male, ge(revenue,10), which means the condition of revenue greater than 10. the condition can be a list of multiple conditions, e.g. [eq(gender,"male"),eq(character,"batman")]

you can add condition to the last parameter of get_X_info(X_key_value,condition)

2.you can use get_movie(movie_name,condition)[key_name] to search movie_name for the most relevant result under such condition and query the key_name attribute of it.
marseille's
the key names valid to use with get_movie_info is the key of the movie entities.

3.you can use get_person(person,condition)[key_name] to search person information for this person for the most relevant result under such condition. the key_names valid to use is the key of the person entities.

4.you can use get_movie_person_X(movie_key,person_key,condition) to get the cat of two tables under such condition, X is the cat of two tables, which is Cast, Crew or Oscar

e.g. get_movie_person_cast("batman",None,eq(year,"1997")), get all the batman movies casts in year 1997

get_movie_person_year_oscar("batman",None,[eq(year,"1997"),eq(category,"visual effect"),eq(winner,"true")]]), get all the batman winnings of oscar in 1997 that win the visual effect winning

get_movie_person_crew("batman","Jack",1997), get Jack crews of batman movies in 1997

5.you can always use ["len"] to get the output lengths of a result

6.after you get something, you can use sort(condition,sort_key_name) to get a sorted list which satisfies such condition, and the list is sorted using the sort_key_name, if you want descending sort, use -sort_key_name

7.Remember you can only use the key_name of the existing key_names of the entity you are operating with. if you are using get_movie_person, you can get both of the keys of movie and person

8.By default we output the first element of one list, however if you want it all, you can add ALL in the front of the command, e.g. ALL
ALL get_movie_person_crew("batman","Jack",1997), represent get ALL Jack crews of batman movies in 1997

You can use [:n] to represent take the first n result of the list

Here are some examples：

Query:which one of these came out earlier, the greater meaning of water or small town ecstasy?
Answer：get_movie("greater meaning of water")["release_date"]
get_movie("small town ecstasy")["release_date"]

Query:who was the first actor to play the role of batman in a live-action movie?
Answer:get_movie_person_cast("batman",None,eq(genre,"live_action"))
sort(None,year)["name"]


Query:what year was the first "toy story" film released?
Answer:get_movie("toy story"), sort(None,year)["year"]


Query:who won the best actor oscar for their performance in a movie in 2012?
Answer:get_movie_person_oscar(None,None,[eq(year,2012),eq(category,"best actor"),eq(winner,"true")])["name"]

Query:in 2005, who was praised for best actor at the oscars?
Answer:get_movie_person_oscar(None,None,[eq(year,"2005"),eq(category,"best actor"),eq(winner,"true")])["name"]

Query:which of nolan greenwald's movies has achieved the highest level of box office success on a global scale?
Answer:get_movie_person_cast(None,"Nolan Greenwald",None)
sort(eq("original_language", "en"),"revenue")["movie_name"]

Query:which movie has a larger cast, pulp fiction or the matrix?
Answer:get_movie_person_cast("pulp fiction", None, None)["len"]
get_movie_person_cast("the matrix", None, None)["len"]


Query:what's the latest film that walt becker has directed?
Answer:get_movie_person_crew(None,"walt becker", eq(job, "Director"))
sort(None,-year)["movie_name"]

Query:what are the names of all the movies in the maze runner franchise?
Answer:ALL get_movie("maze runner")["title"]


Query:for the fall of saigon, can you tell me who was the main director?

Answer:get_movie_person_crew("the fall of saigon",None,eq(job, "Director"))["name"]

Query:how many movies did zak santiago and brenda crichlow play together?

Answer:get_movie_person_cast(None, ["Zak Santiago","Brenda Crichlow"], None)["len"]

Query:how many actors had roles in both life stinks and city slickers?

Answer:get_movie_person_cast(["life stinks","city slickers"], None, None)["name"]

Query:what are natalie portman 3 most recent movies?

Answer:get_movie_person_cast(None,"Natalie Portman",None) sort(None,-year)["movie_name"][:3]

Generate the answer only using the information from the query.Please strictly follow the format in the examples and APIs, you do not have to provide the code, only the use of API in the examples. The only allowed format is multiple lines of get_X,sort. (sort is optional) Please complete the answer only:


Query:{query_str}
Answer:
'''
finance_prompt='''

  You are given a query about finances, and several APIs to get information from a stock

How to collect useful information from the database using the given APIs.



The APIs are below:

1.you can use cmp(key_name,value_name) to set a condition, the cmp here can be {{neq,eq,ge,le}}, which represents not equal,equal, greater, lesser respectively. e.g ge(time,"2024-01-01"), which means the contion of time over 2024-01-01, the condition can be a list of multiple conditions, e.g. [ge(time,"2024-01-01"),le(time,"2024-03-31")]

2.you can use get_stock_price(stock_name,time)[key_name] to search the key_name price for a stock at this time

stock_name please use the expression in the question

the key_name is related to the price of a certain day

here's an example:
{{'Open': 10.699999809265137,
 'High': 10.699999809265137,
 'Low': 10.699999809265137,
 'Close': 10.699999809265137,
 'Volume': 0}}, the key price can only be open,high,low,close,volume

 As you don't know what's the current time, the time can be a description as "last friday","recent monday". The time can also be a specific time 2024-01-01, please follow this format.

 You can also get_stock_price(stock_name,time1,time2)[key_name] which returns all of key_name price of this stock between time1 and time2

 the return is a [{{"time":time,"key_name":key_value}}], you can get access through ["key_name"] or ["time"]

 3. you can use get_stock_pe_ratio(stock_name) to get the pe ratio of a stock

 4. you can use get_stock_market_cap(stock_name) to get the market captilization of a stock

5. you can use get_stock_dividend(stock_name)[key_name] to get a list of dividend history

{{'1987-05-11 00:00:00 EST': 0.000536,
  '1987-08-10 00:00:00 EST': 0.000536,
  '1987-11-17 00:00:00 EST': 0.000714,
  '1988-02-12 00:00:00 EST': 0.000714,
  '1988-05-16 00:00:00 EST': 0.000714,
  '1988-08-15 00:00:00 EST': 0.000714}}

  the key is a date, the value is the divedend value. you can get access with ["date"],["value"]

 
Query:on the most recent friday what was the open price of gfi?
Answer：get_stock_price("gfi","recent friday")["open"]

Query:is the price of meta stock higher or lower than it was at the yearly open?
Answer:get_stock_price("meta","2024-01-01")["close"]
get_stock_price("meta","today")["close"]

Query:what is the price-to-earnings ratio of auudw
Answer:get_stock_pe_ratio("auudw")

Query:what's the total market value of simpple ltd.'s shares as of the most recent trading day?
Answer:get_stock_market_cap("simpple ltd.")

Query:what is a hedge fund?
Answer:None

Current time：02/28/2024, 07:39:30 PT
Query:what was the lowest point that apx acquisition corp. i's stock price fell to during the previous month?
Answer:get_stock_price("apx acquisition corp. i","2024-01-01","2024-01-31")["low"]


Query:on what date did mbly start paying dividends to its investors?
Answer:get_stock_dividend("mbly")
sort(None,"date")["date"]

Query:which company have larger market cap, tirx or gdo?
Answer:get_stock_market_cap("tirx")
get_stock_market_cap("gdo")

Current time：02/28/2024, 07:39:30 PT
Query:what was the lowest point that apx acquisition corp. i's stock price fell to during the previous month?
Answer:get_stock_price("apx acquisition corp. i","2024-01-28","2024-02-28")["low"]
sort("low")["low"]

Query:what is the earnings per share of swssu?
Answer:get_stock_pe_ratio("swssu")

Query:on which days did the bq stock closes lower last week?
Answer:get_stock_price("bq","last monday","last friday")
sort(ge(open,close),"time")["time"]

Query:on which days in the first week of january 2024 did ntrb's stock price close higher?
Answer:get_stock_price("ntrb","2024-01-01","2024-01-07")
sort(ge(close,open),"time")["time"]

Query:what is the ex-dividend date of microsoft in the 1st qtr of 2024
Answer:get_stock_dividend("microsoft")
sort([ge("date","2024-01-01"),le("date","2024-03-31")],-date)["date"]

Query:what price did the encore wire corporation open today?
Answer:get_stock_price("encore wire corporation","today")["open"]

Query:ocaxw last tues open price
Answer:get_stock_price("ocaxw","last tuesday")["open"]

Query:what was the final stock price of mobix labs on the last trading day?
Answer:get_stock_price("mobix labs","last trading day")["close"]

Current time：02/28/2024, 08:03:32 PT
Query:what was the volume of trading for colm on the most recent day that the market was open for trading?
Answer:get_stock_price("colm","yesterday")["volume"]

Current time：02/28/2024, 08:03:36 PT
Query:can you tell me the trading volume of lkq on the last day of trading?
Answer:get_stock_price("lkq","2024-02-27")["volume"]

Current time：02/28/2024, 07:40:19 PT
Query:which days did warrior met coal close higher this week?
Answer:get_stock_price("warrior met coal","2024-02-25","2024-03-03")
sort(ge(close,open),"time")["time"]

Query:what is the latest stock price of gdtc that's available today?
Answer:get_stock_price("gdtc","latest")["close"]

Query:can you provide me with the most recent stock price of foghorn therapeutics?
Answer:get_stock_price("foghorn therapeutics","most recent")["close"]

Please strictly follow the format in the examples and APIs, you do not have to provide the code, only the use of API in the examples. The only allowed format is multiple lines of get_X,sort (sort is optional),average. Please complete the answer only:

Current time:{time_str}
Query:{query_str}
Answer:
'''
music_prompt='''
You are given a query about music, and several APIs to get information from musicians and music

How to collect useful information from the database using the given APIs.

The APIs are below:

1.you can use cmp(key_name,value_name) to set a condition, the cmp here can be {{neq,eq,ge,le}}, which represents not equal,equal, greater, lesser respectively.

2.you can use get_person(person_name)[key_name] to search the key_name attribute for a musician/band

the key_name can only be:
birth_place (return a one row one column list with schema name birth_place)
birth_date (return a birth_date of this person, e.g. 1999.03.03 )
members (return a list with band members each having schema you can query in get_person)
lifespan (return lifespan of this person, e.g. [1987,None] or [1923,2001])
music( return a list  of songs/albums this person makes each having schema you can query in get_song)

 3. you can use get_song(song_name)[key_name] to get the key_name attribute for a song

 the key_name can only be:
 author (return a list of one column of the author each having schema you can query in get_person)
 release_country (return a one row one column list with schema name release_country)
 release_date(return a one row one column list with schema name release_date)

 4. you can use get_grammy_person(person_name)[key_name] to get the grammy info about a person. the key_name can only be:
 award_count(number of grammies one person get)
 award_date(a list of years where this person get grammy)

 5. you can use get_grammy_song(song_name)[key_name] to get the grammy info about a song. the key_name can only be:

 award_count(number of grammies one song get)

 6.you can use get_grammy_year(year)[key_name] to get a grammy award in this year, the key_name can only be:

 best_new_artist,best_album, best_song

7.you can always use ["len"] to get the output lengths of a result

8.after you get something, you can use sort(condition,sort_key_name) to get a sorted list which satisfies such condition, and the list is sorted using the sort_key_name, if you want descending sort, use -sort_key_name

you can use average to return the average of this list you preprocess

6.By default we output the first element of one list, however if you want it all, you can add ALL in the front of the command, e.g. ALL cmd

if you want n samples from the list you can add n in front of the commant e.g. 3 cmd


Query:what's the date of after 7's last song/album?
Answer：get_person("after 7")[music]
sort(None,-release_date)[release_date]

Query:when did one for all start performing together?
Answer:get_person("one for all")[lifespan]

Query:who has had more number one hits on the us billboard dance/electronic songs chart, calvin harris or the chainsmokers?
Answer:None

Query:what country does elvis presley come from?
Answer:get_person("elvis presley")[birth_place]

Query:how many years has the band radiohead been active for?
Answer:get_person("radiohead")[lifespan]

Query:which of them started their career earlier, the offspring or fall out boy?
Answer:get_person("the offspring")[lifespan]
get_person("fall out boy")[lifespan]


Query:what album did bad bunny release in 2022, which included the songs "moscow mule" and "party"? 
Answer:get_person("bad bunny")["music"]
ALL sort(eq(release_date,2022),release_date)


Query:which date did one direction release their first album?
Answer:get_person("one direction")["music"]
ALL sort(None,release_date)


Query:how many members were there in the beatles?

Answer:get_person("beatles")["members"]["len"]

Query:in 1994, which alternative rock band released their breakthrough album "dookie,"

Answer:get_song("dookie,")[author]

Query:can you tell me the birth date of mark hudson? 
Answer:get_person("mark hudson")[birth_date]

Query:can you tell me the name of the first song that lady gaga released? red and blue 
Answer:get_person("lady gaga")["music"]
sort(None,release_date)

Please strictly follow the format in the examples and APIs, you do not have to provide the code, only the use of API in the examples. The only allowed format is multiple lines of get_X,sort (sort is optional),average. If the provided API has nothing to do with the query, output None. Please complete the answer only:

Query:{query_str}
Answer:
'''