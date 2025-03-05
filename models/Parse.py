from models.pycragapi import CRAG
import numpy as np
import re
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timedelta, date

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


def levenshtein_distance(s1, s2):
    # if len(s1) < len(s2):
    #    return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def longest_common_subsequence(s1, s2):
    # 创建一个二维数组以保存LCS的长度
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充dp数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def sort_strings_by_lcs_similarity(strings, query):
    # 计算每个字符串与查询字符串的LCS长度
    similarities = [(string, longest_common_subsequence(query, string)) for string in strings]

    # 根据LCS长度排序，长度较长的意味着更相似
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回只有字符串的列表，已排序
    return [string for string, _ in similarities]


def find_most_similar(strings, query):
    # 初始化最小距离为一个很大的数和最相似的字符串为空
    min_distance = float('inf')
    most_similar = None

    # 遍历每个字符串计算与查询字符串的编辑距离
    for string in strings:
        distance = levenshtein_distance(query, string)
        if distance < min_distance:
            min_distance = distance
            most_similar = string

    return most_similar


def sort_strings_by_similarity(strings, query):
    # 使用编辑距离计算相似度，并将字符串及其相似度分数放入列表
    distances = [(string, levenshtein_distance(query, string)) for string in strings]

    # 根据编辑距离进行排序（距离最小的最相似）
    distances.sort(key=lambda x: x[1])

    # 返回只有字符串的列表，已排序
    return [string for string, _ in distances]


def cosine_similarity(vec1, vec2):
    """ 计算两个向量之间的余弦相似度 """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def sort_strings_by_cosine_similarity(strings, query):
    # 将字符串和查询编码为向量
    embeddings = r.bge_large_embeddings.embed_documents(strings + [query])
    query_vec = embeddings[-1]
    string_vecs = embeddings[:-1]

    # 计算余弦相似度
    similarities = [(index, cosine_similarity(query_vec, vec)) for index, vec in enumerate(string_vecs)]

    # 根据相似度排序，相似度高的排在前面
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回已排序的字符串列表
    return [index for index, _ in similarities]


def extract_parts(input_string):
    args_list = []
    args_idx = 0
    str3_flag = False
    str4_flag = False
    input_string = input_string.replace('"', '').replace("'", "").strip().lower()
    tmp_string = ''
    stack = []
    for i in range(len(input_string)):
        char = input_string[i]
        if args_idx == 0:
            if char == '(':
                args_list.append(tmp_string)
                tmp_string = ''
                args_idx += 1
                stack.append('(')
            else:
                tmp_string += char
        elif args_idx == 1:
            tmp_string += char
            if char == '(':
                stack.append('(')
            if char == ')':
                stack.pop()
                if len(stack) == 0:
                    args_list.append(tmp_string)
                    tmp_string = ''
                    args_idx += 1
        elif args_idx == 2:
            tmp_string += char
            if char == '[':
                str3_flag = True
            if char == ']' and str3_flag:
                args_list.append(tmp_string[1:-1])
                tmp_string = ''
                args_idx += 1
                str4_flag = True
                i += 1
                break
            if tmp_string[-4:] == 'sort':
                str4_flag = True
                i -= 4
                break
    if not str3_flag:
        args_list.append('none')
    if str4_flag:
        print(input_string[i:])
        args_list.append(extract_parts(input_string[i:]))
    else:
        args_list.append('none')
    return args_list


# r = Retriever(batch_size=128, device='cuda:1')
api = CRAG()

confuse_word = {
    'runtime': 'length',
    'movie_name': 'title',
    'year': 'release_date',
    'len': 'length',
    'lenth': 'length'
}


def print_movie_info(m):
    tmp = f"<Movie> \'{m['title']}\' , the detailed information of it are: "
    for key in ['budget', 'release_date', 'revenue', 'rating', 'length', 'original_language', 'original_title']:
        if key in m.keys():
            tmp += f"the {key} is {m[key]}; "
    castnum = min(5, len(m['cast']))
    tmp += f". It has {len(m['cast'])} casts, and the top {castnum} cast are: "
    for cast in m['cast'][:castnum]:
        tmp += f"{cast['name']} palyed {cast['character']}; "
    crewnum = min(5, len(m['crew']))
    tmp += f". It has {len(m['crew'])} crews, and  the directors are: "
    for crew in m['crew']:
        if 'director' == crew['job'].lower():
            tmp += f"{crew['name']} serve as {crew['job']}; "
            crewnum -= 1
        if crewnum == 0:
            break
    tmp += '.</Movie>\n'
    return tmp


def remove_symbols(s):
    return re.sub(r'[^\w\s]', '', s).replace(' ', '')


def clear_conds(conds):
    conds_map = {}
    if 'none' in conds:
        return conds_map
    conds = conds.replace('=', ',')
    if 'sort(' in conds:
        conds = conds.split('sort(')
        assert len(conds) == 2
        sort_conds = conds[1]
        conds = conds[0]
        sort_conds = parse_sort(sort_conds)
        conds_map['sort'] = sort_conds
    if 'eq(' not in conds:
        for key in ['ge(', 'le(']:
            if key in conds:
                geitem = conds.split(key)[-1].split(')')[0]
                gekey, gevalue = geitem.split(',')
                conds_map[key + gekey] = gevalue.strip().lower()
    else:
        conds = [cond.replace(')', '').replace(']', '').strip().split(',') for cond in conds.split('eq(') if
                 cond not in ['[', '']]
        for cond in conds:
            k, v = cond[:2]
            k = k.replace('"', '').replace("'", "").strip()
            conds_map[k] = v.replace('"', '').replace("'", "").strip()
    return conds_map


def get_mathed_result(info, name):
    if len(info) == 0:
        print('no result')
        return []
    for item in info:
        if 'title' in item.keys():
            if item['title'].lower().strip() == name or item['original_title'].lower().strip() == name:
                return item
        elif 'name' in item.keys():
            if item['name'].lower().strip() == name or item['name'].lower().strip() == name:
                return item

    for item in info:
        if 'title' in item.keys():
            if name in item['title'].lower().strip() or name in item['original_title'].lower().strip():
                return item
        elif 'name' in item.keys():
            if name in [item['name'].lower().strip()]:
                return item
    return info[0]


def process_single(args, api_info, all_flag):
    if all_flag:
        info_list = api_info(args[1])['result']
        return args[1], [print_movie_info(info) for info in info_list]

    else:
        info = api_info(args[1])['result']
        info = get_mathed_result(info, args[1])
        if info == []:
            print('no result')
            return agrs[1], []
        if api_info == api.movie_get_person_info:
            return info['name'], print_actor_info(info)
        keys = list(info.keys())
        if args[2] not in keys and args[2] in confuse_word.keys():
            args[2] = confuse_word[args[2]]
        matched_key = args[2]
        print('info', matched_key, args[2], keys)
        if 'title' in info.keys():
            key = 'title'
        else:
            key = 'name'
        if matched_key in info.keys():
            return info[key], info[matched_key]
        else:
            return info[key], print_movie_info(info)


def parse_conds_3_tuple(text):
    conds = text.lower().split(',')
    movie_key, actor_key = conds[:2]
    conds = ','.join(conds[2:])
    movie_key = movie_key.strip().replace('"', '').replace("'", "").strip()
    actor_key = actor_key.strip().replace('"', '').replace("'", "").strip()
    conds = conds.strip()
    return movie_key, actor_key, conds


def parse_conds_2_tuple(text):
    conds = text.lower().split(',')
    movie_key = conds[0]
    conds = ','.join(conds[1:])
    movie_key = movie_key.strip().replace('"', '').replace("'", "").strip()
    conds = conds.strip()
    return movie_key, conds


def parse_sort(s):
    pattern = r'sort\((?P<str1>.*?),\s*(?P<str2>.*?)\)\s*\[(?P<str3>.*?)\]?'
    # 使用正则表达式进行匹配
    match = re.match(pattern, s)

    # 如果匹配成功，提取出捕获的组
    if match:
        return match.groupdict()
    else:
        return None  # 或者可以抛出一个异常


# %%
def print_actor_info(a):
    tmp = f"<actor> {a['name']}, the detailed information of he/she are: "
    for key in ['birthday']:
        if key in a.keys():
            tmp += f"the {key} is {a[key]}; "
    if 'acted_movies' in a.keys():
        tmp += f"he/she has played in {len('acted_movies')} movies:"
        for mid in a['acted_movies']:
            movie = api.movie_get_movie_info_by_id(mid)['result']
            if movie is not None:
                # print(movie)

                tmp += f" \'{movie['title']}\'({movie['release_date']});"
        tmp + '. '
    if 'oscar_awards' in a.keys():
        tmp += f"he/she get {len('oscar_awards')} Oscar nominations(wins): "
        for w in a['oscar_awards']:
            v = 'wined' if w['winner'] else 'nominated'
            tmp += f"she/he {v} {w['year_ceremony']} {w['category']} in movie \'{w['film']}\'; "
    tmp += '.</actor>\n'
    return tmp

oscar_map_dlc =  np.load('models/processed_data/oscar_map_dlc.npy',allow_pickle=True).tolist()
def parse_answer(commd):
    pairs = commd.lower().strip().replace('all get', 'get_all').replace('<|endoftext|>', '').replace('\nsort',
                                                                                                     ' sort').split(
        'get_')
    result = []
    claen_pairs = []
    result_str = []

    for pair in pairs:
        pair = pair.strip()
        try:
            if pair != '':
                print(pair)
                claen_pairs.append(pair)
                args = extract_parts(pair)
                print('args', args)
                for i in range(3):
                    args[i] = args[i].strip()
                all_flag = False
                if 'all' in args[0]:
                    all_flag = True
                    args[0] = args[0].replace('all_', '')
                if args is None:
                    print('error')
                elif args[0] == 'movie':
                    if not ('eq' in args[1] and ',' in args[1]):
                        args[1] = args[1].replace('(', '').replace(')', '').replace('none', '')
                        print('args1', args[1])

                        args[1], tmpresult = process_single(args, api.movie_get_movie_info, all_flag)
                        if all_flag:
                            tmpstr = f'the {len(tmpresult)}  films related to {args[1]} are :{tmpresult}'
                            result_str.append(tmpstr)
                        else:
                            result.append(tmpresult)
                            result_str.append(
                                f"the {args[2]} of {args[1]} is {result[-1]}"
                            )
                    else:
                        movie_key, conds = parse_conds_2_tuple(args[1])
                        print('conds', conds)
                        conds_map = clear_conds(conds)
                        print(conds_map)
                        if False and 'year' in conds_map.keys():
                            movie_list = api.movie_get_year_info(str(conds_map['year']))['result']['movie_list']
                            movie_list = [api.movie_get_movie_info_by_id(movie_id)['result'] for movie_id in
                                          tqdm(movie_list)]
                            movie_list = [movie for movie in movie_list if str(type(movie)) != "<class 'NoneType'>"]
                            genre_str = ''
                            if 'genre' in conds_map.keys() or 'genres' in conds_map.keys():
                                if 'genre' not in conds_map.keys():
                                    conds_map['genre'] = conds_map['genres']
                                new_movie_list = []
                                genre_str = conds_map['genre']
                                for movie in movie_list:
                                    genres = [remove_symbols(g['name'].lower()) for g in movie['genres']]
                                    print(genres)
                                    if remove_symbols(conds_map['genre']) in genres:
                                        new_movie_list.append(movie)
                                movie_list = new_movie_list
                            if 'sort' == args[3][0]:
                                sort_c, sort_key = args[3][1].replace(')', '').split(',')
                                sort_c = sort_c.strip()
                                sort_key = sort_key.strip()
                                assert sort_c == 'none'
                                if sort_key[0] == '-':
                                    reverse = True
                                    sort_key = sort_key[1:]
                                else:
                                    reverse = False
                                moive = sorted(movie_list, key=lambda x: x[sort_key], reverse=reverse)[0]
                                result.append(moive[args[3][2]])
                                result_str.append(
                                    f"the {args[3][2]} of {genre_str} movie sorted by {sort_key} in {conds_map['year']} is {moive[args[3][2]]}."
                                )

                            elif 'len' == args[2]:
                                result.append(len(movie_list))
                                result_str.append(
                                    f"the number of {genre_str} movies in {conds_map['year']} is {len(movie_list)}."
                                )

                            else:
                                print('to do')
                        else:
                            print('to do')
                elif args[0] == 'person':
                    if not ('[' in args[1] and ',' in args[1]):
                        args[1] = args[1].replace('(', '').replace(')', '').replace('none', '')
                        print('args1', args[1])
                        args[1], tmpresult = process_single(args, api.movie_get_person_info, all_flag)
                        result.append(tmpresult)
                        result_str.append(
                            tmpresult
                        )
                elif args[0] in ['movie_person_year_oscar', 'movie_person_oscar']:
                    movie_key, actor_key, conds = parse_conds_3_tuple(args[1])
                    conds_map = clear_conds(conds)
                    print('conds_map', conds_map)
                    if 'category' in conds_map.keys():
                        conds_map['category'] = conds_map['category'].replace('best ', '')
                    if movie_key.strip() == 'none' and actor_key.strip() == 'none' and 'year' in conds_map.keys():
                        if 'year' in conds_map.keys():
                            info_list = api.movie_get_year_info(conds_map['year'])['result']['oscar_awards']
                        else:
                            print('to do')
                        is_winner = 'nominee'
                        if 'winner' in conds_map.keys():
                            info_list = [info for info in info_list if info['winner'] == bool(conds_map['winner'])]
                            if bool(conds_map['winner']):
                                is_winner = 'winner'
                        if 'category' in conds_map.keys():
                            category_list = [info['category'].lower() for info in info_list]
                            if conds_map['category'] in Oscar_map.keys() and Oscar_map[
                                conds_map['category']] in category_list:
                                matched_index = category_list.index(Oscar_map[conds_map['category']])
                                if args[2] in info_list[matched_index].keys():
                                    result.append(info_list[matched_index][args[2]])
                                    result_str.append(
                                        f'The {args[2]} of the {conds_map["year"]} Oscar {is_winner} for best {conds_map["category"]} is (are) {result[-1]}. The details are {info_list[matched_index]}')
                                else:
                                    result.append(info_list[matched_index])
                                    result_str.append(
                                        f'The imformation of the {conds_map["year"]} Oscar {is_winner} for best {conds_map["category"]} is (are) {result[-1]}.')
                            else:
                                result.append(info_list)
                                tmpstr = ''
                                for info in info_list:
                                    if info['winner']:
                                        verb = 'wined'
                                    else:
                                        verb = 'nominated'
                                    tmpstr += f"{info['name']} {verb}  oscar best {info['category']}".lower()
                                    if 'film' in info.keys() and info['film'] is not None:
                                        tmpstr += f" in the movie: {info['film']}.\n ".lower()
                                    else:
                                        tmpstr += '.\n'.lower()
                                result_str.append(
                                    f'The imformation of the {conds_map["year"]} Oscar are: {tmpstr}')
                        else:
                            print('to do')
                    elif movie_key != 'none':
                        info = api.movie_get_movie_info(movie_key)['result']
                        info = get_mathed_result(info, movie_key)
                        if info == []:
                            print('no result')
                            continue
                        movie_key = info['title']
                        if 'oscar_awards' not in info.keys():
                            result_str.append(
                                f'{movie_key} don not have oscar awards')
                            continue
                        info = info['oscar_awards']
                        is_winner = 'nominee'
                        if 'winner' in conds_map.keys():
                            info = [item for item in info if item['winner'] == bool(conds_map['winner'])]
                            if bool(conds_map['winner']):
                                is_winner = 'winner'

                        matched_category = ''
                        if 'category' in conds_map.keys() and conds_map['category'] in Oscar_map.keys():
                            category_list = [info['category'].lower() for info in info]
                            matched_category = Oscar_map[conds_map['category']]
                            info = [item for item in info if item['category'].lower() == matched_category]

                        tmpstr = ''
                        for item in info:
                            if item['winner']:
                                verb = 'wined'
                            else:
                                verb = 'nominated'
                            tmpstr += f"{item['name']} {verb}  oscar best {item['category']}".lower()
                            if 'film' in item.keys() and item['film'] is not None:
                                tmpstr += f" in the movie: {item['film']}.\n ".lower()
                            else:
                                tmpstr += '.\n'.lower()

                        result.append(len(info))
                        result_str.append(
                            f'The number of the Oscar {matched_category} {is_winner}  about {movie_key} is  {result[-1]}. the detailed are {tmpstr}.')



                    elif actor_key != 'none':
                        info = api.movie_get_person_info(actor_key)['result'][0]['oscar_awards']
                        is_winner = 'nominee'
                        if 'winner' in conds_map.keys():
                            info = [item for item in info if item['winner'] == bool(conds_map['winner'])]
                            if bool(conds_map['winner']):
                                is_winner = 'winner'
                        matched_category = ''
                        if 'category' in conds_map.keys() and conds_map['category'] in Oscar_map.keys():
                            category_list = [info['category'].lower() for info in info]
                            matched_category = Oscar_map[conds_map['category']]
                            info = [item for item in info if item['category'].lower() == matched_category]

                        tmpstr = ''
                        for item in info:
                            if item['winner']:
                                verb = 'wined'
                            else:
                                verb = 'nominated'
                            tmpstr += f"{item['name']} {verb}  oscar best {item['category']}".lower()
                            if 'film' in item.keys() and item['film'] is not None:
                                tmpstr += f" in the movie: {item['film']}.\n ".lower()
                            else:
                                tmpstr += '.\n'.lower()

                        result.append(len(info))
                        result_str.append(
                            f'The number of the Oscar {matched_category} {is_winner}  about {actor_key} is  {result[-1]}. the detailed are {tmpstr}.')


                    else:
                        result_str.append('''the top 10 movie win most oscar wards are: Ben-Hur (1959): Directed by William Wyler, starring Charlton Heston, won 11 Oscars (Best Picture, Best Director, Best Actor, Best Supporting Actor, Best Cinematography, Best Art Direction, Best Costume Design, Best Sound, Best Film Editing, Best Special Effects, Best Music Score).

        Titanic (1997): Directed by James Cameron, starring Leonardo DiCaprio and Kate Winslet, won 11 Oscars (Best Picture, Best Director, Best Cinematography, Best Art Direction, Best Costume Design, Best Sound, Best Film Editing, Best Sound Effects Editing, Best Visual Effects, Best Original Song, Best Original Dramatic Score).

        The Lord of the Rings: The Return of the King (2003): Directed by Peter Jackson, starring Elijah Wood, won 11 Oscars (Best Picture, Best Director, Best Adapted Screenplay, Best Art Direction, Best Costume Design, Best Makeup, Best Original Score, Best Original Song, Best Sound Mixing, Best Film Editing, Best Visual Effects).

        West Side Story (1961): Directed by Robert Wise and Jerome Robbins, starring Natalie Wood and Richard Beymer, won 10 Oscars (Best Picture, Best Director, Best Supporting Actor, Best Supporting Actress, Best Cinematography, Best Art Direction, Best Costume Design, Best Sound, Best Film Editing, Best Score).

        Gigi (1958): Directed by Vincente Minnelli, starring Leslie Caron, won 9 Oscars (Best Picture, Best Director, Best Adapted Screenplay, Best Art Direction, Best Cinematography, Best Costume Design, Best Film Editing, Best Original Score, Best Song).

        The Last Emperor (1987): Directed by Bernardo Bertolucci, starring John Lone, won 9 Oscars (Best Picture, Best Director, Best Adapted Screenplay, Best Art Direction, Best Cinematography, Best Costume Design, Best Film Editing, Best Original Score, Best Sound).

        The English Patient (1996): Directed by Anthony Minghella, starring Ralph Fiennes and Juliette Binoche, won 9 Oscars (Best Picture, Best Director, Best Supporting Actress, Best Art Direction, Best Cinematography, Best Costume Design, Best Film Editing, Best Original Score, Best Sound).

        On the Waterfront (1954): Directed by Elia Kazan, starring Marlon Brando, won 8 Oscars (Best Picture, Best Director, Best Actor, Best Supporting Actress, Best Story and Screenplay, Best Art Direction, Best Cinematography, Best Film Editing).

        My Fair Lady (1964): Directed by George Cukor, starring Audrey Hepburn and Rex Harrison, won 8 Oscars (Best Picture, Best Director, Best Actor, Best Cinematography, Best Art Direction, Best Costume Design, Best Sound, Best Score).

        Gandhi (1982): Directed by Richard Attenborough, starring Ben Kingsley, won 8 Oscars (Best Picture, Best Director, Best Actor, Best Screenplay, Best Art Direction, Best Cinematography, Best Costume Design, Best Film Editing).\n\n 

        Here are the top 10 directors with the most Academy Awards, each described in one sentence:

        John Ford: Won 4 Oscars for Best Director (The Informer, The Grapes of Wrath, How Green Was My Valley, The Quiet Man).

        Frank Capra: Won 3 Oscars for Best Director (It Happened One Night, Mr. Deeds Goes to Town, You Can't Take It with You).

        William Wyler: Won 3 Oscars for Best Director (Mrs. Miniver, The Best Years of Our Lives, Ben-Hur).

        Steven Spielberg: Won 2 Oscars for Best Director (Schindler's List, Saving Private Ryan) and received the Irving G. Thalberg Memorial Award.

        Billy Wilder: Won 2 Oscars for Best Director (The Lost Weekend, The Apartment).

        Elia Kazan: Won 2 Oscars for Best Director (Gentleman's Agreement, On the Waterfront).

        Frank Lloyd: Won 2 Oscars for Best Director (The Divine Lady, Cavalcade).

        Joseph L. Mankiewicz: Won 2 Oscars for Best Director (A Letter to Three Wives, All About Eve).

        Robert Wise: Won 2 Oscars for Best Director (West Side Story, The Sound of Music).

        Alejandro G. Iñárritu: Won 2 Oscars for Best Director (Birdman, The Revenant).

        Here are the top 10 actors with the most Academy Awards, each described in one sentence:

        Katharine Hepburn: Won 4 Oscars for Best Actress (Morning Glory, Guess Who's Coming to Dinner, The Lion in Winter, On Golden Pond).

        Meryl Streep: Won 3 Oscars (Best Supporting Actress for Kramer vs. Kramer, Best Actress for Sophie's Choice and The Iron Lady).

        Jack Nicholson: Won 3 Oscars (Best Actor for One Flew Over the Cuckoo's Nest and As Good as It Gets, Best Supporting Actor for Terms of Endearment).

        Daniel Day-Lewis: Won 3 Oscars for Best Actor (My Left Foot, There Will Be Blood, Lincoln).

        Ingrid Bergman: Won 3 Oscars (Best Actress for Gaslight and Anastasia, Best Supporting Actress for Murder on the Orient Express).

        Walter Brennan: Won 3 Oscars for Best Supporting Actor (Come and Get It, Kentucky, The Westerner).

        Bette Davis: Nominated 10 times and won 2 Oscars for Best Actress (Dangerous, Jezebel).

        Spencer Tracy: Nominated 9 times and won 2 Oscars for Best Actor (Captains Courageous, Boys Town).

        Marlon Brando: Won 2 Oscars for Best Actor (On the Waterfront, The Godfather).

        Denzel Washington: Won 2 Oscars (Best Supporting Actor for Glory, Best Actor for Training Day).
        ''')
                elif args[0] == 'movie_person_cast':
                    movie_key, actor_key, conds = parse_conds_3_tuple(args[1])
                    conds_map = clear_conds(conds)
                    if len(conds_map.keys()) != 0:
                        conds_str = ', of which these(this) movies(person) meet: '
                        for key, value in conds_map.items():
                            conds_str += f'the {key} is {value}; '
                    else:
                        conds_str = ''
                    if actor_key != 'none':
                        if actor_key[0] == '[':
                            actor_key = args[1].split('[')[-1].split(']')[0]
                            moive_list_idx = None
                            for actor in actor_key.split(','):
                                info = api.movie_get_person_info(actor)['result'][0]
                                print(info)
                                if moive_list_idx is None:
                                    moive_list_idx = set(info['acted_movies'])
                                else:
                                    moive_list_idx = moive_list_idx & set(info['acted_movies'])
                            if args[2] == 'len':
                                result_str.append(
                                    f'the number of movies acted by {actor_key} are {len(moive_list_idx)}')
                                continue
                            moive_list = []
                            for movie_idx in moive_list_idx:
                                movie = api.movie_get_movie_info_by_id(movie_idx)['result']
                                if str(type(movie)) != "<class 'NoneType'>":
                                    moive_list.append(movie)
                                else:
                                    print('movie_idx', movie_idx)
                        else:
                            info = api.movie_get_person_info(actor_key)['result'][0]
                            moive_list = []
                            for movie_idx in info['acted_movies']:
                                movie = api.movie_get_movie_info_by_id(movie_idx)['result']
                                if str(type(movie)) != "<class 'NoneType'>":
                                    moive_list.append(movie)
                                else:
                                    print('error idx', movie_idx)
                        if movie_key != 'none':
                            moive_list = [m for m in moive_list if
                                          movie_key in m['title'].lower() or movie_key in m['original_title']]
                        if 'year' in conds_map.keys():
                            matched_moive_list = []
                            for movie in moive_list:
                                if movie['release_date'].split('-')[0] == str(conds_map['year']):
                                    matched_moive_list.append(movie)
                            moive_list = matched_moive_list
                        if 'ge(year' in conds_map.keys() or 'le(year' in conds_map.keys():
                            matched_moive_list = []
                            for movie in moive_list:
                                flag = 1
                                if 'ge(year' in conds_map.keys() and movie['release_date'].split('-')[0] >= str(
                                        conds_map['ge(year']):
                                    pass
                                else:
                                    flag = 0
                                if 'le(year' in conds_map.keys() and movie['release_date'].split('-')[0] <= str(
                                        conds_map['le(year']):
                                    pass
                                else:
                                    flag = 0
                                if flag:
                                    matched_moive_list.append(movie)
                            moive_list = matched_moive_list

                        if 'character' in conds_map.keys():
                            matched_moive_list = []
                            for movie in moive_list:
                                character = ""
                                for cast in movie['cast']:
                                    if remove_symbols(actor_key) in remove_symbols(
                                            cast['name'].lower().strip()) or remove_symbols(
                                        cast['name'].lower().strip()) in remove_symbols(actor_key):
                                        break
                                character = remove_symbols(cast['character'].lower())
                                if len(character) and (
                                        remove_symbols(
                                            conds_map['character']) in character or character in remove_symbols(
                                    conds_map['character'])):
                                    print(cast['character'], remove_symbols(conds_map['character']))
                                    matched_moive_list.append(movie)
                                    continue
                                character = remove_symbols(cast['name'].lower())
                                if len(character) and (
                                        remove_symbols(
                                            conds_map['character']) in character or character in remove_symbols(
                                    conds_map['character'])):
                                    print(cast['character'], remove_symbols(conds_map['character']))
                                    matched_moive_list.append(movie)
                                    continue
                            moive_list = matched_moive_list
                        if 'order' in conds_map.keys():
                            matched_moive_list = []
                            for movie in moive_list:
                                character = ""

                                for cast in movie['cast']:
                                    if remove_symbols(actor_key) in remove_symbols(
                                            cast['name'].lower().strip()) or remove_symbols(
                                        cast['name'].lower().strip()) in remove_symbols(actor_key):
                                        character = cast['order']
                                        break
                                if character != '' and str(character) == str(conds_map['order']):
                                    matched_moive_list.append(movie)
                                    continue
                            moive_list = matched_moive_list

                        if 'sort' in args[3][0]:
                            sort_c, sort_key = args[3][1].replace(')', '').split(',')
                            sort_c = sort_c.strip()
                            sort_key = sort_key.strip()
                            assert sort_c == 'none'
                            if sort_key[0] == '-':
                                reverse = True
                                sort_key = sort_key[1:]
                            else:
                                reverse = False
                            if sort_key in ['year']:
                                sort_key = 'release_date'
                            if sort_key in ['movie_name']:
                                sort_key = 'title'
                            moive_list = sorted(moive_list, key=lambda x: x[sort_key], reverse=reverse)
                            if args[3][2] in ['title', 'movie_name', 'name']:
                                args[3][2] = 'title'

                            if all_flag:
                                result.append([moive['title'] for moive in moive_list])
                                tmpstr = [print_movie_info(movie) for moive in moive_list]
                                result_str.append(
                                    f"the  movies acted by {actor_key} sorted by {sort_key}{conds_str} are: {tmptmpstr} "
                                )
                            else:
                                num = min(10, len(moive_list))
                                result.append([(movie['title'], movie[sort_key]) for movie in moive_list[:num]])
                                tmpstr = [print_movie_info(movie) for moive in moive_list[:num]]
                                result_str.append(
                                    f"the top {num} movie acted by {actor_key} sorted by {sort_key}{sort_key} are : {tmpstr}")

                        else:
                            result.append(len(moive_list))
                            tmpstr = ' '.join([print_movie_info(m) for m in moive_list])
                            result_str.append(
                                f"the number of movie acted by {actor_key}  is {result[-1]}{conds_str} they are {tmpstr}."
                            )

                    elif movie_key != 'none':
                        # print(info)
                        info = api.movie_get_movie_info(movie_key)['result']
                        info = get_mathed_result(info, movie_key)
                        if info == []:
                            print('no result info')
                            continue
                        movie_key = info['title']
                        if 'character' in conds_map.keys():
                            matched_character = []
                            for character in info['cast']:
                                target = remove_symbols(conds_map['character'])
                                if remove_symbols(character['character'].lower()) in target or target in remove_symbols(
                                        character['character'].lower()) or remove_symbols(
                                    character['name'].lower()) in target or target in remove_symbols(
                                    character['name'].lower()):
                                    matched_character.append(character)

                            if len(matched_character) > 1 and 'gender' in conds_map.keys():
                                matched_character = [c for c in matched_character if
                                                     str(c['gender']) == conds_map['gender']]
                            if len(matched_character) != 1:
                                print('error len(matched_character)', len(matched_character))

                            for ch in matched_character:
                                actor = api.movie_get_person_info_by_id(ch['id'])['result']

                                result.append(actor)
                                result_str.append(
                                    f" the information of the actor(actress) who acted {ch['character']} in {movie_key} is {print_actor_info(actor)}."
                                )

                        result.append([p['name'] for p in info['cast']])
                        tmpstr = ""
                        for p in info['cast'][:10]:
                            tmpstr += f'the {p["name"]} played {p["character"]};\n'
                        result_str.append(
                            f"the number of cast in {movie_key} is {len(info['cast'])}. the top 10 cast in {movie_key} sorted by order are: {tmpstr}"
                        )
                    else:
                        print('to do')
                elif args[0] == 'movie_person_crew':
                    movie_key, actor_key, conds = parse_conds_3_tuple(args[1])
                    conds_map = clear_conds(conds)
                    if len(conds_map.keys()) != 0:
                        conds_str = ', of which these(this) movies(person) meet: '
                        for key, value in conds_map.items():
                            if key in ['year', 'release_date']:
                                conds_str += f'the {key} is {value}; '
                    else:
                        conds_str = ''
                    if movie_key != 'none':
                        info = api.movie_get_movie_info(movie_key)['result']
                        info = get_mathed_result(info, movie_key)
                        if info == []:
                            print('no result')
                            continue
                        movie_key = info['title']
                        if 'job' in conds_map.keys():
                            matched_character = []
                            for character in info['crew']:
                                if character['job'].lower() == conds_map['job']:
                                    matched_character.append(character)
                            if matched_character == []:
                                tmp = f"{movie_key} has {len(info['crew'])} crews, and  they are: "
                                for crew in info['crew']:
                                    tmp += f"{crew['name']} serve as {crew['job']}; "
                                result_str.append(tmp)
                            elif args[2] == 'len':
                                result.append(len(matched_character))
                                result_str.append(
                                    f"the number of {conds_map['job']}  in {movie_key} is {len(matched_character)}"
                                )
                            elif args[2] in matched_character[0].keys():
                                result.append([character[args[2]] for character in matched_character])
                                result_str.append(
                                    f"the {conds_map['job']} {args[2]} of {movie_key} is {','.join([character[args[2]] for character in matched_character])}"
                                )
                            else:
                                tmpres = []
                                for character in matched_character:
                                    person = api.movie_get_person_info_by_id(character['id'])['result']
                                    tmp_str = ''
                                    if args[2] in person.keys():
                                        tmpres.append(person[args[2]])
                                        tmp_str = args[2]
                                    else:
                                        tmpres.append(person)
                                        tmp_str = 'information'
                                result.append(tmpres)
                                result_str.append(
                                    f"the {tmp_str} of crew in {movie_key} are {'; '.join(result[-1])}"
                                )

                        else:
                            print('to do')
                    elif actor_key != 'none':
                        info = api.movie_get_person_info(actor_key)['result'][0]

                        if 'job' in conds_map.keys() and conds_map['job'] == 'director':
                            directed_movies = [api.movie_get_movie_info_by_id(idx)['result'] for idx in
                                               info['directed_movies']]
                        elif 'directed_movies' in info.keys():
                            directed_movies = [api.movie_get_movie_info_by_id(idx)['result'] for idx in
                                               info['directed_movies']]
                        elif 'acted_movies' in info.keys():
                            directed_movies = [api.movie_get_movie_info_by_id(idx)['result'] for idx in
                                               info['acted_movies']]
                        else:
                            print('no result')
                        directed_movies = [movie for movie in directed_movies if
                                           str(type(movie)) != "<class 'NoneType'>"]
                        if 'year' in conds_map.keys() or 'release_date' in conds_map.keys():
                            if 'release_date' in conds_map.keys():
                                conds_map['year'] = conds_map['release_date']
                            directed_movies = [movie for movie in directed_movies if
                                               movie['release_date'][:4] == conds_map['year'][:4]]
                        if 'sort' in args[3][0]:
                            sort_c, sort_key = args[3][1].replace(')', '').split(',')
                            sort_c = sort_c.strip()
                            sort_key = sort_key.strip()
                            assert sort_c == 'none'
                            if sort_key[0] == '-':
                                reverse = True
                                sort_key = sort_key[1:]
                            else:
                                reverse = False
                            if sort_key in ['year']:
                                sort_key = 'release_date'
                            directed_movies = sorted(directed_movies, key=lambda x: x[sort_key], reverse=reverse)
                            if args[3][2] in ['title', 'movie_name', 'name', 'none']:
                                args[3][2] = 'title'

                            num = min(10, len(directed_movies))
                            # print(directed_movies)
                            result.append([movie['title'] for movie in directed_movies[:num]])
                            tmpstr = ' '.join([print_movie_info(movie) for movie in directed_movies[:num]])
                            result_str.append(
                                f"the top {num} movies sorted by {sort_key} participated by {actor_key} are: {tmpstr}."
                            )
                        else:
                            result.append(len(directed_movies))
                            result.append([movie['title'] for movie in directed_movies[:]])
                            tmpstr = ' '.join([print_movie_info(movie) for movie in directed_movies[:]])
                            result_str.append(
                                f"the number of movies participated by {actor_key} is: {len(directed_movies)}{conds_str} they are: {tmpstr}")
                    else:
                        print('to do3')
                else:
                    print('2 to do')
        except:
            continue

    return result, result_str


Oscar_map = {'music (original score)': 'music (original score)',
             'visual effects': 'visual effects',
             'animated feature film': 'animated feature film',
             'foreign language film': 'foreign language film',
             'directing': 'directing',
             'director': 'directing',
             'makeup and hairstyling': 'makeup and hairstyling',
             'music (original song)': 'music (original song)',
             'picture': 'best picture',
             'sound mixing': 'sound mixing',
             'writing (adapted screenplay)': 'writing (adapted screenplay)',
             'writing (original screenplay)': 'writing (original screenplay)',
             'special award': 'special award',
             'actor in a supporting role': 'actor in a supporting role',
             'documentary (short subject)': 'documentary (short subject)',
             'cinematography': 'cinematography',
             'actor in a leading role': 'actor in a leading role',
             'costume design': 'costume design',
             'actress in a supporting role': 'actress in a supporting role',
             'honorary award': 'honorary award',
             'production design': 'production design',
             'actress in a leading role': 'actress in a leading role',
             'short film (live action)': 'short film (live action)',
             'short film (animated)': 'short film (animated)',
             'sound editing': 'sound editing',
             'film editing': 'film editing',
             'documentary (feature)': 'documentary (feature)',
             'feature': 'documentary (feature)',
             'actor': 'actor in a leading role',
             'actress': 'actress in a leading role',
             'supporting actor': 'actor in a supporting role',
             'supporting actress': 'actress in a supporting role',
             'animated feature film': 'animated feature film',
             'documentary feature': 'documentary (feature)',
             'visual effect': 'visual effects',
             'film': 'best picture',
             'original score': 'music (original score)',
             'original song': 'music (original song)',
             'music (original score)': 'music (original score)',
             'music (original song)': 'music (original song)',
             }

finance_confuse_word = {
    'close': 'Close',
    'open': 'Open',
    'low': 'Low',
    'high': 'High',
    'volume': 'Volume',
}

def parse_time(time_str):
    # 假设所有时间都是EST时区，因此我们可以直接去掉EST部分进行解析
    return datetime.strptime(time_str[:-4], '%Y-%m-%d %H:%M:%S')
def find_nearest_time(times, query_time_str):
    query_time = parse_time(query_time_str)
    nearest_time = None
    for time_str in times:
        time = parse_time(time_str)
        if time <= query_time and (nearest_time is None or time > nearest_time):
            nearest_time = time
    if nearest_time:
        nearest_time_str = nearest_time.strftime('%Y-%m-%d %H:%M:%S') + ' EST'
        return nearest_time_str
    else:
        return None


def convert_pt_to_est(pt_time_str, pt_format='%m/%d/%Y, %H:%M:%S'):
    # 解析PT时间字符串为datetime对象（没有时区信息）
    pt_time = datetime.strptime(pt_time_str, pt_format)

    # 将PT时间加上3小时以模拟时区差异（未考虑夏令时）
    est_time = pt_time + timedelta(hours=3)

    # 处理日期变更（如果有必要）
    # 注意：这里简化处理，仅检查小时数是否小于原始PT小时数
    if est_time.hour < pt_time.hour:
        est_time -= timedelta(days=1)

        # 格式化EST时间字符串并返回
    est_time_str = est_time.strftime('%Y-%m-%d %H:%M:%S') + ' EST'
    return est_time_str


def finance_parse_answer(cmd, query_time):
    m, d, y = query_time.split(',')[0].split('/')
    nowtime = ''.join([y, m, d])
    pairs = cmd.replace('ALL get', 'get_all').replace('<|endoftext|>', '').replace('\nsort', ' sort').split('get_')
    results = []
    claen_pairs = []
    all_result_str = []
    for pair in pairs[:]:
        try:
            pair = pair.strip()
            result = []
            if pair not in ['', 'None', 'none']:
                print(pair, nowtime)
            else:
                continue
            claen_pairs.append(pair)
            args = extract_parts(pair)
            print(args)
            all_flag = False
            result_str_map = {}
            result_str = ""
            if 'all' in args[0]:
                all_flag = True
                args[0] = args[0].replace('all_', '')
            if args[0] == 'stock_price':
                stock_key, conds = parse_conds_2_tuple(args[1])
                stock_key = stock_key.upper().replace(')','')
                print(stock_key)

                result_str_map['func'] = args[0]
                result_str_map['entity'] = stock_key

                if conds in ['most recent)', 'latest)']:
                    price_history = finance_get_result(api.finance_get_detailed_price_history, stock_key)
                    est_query_time = convert_pt_to_est(query_time[:-3])
                    findkey = find_nearest_time(list(price_history.keys()), est_query_time)
                    if findkey is None:
                        print('no res')
                        continue
                    result = round(price_history[findkey]['Open'], 2)
                    result_str = f'The stock_price of {stock_key} most recent ({findkey}) is {result}.'
                    all_result_str.append(result_str)
                    continue

                price_history = finance_get_result(api.finance_get_price_history, stock_key)
                if price_history is None:
                    print('error')
                conds_time = clear_time_conds(conds)
                print('before clear', conds_time, list(price_history.keys())[:3])
                new_price_history = {}
                for key in price_history.keys():
                    new_price_history[clear_time_format(key)] = {k:round(v,5) for k,v in price_history[key].items()}
                conds_time = clear_text_conds_time(conds_time, list(new_price_history.keys()), nowtime)
                print('after clear', conds_time)
                cond_flag = 0
                tmpstr=''
                if len(conds_time) == 1:
                    if conds_time[0] not in new_price_history.keys():
                        tmpstr = f'There is no record of {stock_key} stock_price in {conds_time[0]}. but '
                        conds_time[0] = str(lower_bound(new_price_history, conds_time[0]))
                    print('conds_time', conds_time[0])
                    result = {conds_time[0]: new_price_history[conds_time[0]]}
                    result_str_map['time'] = conds_time[0]
                    result_str = tmpstr+f'the stock_price of {stock_key} in {conds_time[0]} is {new_price_history[conds_time[0]]}.'
                    cond_flag = 0
                elif len(conds_time) == 2:
                    result = {key: new_price_history[key] for key in new_price_history.keys() if
                              key >= conds_time[0] and key <= conds_time[1]}
                    result_str_map['time'] = [conds_time[0], conds_time[1]]
                    result_str = f'the stock_price of {stock_key} between {conds_time[0]} and {conds_time[1]} are {result}.'
                    cond_flag = 1
                else:
                    print('error')
                    result = {}
                if len(result) == 0:
                    print('error')

                if args[2] != 'none' and 'ge' not in args[2]:
                    if args[2] in finance_confuse_word.keys():
                        args[2] = finance_confuse_word[args[2]]
                    result_str_map['attr'] = args[2]
                    result = {key: round(result[key][args[2]], 3) for key in result.keys()}
                    if cond_flag:

                        result_str = f'the {args[2]} stock_price of {stock_key} between {conds_time[0]} and {conds_time[1]} are (The format is (date, amount)):{result}, the average stock_price is {sum([v for v in result.values()])/len(result)}.'
                    else:
                        result_str = tmpstr+f'the {args[2]} stock_price of {stock_key} in {conds_time[0]} is {result[conds_time[0]]}.'
                    if 'sort' in args[3]:
                        sort_cons = args[3][1]
                        if ',' in sort_cons:
                            sort_key = sort_cons.split(',')[1].replace(')', '').strip()
                        else:
                            sort_key = sort_cons.replace(')', '').strip()
                        if sort_key[0] == '-':
                            reverse = True
                            sort_key = sort_key[1:]
                        else:
                            reverse = False
                        sort_key = finance_confuse_word[sort_key]
                        resultidx = sorted(result.keys(), key=lambda x: result[x], reverse=reverse)[0]
                        result = {resultidx: result[resultidx]}
                        if cond_flag:
                            result_str = f'the {args[2]} stock_price sorted by {sort_key} of {stock_key} between {conds_time[0]} and {conds_time[1]} are (The format is (date, amount)):{result}.'
                        else:
                            result_str = f'the {args[2]} stock_price of {stock_key} sorted by {sort_key} in {conds_time[0]} is {result}.'
                elif 'sort' in args[3]:
                    print(args[3])
                    sort_cons = args[3][1]
                    if 'ge' in args[2]:
                        sort_cons = args[2]
                    flag = [1 for key in finance_confuse_word.keys() if key in sort_cons]
                    if sum(flag):
                        # 'open' in sort_cons =='ge(open,close),time)
                        left_key, right_key = sort_cons.split(',')[:2]
                        right_key = finance_confuse_word[right_key.replace(')', '').strip()]
                        print("left_key,right_key", left_key, right_key)
                        ge_lael = '>'
                        if 'ge' in left_key:
                            left_key = finance_confuse_word[left_key.replace('ge(', '').strip()]
                            result = {key: result[key] for key in result.keys() if
                                      result[key][left_key] > result[key][right_key]}
                            ge_lael = '>'
                        elif 'le' in left_key:
                            left_key = finance_confuse_word[left_key.replace('le(', '').strip()]
                            result = {key: result[key] for key in result.keys() if
                                      result[key][left_key] < result[key][right_key]}
                            ge_lael = '<'
                        elif 'eq' in left_key:
                            left_key = finance_confuse_word[left_key.replace('eq(', '').strip()]
                            result = {key: result[key] for key in result.keys() if
                                      result[key][left_key] == result[key][right_key]}
                            ge_lael = '='
                        else:
                            print('ge to do')

                        result_str = f' the date when {stock_key} {left_key} price {ge_lael} {right_key} price are {[ k for k in result.keys()]}'+ f' The details are: {result}'

                    else:
                        print('flag to do')
                else:
                    print('no args to do')
            elif args[0] == 'stock_pe_ratio':
                stock_key = args[1].replace(')', '').strip().upper()
                print(stock_key)
                pe_ratio = finance_get_result(api.finance_get_pe_ratio, stock_key)
                if pe_ratio is None:
                    print('error')
                result = round(pe_ratio, 3)
                result_str = f'the stock_pe_ratio of {stock_key} is {result}.'
            elif args[0] == 'stock_market_cap':
                stock_key = args[1].replace(')', '').strip().upper()
                print(stock_key)
                market_capitalization = finance_get_result(api.finance_get_market_capitalization, stock_key)

                result = round(market_capitalization, 3)
                result_str = f'the stock_market_cap of {stock_key} is {result}.'
            elif args[0] == 'stock_dividend':
                stock_key = args[1].replace(')', '').strip().upper()
                print('stock_dividend', stock_key)
                stock_dividend = finance_get_result(api.finance_get_dividends_history, stock_key)
                # print(stock_dividend.keys())
                if stock_dividend is None:
                    print('error')
                new_stock_dividend = {}
                for key in stock_dividend.keys():
                    new_stock_dividend[clear_time_format(key)] = round(stock_dividend[key], 3)
                result = sorted(new_stock_dividend.items(), key=lambda x: x[0])
                result_str = f'{len(result)} records about stock_dividend of {stock_key} sorted by date are (The format is (date, amount)): {result}. '
                if 'ge' in args[2] and 'le' in args[2]:
                    _, t1, _, t2 = clear_time_conds(args[2])
                    result = [key for key in result if key[0] >= t1 and key[0] <= t2]
                    result_str = f'{len(result)} records about stock_dividend of {stock_key} between {t1} and {t2} are (The format is (date, amount)): {result}. '
                if 'sort' in args[3]:
                    if'ge(date' in args[3][1] and 'le(date' in args[3][1]:
                        _, t1, _, t2,_ = clear_time_conds(args[3][1])
                        t2 = t2.replace(']','')
                        result = [key for key in result if key[0] >= t1 and key[0] <= t2]
                        result_str = f'{len(result)} records about stock_dividend of {stock_key} between {t1} and {t2} are (The format is (date, amount)): {result}. '
                    elif 'ge(date' in args[3][1] or 'le(date' in args[3][1]:
                        op, t1, _ = clear_time_conds(args[3][1])
                        t1 = t1.replace(']','')
                        if 'ge' in op:
                            result = [key for key in result if key[0] >= t1]
                        else:
                            result = [key for key in result if key[0] <= t1]
                        result_str = f'{len(result)} records about stock_dividend of {stock_key} that {op} {t1}   are (The format is (date, amount)): {result}. '
            elif args[0] =='stock_eps':
                stock_key = args[1].replace(')', '').strip().upper()
                print('stock_eps', stock_key)
                eps = finance_get_result(api.finance_get_eps, stock_key)
                result = round(eps, 4)
                result_str = f'the stock_eps of {stock_key} is {result}.'
            else:
                print('to do func')
            if result in [[], {}]:
                error = 1
            results.append(result)
            if result_str != "":
                all_result_str.append(result_str)
        except:
            continue
    return results, all_result_str


def lower_bound(lst, val):
    lst = [int(i) for i in sorted(list(lst.keys()))]
    val = int(val)
    left, right = 0, len(lst)
    while left < right:
        mid = (left + right) // 2
        if lst[mid] < val:
            left = mid + 1
        else:
            right = mid
    if left >= len(lst):
        return lst[-1]
    if left == len(lst) or lst[left] >= val:
        return lst[left]
    else:
        return lst[left + 1]


def finance_get_result(fun, key):
    tmpres = fun(key)['result']
    if tmpres is None or tmpres == {}:
        try:
            name = api.finance_get_company_name(key)['result'][0]
        except:
            name = api.finance_get_info(key)['result']['longName']
        print('name', name)
        ticker = api.finance_get_ticker_by_name(name)['result']
        print('ticker', ticker)
        if ticker is not None:
            tmpres = fun(ticker)['result']
            return tmpres
        print('no result')
        return None
    else:
        return tmpres


def is_valid_date_format(date_string):
    pattern = r'^\d{4}\d{2}\d{2}$'
    return bool(re.match(pattern, date_string))


def clear_text_conds_time(conds_time, history, nowtime):
    for i in range(len(conds_time)):
        time = conds_time[i]
        if not is_valid_date_format(time):
            if time == 'yesterday':
                conds_time[i] = get_yesterday(nowtime)
            elif time =='day before yesterday':
                conds_time[i] = get_yesterday(nowtime,2)
            elif time == 'today':
                conds_time[i] = nowtime
            elif time in ['recent trading day', 'last trading day','most recent day'] or 'recent day' in time or 'last day' in time:
                conds_time[i] = find_closest_date(history, nowtime)
            elif time in ['ipo']:
                conds_time[i] = find_early_date(history)
            else:
                method, day = time.split(' ')
                method = finance_method_map[method]
                day = finance_data_map[day]
                conds_time[i] = method(nowtime, day)
    return conds_time


def find_early_date(date_list):
    # 将字符串日期转换为 datetime 对象
    # print(date_list)
    dates = [datetime.strptime(date, "%Y%m%d") for date in date_list]
    # query_date = datetime.strptime(query_date, "%Y%m%d")

    # 对日期列表进行排序
    dates.sort()
    return dates[0].strftime("%Y%m%d")


def find_closest_date(date_list, query_date):
    # 将字符串日期转换为 datetime 对象
    # print(date_list)
    dates = [datetime.strptime(date, "%Y%m%d") for date in date_list]
    query_date = datetime.strptime(query_date, "%Y%m%d")

    # 对日期列表进行排序
    dates.sort()

    # 初始化最接近日期
    closest_date = None

    # 查找小于查询日期的最大日期
    for date in dates:
        if date < query_date:
            closest_date = date
        else:
            break

    # 如果找到符合条件的日期，返回其字符串形式；否则返回 None
    if closest_date:
        return closest_date.strftime('%Y%m%d')
    else:
        return None


def clear_time_conds(conds):
    conds_time = []
    if 'none' in conds:
        return conds_time
    conds_time = [cond.replace(')', '').replace('(', '').replace('-', '').strip() for cond in conds.split(',')]
    return conds_time


def get_ticker(stock_key):
    company_name = api.finance_get_company_name(stock_key)['result'][0]
    ticker = api.finance_get_ticker_by_name(company_name)['result'][0]
    return ticker


def clear_time_format(key):
    # '2023-07-10 00:00:00 EST'
    key = key.split(' 00:00:00 EST')[0]
    key = key.replace('-', '').strip()
    return key


def get_yesterday(date_string,num=1):
    date_obj = datetime.strptime(date_string, "%Y%m%d")
    yesterday = date_obj - timedelta(num)
    return yesterday.strftime("%Y%m%d")


def get_recent_week_day(date_string, weekday_num):
    date_obj = datetime.strptime(date_string, "%Y%m%d")
    if date_obj.weekday() >= weekday_num - 1:
        offset = date_obj.weekday() - weekday_num + 1
        previous_week_day = date_obj - timedelta(days=offset)
        return previous_week_day.strftime("%Y%m%d")
    else:
        return get_last_week_day(date_string, weekday_num)


def get_last_week_day(date_string, weekday_num):
    date_obj = datetime.strptime(date_string, "%Y%m%d")
    offset = date_obj.weekday() + (7 - weekday_num + 1)
    previous_week_day = date_obj - timedelta(days=offset)

    return previous_week_day.strftime("%Y%m%d")


def print_singers_info(members):
    tmpstr = '<singer>'
    for name in members:
        lifespan = api.music_get_lifespan(name)['result']
        tmpstr += f' {name}'
        if lifespan == [None, None]:
            tmpstr += ';'
        else:
            tmpstr += f', his/her lifespan is begin in {lifespan[0]} and end in {lifespan[1]};'
    tmpstr += '</singer>'
    return tmpstr


def get_best_matched(info, name):
    for item in info:
        if remove_symbols(item.lower()) == remove_symbols(name):
            return item
    for item in info:
        if remove_symbols(name) in remove_symbols(item.lower()):
            return item
    return None


def print_songs_info(songs):
    tmpstr = '<music>'
    for song in songs:
        year = api.music_get_song_release_date(song)['result']
        if year is None:
            year = 'none'
        coutry = api.music_get_song_release_country(song)['result']
        if coutry is None:
            year = 'none'
        tmpstr += f" {song} ({year},{coutry});"

    tmpstr += '</music>'
    return tmpstr

music_grammy_map =  np.load('models/processed_data/grammy.npy',allow_pickle=True).tolist()

def music_parse_answer(cmd):
    pairs = cmd.strip().replace('ALL get', 'get_all').replace('<|endoftext|>', '').replace('\nsort', ' sort').replace(
        'all sort', 'sort').split('get_')
    result = []
    claen_pairs = []
    all_result_str = []
    for pair in pairs:
        pair = pair.strip()
        result_str = ""
        try:
            if pair != '' and pair != 'none':
                print(pair)
                claen_pairs.append(pair)
                args = extract_parts(pair)
                if 'year' in args[3][1]:
                    args[3][1] = args[3][1].replace('last year', '2023').replace('last_year', '2023')
                    args[3][1] = args[3][1].replace('this year', '2024').replace('this_year', '2024')
                print('args', args)
                all_flag = False
                if 'all' in args[0]:
                    all_flag = True
                    args[0] = args[0].replace('all_', '')
                if args is None:
                    print('error')
                if args[0] == 'person':
                    person_name = args[1].replace(')', '')
                    person_name = get_best_matched(api.music_search_artist_entity_by_name(person_name)['result'],
                                                   person_name)
                    if person_name is None:
                        print('no result')
                        continue
                    if 'music' in args[2]:
                        music = api.music_get_artist_all_works(person_name)['result']
                        if music is None:
                            continue
                        if 'sort' == args[3][0]:
                            if 'eq(release_date' in args[3][1]:
                                year = args[3][1].split('eq(release_date,')[1][:4]

                                release_date = [(m, api.music_get_song_release_date(m)['result']) for m in music if
                                                api.music_get_song_release_date(m)['result'] is not None and
                                                api.music_get_song_release_date(m)['result'][:4] == year]
                                tmpstr = print_songs_info([m for m, y in release_date])
                                result_str = f'the {len(release_date)} songs palyed by {person_name} released in {year} are (the format is (name,date): {tmpstr}.'
                            else:
                                if 'ge(release_date' in args[3][1] and 'le(release_date' in args[3][1]:
                                    geyeara = args[3][1].split('ge(release_date,')[-1][:4]
                                    leyeara = args[3][1].split('le(release_date,')[-1][:4]
                                    music = [m for m in music if
                                             api.music_get_song_release_date(m)['result'] is not None and
                                             api.music_get_song_release_date(m)['result'][:4] >= geyeara and
                                             api.music_get_song_release_date(m)['result'][:4] <= leyeara]
                                    sort_c = f', where the release date in [{geyeara},{leyeara}].'
                                else:
                                    sort_c = ''
                                sort_key = args[3][1].replace(')', '').split(',')[-1]
                                sort_key = sort_key.strip()
                                if sort_key[0] == '-':
                                    reverse = True
                                    sort_key = sort_key[1:]
                                else:
                                    reverse = False
                                if 'release_date' in sort_key:
                                    release_date = [(m, api.music_get_song_release_date(m)['result']) for m in music if
                                                    api.music_get_song_release_date(m)['result'] is not None]
                                release_date = sorted(release_date, key=lambda x: x[1], reverse=reverse)[:10]
                                tmpstr = print_songs_info([m for m, y in release_date])
                                result_str = f'{person_name} have {len(music)} songs{sort_c}, the top {len(release_date)} songs sorted by release_date are (format is (name,data,country)): "{tmpstr}".'
                        else:
                            result_str = f'{person_name} have {len(music)} songs and albums, the detail are {print_songs_info(music)}.'
                    elif 'lifespan' in args[2]:
                        lifespan = api.music_get_lifespan(person_name)['result']
                        if lifespan == [None, None]:
                            print('empty lifespan')
                        result_str = f'the lifespan of {person_name} is begin in {lifespan[0]} and end in {lifespan[1]}.'
                    elif 'members' in args[2]:
                        members = api.music_get_members(person_name)['result']
                        if members is not None and len(members):
                            result_str = f'{person_name}  had a total of {len(members)} members(bands): {print_singers_info(members)}'
                    elif 'birth_place' in args[2]:
                        birth_place = api.music_get_artist_birth_place(person_name)['result']
                        result_str = f'the birth_place of {person_name} is {birth_place}.'
                    elif 'birth_date' in args[2]:
                        birth_date = api.music_get_artist_birth_date(person_name)['result']
                        result_str = f'the birth_date of {person_name} is {birth_date}.'
                    else:
                        print('todo 2')
                elif args[0] == 'song':
                    song_name = args[1].replace(')', '')
                    song_name = get_best_matched(api.music_search_song_entity_by_name(song_name)['result'], song_name)
                    if song_name is None:
                        print('no result')
                        continue
                    print('song_name', song_name)
                    if 'release_date' in args[2]:
                        release_date = api.music_get_song_release_date(song_name)['result']
                        if release_date is not None:
                            result_str = f'the release_date of {song_name} is {release_date}.'
                    elif 'author' in args[2]:
                        author = api.music_get_song_author(song_name)['result']
                        if author is not None:
                            result_str = f'the author of {song_name} is {author}.'
                    else:
                        print('to do')
                elif args[0] == 'grammy_person':
                    person_name = args[1].replace(')', '')
                    person_name = get_best_matched(api.music_search_artist_entity_by_name(person_name)['result'],
                                                   person_name)
                    if person_name is None:
                        print('no result')
                        continue

                    award_date = api.music_grammy_get_award_date_by_artist(person_name)['result']
                    award_count = api.music_grammy_get_award_count_by_artist(person_name)['result']
                    result_str = f'the grammy award count of {person_name} until now is {award_count}, the award date are  {award_date}.'

                elif args[0] == 'grammy_year':
                    year = args[1].replace(')', '')
                    artist = api.music_grammy_get_best_artist_by_year(year)['result'][0]
                    song = api.music_grammy_get_best_song_by_year(year)['result'][0]
                    album = api.music_grammy_get_best_album_by_year(year)['result'][0]
                    album_auhor = api.music_get_song_author(album)['result']
                    album_auhor = album_auhor if album_auhor is not None else 'unknonwn'
                    song_auhor = api.music_get_song_author(song)['result']
                    song_auhor = song_auhor if song_auhor is not None else 'unknonwn'
                    year = int(year)
                    if year in music_grammy_map.keys():
                        result_str = music_grammy_map[year]
                    else:
                        result_str = f'''the grammy best New artist in {year} is {artist}.  the best grammy song in {year} is {song}, sung by {song_auhor}. the best grammy album in {year} is {album}, made by {album_auhor}. '''
                elif args[0] == 'grammy_song':
                    song_name = args[1].replace(')', '')
                    song_name = get_best_matched(api.music_search_song_entity_by_name(song_name)['result'], song_name)
                    if song_name is None:
                        print('no result')
                        continue
                    if 'award_count' in args[2]:
                        award_count = api.music_grammy_get_award_count_by_song(song_name)['result']
                        if award_count is not None:
                            result_str = f'the song {song_name} get grammy award {award_count} times.'
                else:
                    print('to do')
            if result_str != "":
                all_result_str.append(result_str)
        except:
            continue
    return all_result_str


# 示例


def open_parse_answer(cmd):
    try:
        pairs = cmd.strip().replace('ALL get', 'get_all').replace('<|endoftext|>', '').replace('\nsort',
                                                                                               ' sort').replace(
            'all sort', 'sort').split('get_')
        result = []
        claen_pairs = []
        all_result_str = []
        for pair in pairs:
            pair = pair.strip()
            result_str = ""
            if pair != '' and pair != 'none':
                claen_pairs.append(pair)
                args = extract_parts(pair)
                print('args', args)
                if args[0] == 'open':
                    name = args[1][:-1]
                    print('ori name', name)
                    info = api.open_search_entity_by_name(name)['result']
                    if len(info) == 0:
                        print('no entity')
                        continue
                    print(info)
                    name = get_best_matched(info, name)
                    print('new name', name)
                    if name is None:
                        print('no entity')
                        continue
                    res = api.open_get_entity(name)['result']
                    result_str = f"the summary text of {name}: {res['summary_text']}"
                    tmp_str = f'\n\n The detail of {name} are :'
                    for key, value in res['summary_structured'].items():
                        value = value.replace('[[', '').replace(']]', ',').replace('|', ' ')
                        if value in ['', None] or key in ['website', 'associated_acts', 'coordinates', 'mottoeng']:
                            continue
                        if 'image' in key:
                            continue
                        if 'logo' in key:
                            continue
                        if '<ref' in value:
                            value = value.split('<ref')[0]
                        tmp_str += f'the {key} is {value}.\n'
                    result_str += tmp_str
                else:
                    print('error')
            if result_str != '':
                all_result_str.append(result_str)
    except:
        all_result_str = []
    return all_result_str


def generate_dates(input_date):
    # Determine if the input is year only or year-month
    if len(input_date) == 4:  # Year only
        year = int(input_date)
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
    elif len(input_date) == 7:  # Year and month
        year, month = map(int, input_date.split('-'))
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year, 12, 31)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)
    else:
        return "Invalid input format. Please use 'yyyy' or 'yyyy-mm'."

    # Generate all dates in the range
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return dates


# Example usage

nba_teams = {
    'Heat': 'Miami Heat',  # 迈阿密热火队
    'Lakers': 'Los Angeles Lakers',  # 洛杉矶湖人队
    'Celtics': 'Boston Celtics',  # 波士顿凯尔特人队
    'Bulls': 'Chicago Bulls',  # 芝加哥公牛队
    'Rockets': 'Houston Rockets',  # 休斯敦火箭队
    'Knicks': 'New York Knicks',  # 纽约尼克斯队
    'Nets': 'Brooklyn Nets',  # 布鲁克林篮网队（原名New Jersey Nets）
    '76ers': 'Philadelphia 76ers',  # 费城76人队
    'Clippers': 'Los Angeles Clippers',  # 洛杉矶快船队
    'Spurs': 'San Antonio Spurs',  # 圣安东尼奥马刺队
    'Thunder': 'Oklahoma City Thunder',  # 俄克拉荷马城雷霆队
    'Warriors': 'Golden State Warriors',  # 金州勇士队
    'Raptors': 'Toronto Raptors',  # 多伦多猛龙队
    'Suns': 'Phoenix Suns',  # 菲尼克斯太阳队
    'Spurs': 'San Antonio Spurs',  # 圣安东尼奥马刺队（与上面的Spurs重复，但确保包含）
    'Thunder': 'Oklahoma City Thunder',  # 俄克拉荷马城雷霆队（与上面的Thunder重复，但确保包含）
    'Mavericks': 'Dallas Mavericks',  # 达拉斯独行侠队（原名Dallas Mavericks）
    'Grizzlies': 'Memphis Grizzlies',  # 孟菲斯灰熊队
    'Kings': 'Sacramento Kings',  # 萨克拉门托国王队
    'Jazz': 'Utah Jazz',  # 犹他爵士队
    'Nuggets': 'Denver Nuggets',  # 丹佛掘金队
    'Timberwolves': 'Minnesota Timberwolves',  # 明尼苏达森林狼队
    'Trail Blazers': 'Portland Trail Blazers',  # 波特兰开拓者队
    'Pelicans': 'New Orleans Pelicans',  # 新奥尔良鹈鹕队
    'Pistons': 'Detroit Pistons',  # 底特律活塞队
    'Cavaliers': 'Cleveland Cavaliers',  # 克里夫兰骑士队
    'Wizards': 'Washington Wizards',  # 华盛顿奇才队
    'Magic': 'Orlando Magic',  # 奥兰多魔术队
    'Hawks': 'Atlanta Hawks',  # 亚特兰大老鹰队
    'Hornets': 'Charlotte Hornets',  # 夏洛特黄蜂队
    'Pacers': 'Indiana Pacers',  # 印第安纳步行者队
    'Bucks': 'Milwaukee Bucks',  # 密尔沃基雄鹿队
    'Raptors': 'Toronto Raptors'  # 多伦多猛龙队（与上面的Raptors重复，但确保包含）
}


def contains_alpha(s):
    return any(char.isalpha() for char in s)


def check_score_name(name):
    for year in range(2024, 2000, -1):
        result = api.sports_soccer_get_games_on_date(str(year), name)['result']
        if result is not None:
            return True
    return False


def check_nba_name(name):
    for year in range(2024, 2000, -1):
        result = api.sports_nba_get_games_on_date(str(year), name)['result']
        if result is not None:
            return True
    return False


def calculate_date(date_str, days):
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    new_date_obj = date_obj + timedelta(days=days)
    new_date_str = new_date_obj.strftime('%Y-%m-%d')
    return new_date_str


def process_nba_result(result, name, date, oppo):
    season_type = ['ALL games', 'All Star', 'Playoffs', 'Pre Season', 'Regular Season']
    print(result.keys())
    tmpresult = {key: [] for key in season_type}
    for i in range(len(result['season_id'])):
        tmpresult[result['season_type'][str(i)].replace('-', ' ')].append(str(i))
    result_str = f"There are {len(result['season_id'])} games played by {name} during {date}, "
    detail_infos = [
        ('fgm', 'Field Goal Made'),
        ('fga', 'Field Goal Attempt'),
        ('fg_pct', 'Field Goal Percentage'),
        ('fg3m', '3-Point Field Goal Made'),  # 假设这是三分球命中数
        ('fg3a', '3-Point Field Goal Attempt'),  # 假设这是三分球尝试数
        ('fg3_pct', '3-Point Field Goal Percentage'),  # 假设这是三分球命中率
        ('ftm', 'Free Throw Made'),
        ('fta', 'Free Throw Attempt'),
        ('ft_pct', 'Free Throw Percentage'),
        ('oreb', 'Offensive Rebound'),
        ('dreb', 'Defensive Rebound'),
        ('reb', 'Rebound'),
        ('ast', 'Assist'),
        ('stl', 'Steal'),
        ('blk', 'Block'),
        ('tov', 'Turnover'),
        ('pf', 'Personal Foul'),
        ('pts', 'obtained Points (score)'),
        ('lose_pts', 'lost Points (score)')
    ]

    hga, aga = 0, 0
    hw, hl, aw, al, hgl, agl = 0, 0, 0, 0, 0, 0
    hwin_team = []
    hloss_team = []
    awin_team = []
    aloss_team = []
    noresult = 0
    home_info = {u: 0 for u, v in detail_infos}
    home_info['cnt'] = 0
    away_info = {u: 0 for u, v in detail_infos}
    away_info['cnt'] = 0
    for id in result['season_type']:
        if result['team_name_home'][id] == name:
            home_info['cnt'] += 1
            if result['wl_home'][id] == 'W':
                hw += 1
                hwin_team.append(result['team_name_away'][id])
            elif result['wl_home'][id] == 'L':
                hl += 1
                hloss_team.append(result['team_name_away'][id])
            else:
                noresult += 1
            for u, v in detail_infos:
                if u != 'lose_pts':
                    home_info[u] += result[u + '_home'][id]
            home_info['lose_pts'] += result['pts_away'][id]
        else:
            away_info['cnt'] += 1
            if result['wl_away'][id] == 'W':
                aw += 1
                awin_team.append(result['team_name_home'][id])
            elif result['wl_away'][id] == 'L':
                al += 1
                aloss_team.append(result['team_name_home'][id])
            else:
                noresult += 1
            for u, v in detail_infos:
                if u != 'lose_pts':
                    away_info[u] += result[u + '_away'][id]
            away_info['lose_pts'] += result['pts_home'][id]
    for u, v in detail_infos:
        if 'pct' in u:
            if home_info['cnt'] > 0:
                home_info[u] /= home_info['cnt']
            if away_info['cnt'] > 0:
                away_info[u] /= away_info['cnt']
    hwin_team = list(set(hwin_team))
    hloss_team = list(set(hloss_team))
    awin_team = list(set(awin_team))
    aloss_team = list(set(aloss_team))
    if oppo:
        result_str += f" winning {hw + aw} times ({hw} at home, {aw} at away) and losing {al + hl} times ({hl} at home, {al} at away), {noresult} games with no result. {name} win against Teams {hwin_team}, and lost to Teams{hloss_team}  at home games. And they won against Teams {awin_team}, and lost to Teams{aloss_team}  at away games."
    else:
        result_str += f" winning {hw + aw} times ({hw} at home, {aw} at away) and losing {al + hl} times ({hl} at home, {al} at away), {noresult} games with no result."

    for u, v in detail_infos[-2:]:
        if 'pct' in u and (home_info['cnt'] + away_info['cnt']) > 0:
            number = (home_info[u] * home_info['cnt'] + away_info[u] * away_info['cnt']) / (
                        home_info['cnt'] + away_info['cnt'])
        else:
            number = home_info[u] + away_info[u]
        number = round(number, 2)
        result_str += f' the total {v} is {number}, where {round(home_info[u], 2)} at home, {round(away_info[u], 2)} at away.'
    result_str += '\n\n The statistics for specific seasons are:'
    for season in season_type:
        if len(tmpresult[season]) == 0:
            continue
        hga, aga = 0, 0
        hw, hl, aw, al, hgl, agl = 0, 0, 0, 0, 0, 0
        hwin_team = []
        hloss_team = []
        awin_team = []
        aloss_team = []
        noresult = 0
        home_info = {u: 0 for u, v in detail_infos}
        home_info['cnt'] = 0
        away_info = {u: 0 for u, v in detail_infos}
        away_info['cnt'] = 0
        for id in tmpresult[season]:
            if result['team_name_home'][id] == name:
                home_info['cnt'] += 1
                if result['wl_home'][id] == 'W':
                    hw += 1
                    hwin_team.append(result['team_name_away'][id])
                elif result['wl_home'][id] == 'L':
                    hl += 1
                    hloss_team.append(result['team_name_away'][id])
                else:
                    noresult += 1
                for u, v in detail_infos:
                    if u != 'lose_pts':
                        home_info[u] += result[u + '_home'][id]
                home_info['lose_pts'] += result['pts_away'][id]
            else:
                away_info['cnt'] += 1
                if result['wl_away'][id] == 'W':
                    aw += 1
                    awin_team.append(result['team_name_home'][id])
                elif result['wl_away'][id] == 'L':
                    al += 1
                    aloss_team.append(result['team_name_home'][id])
                else:
                    noresult += 1
                for u, v in detail_infos:
                    if u != 'lose_pts':
                        away_info[u] += result[u + '_away'][id]
                away_info['lose_pts'] += result['pts_home'][id]
        for u, v in detail_infos:
            if 'pct' in u:
                if home_info['cnt'] > 0:
                    home_info[u] /= home_info['cnt']
                if away_info['cnt'] > 0:
                    away_info[u] /= away_info['cnt']
        hwin_team = list(set(hwin_team))
        hloss_team = list(set(hloss_team))
        awin_team = list(set(awin_team))
        aloss_team = list(set(aloss_team))
        if oppo:
            result_str += f"<{season}>{name} play {home_info['cnt'] + away_info['cnt']} {season}  games during {date}, winning {hw + aw} times ({hw} at home, {aw} at away) and losing {al + hl} times ({hl} at home, {al} at away), {noresult} games with no result.  {name} win against Teams {hwin_team}, and lost to Teams{hloss_team}  at home games. And they won against Teams {awin_team}, and lost to Teams{aloss_team}  at away games.\n\n  "
        else:
            result_str += f"<{season}>{name} play {home_info['cnt'] + away_info['cnt']} {season}  games during {date}, winning {hw + aw} times and losing {al + hl} times, {noresult} games with no result. "
        result_str += f'The Detailed statistics of {name} about {season} games are: '
        for u, v in detail_infos:
            if 'pct' in u and (home_info['cnt'] + away_info['cnt']) > 0:
                number = (home_info[u] * home_info['cnt'] + away_info[u] * away_info['cnt']) / (
                            home_info['cnt'] + away_info['cnt'])
            else:
                number = home_info[u] + away_info[u]
            number = round(number, 2)
            result_str += f'the {v} is {number}, where {round(home_info[u], 2)} at home, {round(away_info[u], 2)} at away; '
        result_str += f'.</{season}>\n\n'
    return result_str


def get_last_week_dates(date_str, date_format='%Y%m%d'):
    # 将字符串日期转换为datetime对象
    given_date = datetime.strptime(date_str, date_format)
    sunday_adjustment = (given_date.weekday() + 1) % 7
    sunday_of_given_week = given_date - timedelta(days=sunday_adjustment)
    print(sunday_adjustment, sunday_of_given_week)
    sunday_of_last_week = sunday_of_given_week - timedelta(days=7)
    last_week_dates = [sunday_of_last_week + timedelta(days=i) for i in range(7)]

    # 返回上周的所有日期，格式化为字符串
    return [date.strftime('%Y-%m-%d') for date in last_week_dates]


def process_soccer_info(result, name, date):
    W = 0
    L = 0
    hga, aga = 0, 0
    hd, ad = 0, 0
    hw, hl, aw, al, hgl, agl = 0, 0, 0, 0, 0, 0
    hwin_team = []
    hloss_team = []
    hd_team = []
    ad_team = []
    awin_team = []
    aloss_team = []
    noresult = 0
    for key in result['date'].keys():
        if result['result'][key] is None:
            noresult += 1
            continue
        if result['venue'][key] == 'Home':
            hgl += int(result['GF'][key])
            hga += int(result['GA'][key])
            if result['result'][key] == 'W':
                hwin_team.append(result['opponent'][key])
                hw += 1
            elif result['result'][key] == 'L':
                hloss_team.append(result['opponent'][key])
                hl += 1
            elif result['result'][key] == 'D':
                hd_team.append(result['opponent'][key])
                hd += 1
        else:
            agl += int(result['GF'][key])
            aga += int(result['GA'][key])
            if result['result'][key] == 'W':
                aw += 1
                awin_team.append(result['opponent'][key])
            elif result['result'][key] == 'D':
                ad_team.append(result['opponent'][key])
                ad += 1
            elif result['result'][key] == 'L':
                al += 1
                aloss_team.append(result['opponent'][key])
    W = hw + aw
    L = hl + al
    D = hd + ad
    gf = hgl + agl
    ga = hga + aga
    hwin_team = list(set(hwin_team))
    hloss_team = list(set(hloss_team))
    hd_team = list(set(hd_team))
    ad_team = list(set(ad_team))
    awin_team = list(set(awin_team))
    aloss_team = list(set(aloss_team))
    result_str = f"{name} play {len(result['date'])} times during {date}, winning {W} times and losing {L} times and draw {D} times, {noresult} games with no result,  scoring a total of {gf} points and losing a total of {ga} points, where they played {hw + hl} times during {date} at home games, {hw} wins, {hl} loses, {hd} draws, scored {hgl} points, lost {hga} points,  and get {aw} wins, {al} loses, {ad} draws, scored {agl} points at away games.\n During {date}, Team {name} won against Teams {hwin_team}, lost to Teams{hloss_team}  and draw to {hd_team} at home games. And they won against Teams {awin_team}, lost to Teams{aloss_team}  and draw to {ad_team} at away games.\n\n "

    if len(result['date']) > 5:
        tmp_str = 'the Detailed data for the latest 5 games:\n'
    else:
        tmp_str = f'the Detailed data for the {len(result["date"])} games:\n'
    for key in list(result['date'].keys())[::-1]:
        tmp_str += process_single_soccer(result, key, name)
    result_str += tmp_str
    return result_str


def process_single_soccer(result, key, name):
    game_type, season, hometeam = key.replace(')', '').replace('(', '').replace("'", '').split(',')[:3]
    game_type = game_type.strip()
    season = season.strip()
    hometeam = hometeam.strip()
    result.keys(), result['date']
    result_map = {
        'W': 'win',
        'L': 'lose',
        'D': 'draw'
    }

    tmpstr = f"<game> At {result['date'][key]}, {result['day'][key]}, local time {result['time'][key]}, in a match of the {season} season of the {game_type}, round {result['round'][key]}, {name} play at {result['venue'][key]} against {result['opponent'][key]}"
    if result['result'][key] in result_map.keys():
        tmpstr += f", and {result_map[result['result'][key]]} this game."
    else:
        tmpstr += ', and the result is unknown.'
    tmpstr += f" The specific score data of {name} is as follows: The Goals For (GF) is {result['GF'][key]}; The Goals Against (GA) is {result['GA'][key]}; The Expected Goals (xG) is {result['xG'][key]}; The Expected Goals Against (xGA) is {result['xGA'][key]}; The Possession is {result['Poss'][key]}; The Captain is {result['Captain'][key]}; The Formation is {result['Formation'][key]}; The number of Attendance is {result['Attendance'][key]}; The Referee is {result['Referee'][key]}.</game>\n"
    return tmpstr


def get_this_week_dates(date_str, date_format='%Y%m%d'):
    # 将字符串日期转换为datetime对象
    given_date = datetime.strptime(date_str, date_format)
    sunday_adjustment = (given_date.weekday() + 1) % 7
    sunday_of_given_week = given_date - timedelta(days=sunday_adjustment)
    print(sunday_adjustment, sunday_of_given_week)
    last_week_dates = [sunday_of_given_week + timedelta(days=i) for i in range(7)]

    # 返回上周的所有日期，格式化为字符串
    return [date.strftime('%Y-%m-%d') for date in last_week_dates]


def locate_nba_name(name):
    name = name.strip().lower().capitalize()
    if name in nba_teams.keys():
        return  nba_teams[name]
    for v in nba_teams.values():
        if name in v:
            return v
    return None


def sports_parse_answer(cmd,query_time):
    try:
        m, d, y = query_time.split(',')[0].split('/')
        nowtime = ''.join([y, m, d])
        pairs = cmd.strip().replace('ALL get', 'get_all').replace('<|endoftext|>', '').replace('\nsort',
                                                                                               ' sort').replace(
            'all sort', 'sort').split('get_')
        result = []
        claen_pairs = []
        all_result_str = []
        for pair in pairs:
            pair = pair.strip()
            result_str = ""
            if pair != '' and 'none' not in pair:
                claen_pairs.append(pair)
                args = extract_parts(pair)
                print('args', args)
                if args[0] == 'nba_on_date':
                    name, date = args[1].split(',')
                    date = date[:-1].strip()
                    name = name.strip().title()
                    print(name, date)
                    if name == 'Philadelphia 76Ers':
                        name = 'Philadelphia 76ers'
                    if check_nba_name(name) == False:
                        name = locate_nba_name(name)
                        if check_nba_name(name) == False:
                            print('no name')
                            continue
                    if date == 'yesterday':
                        date = get_yesterday(nowtime)
                        date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
                        print('date', date)
                        result = api.sports_nba_get_games_on_date(date, name)['result']
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'today':
                        result = api.sports_nba_get_games_on_date(f'{nowtime[:4]}-{nowtime[4:6]}-{nowtime[6:8]}', name)[
                            'result']
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'last week':
                        last_week_dates = get_last_week_dates(nowtime)
                        result = {}
                        for day in last_week_dates:
                            tmp = api.sports_nba_get_games_on_date(day, name)['result']
                            if tmp is not None:
                                result.update(tmp)
                        if result == {}:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'this week':
                        last_week_dates = get_this_week_dates(nowtime)
                        result = {}
                        for day in last_week_dates:
                            tmp = api.sports_nba_get_games_on_date(day, name)['result']
                            if tmp is not None:
                                result.update(tmp)
                        if result == {}:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'last time':
                        for shift in range(1, 60):
                            day = calculate_date(nowtime, -shift)
                            result = api.sports_nba_get_games_on_date(day, name)['result']
                            if result is not None:
                                break
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on last time.')
                            continue
                    elif date == 'next time':
                        for shift in range(1, 60):
                            day = calculate_date(nowtime, shift)
                            result = api.sports_nba_get_games_on_date(day, name)['result']
                            if result is not None:
                                break
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on next time.')
                            continue
                    elif contains_alpha(date) == False:
                        result = api.sports_nba_get_games_on_date(date, name)['result']
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    else:
                        result = None
                    if result is not None:
                        result_str = process_nba_result(result, name, date, 'opponent' in args[2])
                elif args[0] == 'soccer_on_date':
                    name, date = args[1].split(',')
                    date = date[:-1].strip()
                    name = name.strip().title()
                    print(name, date)
                    if check_score_name(name) == False:
                        if 'Manchester' in name:
                            name = 'Manchester Utd'
                        elif 'Paris' in name:
                            name = 'Paris S-G'
                        elif 'Nott' in name:
                            name = 'Nott\'ham Forest'
                        else:
                            print('no name')
                            continue
                    if date == 'yesterday':
                        date = get_yesterday(nowtime)
                        date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
                        print('date', date)
                        result = api.sports_soccer_get_games_on_date(date, name)['result']
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'today':
                        result = \
                        api.sports_soccer_get_games_on_date(f'{nowtime[:4]}-{nowtime[4:6]}-{nowtime[6:8]}', name)[
                            'result']
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'last week':
                        last_week_dates = get_last_week_dates(nowtime)
                        result = {}
                        for day in last_week_dates:
                            tmp = api.sports_soccer_get_games_on_date(day, name)['result']
                            if tmp is not None:
                                result.update(tmp)
                        if result == {}:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'this week':
                        last_week_dates = get_this_week_dates(nowtime)
                        result = {}
                        for day in last_week_dates:
                            tmp = api.sports_soccer_get_games_on_date(day, name)['result']
                            if tmp is not None:
                                result.update(tmp)
                        if result == {}:
                            all_result_str.append(f'There is no match record about {name} on {date}.')
                            continue
                    elif date == 'last time':
                        for shift in range(1, 60):
                            day = calculate_date(nowtime, -shift)
                            result = api.sports_soccer_get_games_on_date(day, name)['result']
                            if result is not None:
                                break
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on last time.')
                            continue
                    elif date == 'next time':
                        for shift in range(1, 60):
                            day = calculate_date(nowtime, shift)
                            result = api.sports_soccer_get_games_on_date(day, name)['result']
                            if result is not None:
                                break
                        if result is None:
                            all_result_str.append(f'There is no match record about {name} on next time.')
                            continue
                    elif contains_alpha(date) == False:
                        result = api.sports_soccer_get_games_on_date(date, name)['result']
                    else:
                        result = None
                    if result is not None:
                        result_str = process_soccer_info(result, name, date)
                    else:
                        all_result_str.append(f'There is no match record about {name} on {date}.')
                        continue
                else:
                    print('to do')
            if result_str != '':
                all_result_str.append(result_str)
    except:
        all_result_str = []
    return all_result_str


finance_method_map = {
    'recent': get_recent_week_day,
    'last': get_last_week_day,
    'this': get_recent_week_day,
}
finance_data_map = {
    'monday': 1,
    'tuesday': 2,
    'wednesday': 3,
    'thursday': 4,
    'friday': 5,
    'saturday': 6,
    'sunday': 7,
}
print(parse_answer('get_movie_person_oscar(None,None,[eq(year,2021),eq(category,"best actor"),eq(winner,true)])["name"] '))
print(music_parse_answer('get_grammy_year("2004")'))
print(finance_parse_answer('stock_price("oramed pharmaceuticals","latest")["close"]','02/28/2024, 07:49:44 PT'))
print(sports_parse_answer("get_nba_on_date('new orleans pelicans','2022-01')['gf']", '03/13/2024, 09:13:49 PT'))
print(parse_answer('get_movie("star wars")[len"]get_movie("harry potter")[len]'))
print(finance_parse_answer('stock_price("gfi","recent friday")["open"]', '03/13/2024, 09:13:49 PT'))