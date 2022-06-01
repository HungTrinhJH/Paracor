from rest_framework import viewsets, status, views
from rest_framework.decorators import action, parser_classes
from rest_framework.response import Response
from .serializers import *
from django.utils.encoding import smart_str
from re import findall, split
from contextlib import closing
from django.db import connection
from rest_framework.parsers import MultiPartParser
from concordance.pagination import RawDataPagination, SentenceDataPagination
import json
from .lemma import wordToLemma
from .help import getListSentence
import math
import ast
from django.http import QueryDict
from django.contrib.auth.hashers import make_password, check_password
import jwt
from datetime import datetime, timedelta
from .trainer import *

Trainer = ModelMng(model_name='BERTBiDataModelV2_Adam_Softmax_BIDI_HTanh_64_1',
                   model_dir=r'static')
Trainer.is_train(False)


class Predict(views.APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        data = json.loads(smart_str(request.body, 
                        encoding='utf-8', strings_only=False, errors='strict'))
        result = Test(Trainer=Trainer, test_data_raw=data)
        return Response(result, 200)


class PredictFile(views.APIView):
    def post(self, request):
        file = smart_str(request.FILES['filename'].read(), 
                                encoding='utf-8', strings_only=False, errors='strict')
        file = file.replace("\r","")
        pattern = r'(.*)\n(.*)\n*'
        extract = re.findall(pattern, file)
        data = []
        for item in extract:
            data.append({"en_sent": item[0], "vi_sent":item[1]})
        result = Test(Trainer=Trainer, test_data_raw=data)
        return Response(result, 200)

class FileUploadView(views.APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        data = smart_str(request.FILES['filename'].read(
        ), encoding='utf-8', strings_only=False, errors='strict')
        lang = request.POST['lang']
        if not lang:
            Response("missing parameter", status=404)

        pattern = r'(ED|VD)(\d{6})(\d{2})(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))(?:\t(\S+))'
        sql = 'INSERT INTO {}Data (lang, sentence_id, word_id, word, lemma, links, morpho, pos, phrase, grm, ner, semantic) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'.format(
            lang.capitalize())
        sql1 = 'INSERT INTO {}Sentence (sentence_id, sentence) VALUES (%s,%s)'.format(
            lang.capitalize())

        list_field = findall(pattern, data)

        newlist = getListSentence(list_field)

        with closing(connection.cursor()) as cursor:
            cursor.executemany(sql, list_field)
            cursor.executemany(sql1, newlist)

        return Response(status=204)


class TotalStatistics(views.APIView):
    def get(self, request, format=None):
        sql = "select * from TotalStatistics"
        with closing(connection.cursor()) as cursor:
            cursor.execute(sql)
            return Response(cursor, status=200)


class EditDataRaw(views.APIView):
    def put(self, request, format=None):
        put = request.body
        put = ast.literal_eval(put.decode('utf-8'))
        put = put['body']
        lang = put['lang']
        id = put['id']
        sql = "UPDATE {}Data SET".format("En" if lang == "ED" else "Vn")
        for i in put:
            if i == "id" or i == 'lang':
                continue
            if "'" in put[i]:
                sql += " {}=\"{}\",".format(i, put[i])
            else:
                sql += " {}='{}',".format(i, put[i])
        sql = sql[:-1]
        sql += " where id={}".format(id)
        with closing(connection.cursor()) as cursor:
            cursor.execute(sql)
        return Response(status=200)


class Statistics(views.APIView):
    def get(self, request, format=None):
        req = request.GET
        lang = req.get('lang')
        ner = req.get('ner')
        pos = req.get('pos')
        semantic = req.get('semantic')
        size = req.get('size')
        sql = "select word, sum(count) as count from {}Statistics".format(
            lang.capitalize())
        sql1 = "select * from TotalStatistics"
        if ner:
            sql += " where ner='"+ner+"'"
        if pos:
            sql += " where pos='"+pos+"'"
        if semantic:
            sql += " where semantic='"+semantic+"'"
        sql += " group by word order by count desc"
        if size:
            sql += " limit " + size

        temp = None
        result = []
        with closing(connection.cursor()) as cursor:
            cursor.execute(sql1)
            total = cursor.fetchall()
            cursor.execute(sql)
            temp = cursor.fetchall()
        for item in temp:
            result.append({
                "word": item[0],
                "count": item[1],
                "percent": round(item[1] / total[1][1], 2),
                "F": round(-math.log(item[1]/total[1][1]), 2)
            })
        return Response(result, status=200)


class DetailSentence(views.APIView):
    def get(self, request, format=None):
        req = request.GET

        id = req.get("id")
        lang = req.get('lang')

        check = ['en', 'vn']
        if lang not in check:
            return Response("client request error", 400)

        check.remove(lang)

        sql = "select * from {}Data where sentence_id = %s"

        result = {}

        with closing(connection.cursor()) as cursor:
            cursor.execute(sql.format(lang.capitalize()), [id])
            result['source'] = cursor.fetchall()

            cursor.execute(sql.format(check[0].capitalize()), [id])
            result['target'] = cursor.fetchall()

        return Response(result, 200)


class Search(views.APIView):
    def get(self, request, format=None):
        req = request.GET
        lang = req.get('lang')
        qt = req.get('qt')
        pos = req.get('pos')
        ner = req.get('ner')
        structure = req.get('structure')
        page = req.get('page')

        check = ['en', 'vn']
        check.remove(lang)

        if structure:
            structure = " ".join(structure.split("_"))
            sql = "select * from (select sentence_id , GROUP_CONCAT(pos ORDER BY word_id SEPARATOR ' ') as pos_sentence, GROUP_CONCAT(word ORDER BY word_id SEPARATOR ' ') as sentence FROM {}Data group by sentence_id) as new where new.pos_sentence REGEXP ' *{} '".format(
                lang.capitalize(), structure)
            sql_count = "select count(*) as count from (select sentence_id , GROUP_CONCAT(pos ORDER BY word_id SEPARATOR ' ') as pos_sentence, GROUP_CONCAT(word ORDER BY word_id SEPARATOR ' ') as sentence FROM {}Data group by sentence_id) as new where new.pos_sentence REGEXP ' *{} '".format(
                lang.capitalize(), structure)
            result = {
                "total": 0,
                "source": {},
                "target": {}
            }
            sql += " limit 100 offset {}".format(int(page)*100-100)
            with closing(connection.cursor()) as cursor:
                cursor.execute(sql)
                result['source'][structure] = cursor.fetchall()
                cursor.execute(sql_count)
                result['total'] = cursor.fetchall()[0][0]
            id_results = []
            for item in result['source'][structure]:
                if item[0] not in id_results:
                    id_results.append(item[0])
            if id_results:
                sql = "select sentence_id, sentence from {}Sentence where sentence_id in %s order by id"

                with closing(connection.cursor()) as cursor:
                    cursor.execute(sql.format(
                        check[0].capitalize()), [id_results])
                    result['target'][structure] = cursor.fetchall()
            return Response(result, 200)

        keyword = req.get('q').lower().strip()
        if not keyword:
            return Response("missing keyword", 400)

        if qt == 'mat':
            keyword = keyword.replace(' ', '_')
            sql = "select sentence_id ,word, links from {}Data where binary word = binary %s ".format(
                lang.capitalize())
            sql_count = "select count(sentence_id) as count from {}Data where binary word = binary %s ".format(
                lang.capitalize())
        elif qt == 'mor':
            keyword = wordToLemma(keyword, lang)
            sql = "select sentence_id ,word, links from {}Data where morpho = %s ".format(
                lang.capitalize())
            sql_count = "select count(sentence_id) from {}Data where morpho = %s ".format(
                lang.capitalize())
        elif qt == 'phrase':
            list_key_word = keyword.split(" ")
            if "'" not in keyword:
                if lang == 'en':
                    sql = "select sentence_id, replace(sentence, '_', ' ') as sentence from {}Sentence where binary sentence like binary '% {} %' ".format(
                        lang.capitalize(), keyword)
                else:
                    sql = "select sentence_id, replace(sentence, '_', ' ') as sentence1, sentence from {}Sentence where binary sentence like binary '%{}%' ".format(
                        lang.capitalize(), list_key_word[0])
            else:
                if lang == 'en':
                    sql = '''select sentence_id, replace(sentence, '_', ' ') as sentence from {}Sentence where binary sentence like binary "% {} %" '''.format(
                        lang.capitalize(), keyword)
                else:
                    sql = '''select sentence_id, replace(sentence, '_', ' ') as sentence1, sentence from {}Sentence where binary sentence like binary "%{}%" '''.format(
                        lang.capitalize(), list_key_word[0])
            result = {
                "total": 0,
                "source": [],
                "target": []
            }
            list_id = []
            position = []
            with closing(connection.cursor()) as cursor:
                cursor.execute(sql)
                temp1 = cursor.fetchall()
            if len(temp1)!=0:
                if lang == 'en':
                    result['total'] = len(temp1)
                    for item in temp1[int(page)*100-100:100*int(page)]:
                        left, right = item[1].split(keyword,1)
                        position.append(len(left.split(" "))-1)
                        list_id.append(item[0])
                        result['source'].append(
                            {"key": keyword, "left": left, "right": right, "sentence_id": item[0], "lang": lang})
                else:
                    item_page = 0
                    for item in temp1:
                        pattern = r".".join(list_key_word)
                        keyword = findall(pattern, item[2])
                        if keyword:
                            result['total'] += 1
                            item_page += 1
                            if item_page > int(page)*100-100 and item_page < 100*int(page): 
                                left, right = item[2].split(keyword[0],1)
                                result['source'].append(
                                    {"key": keyword, "left": left, "right": right, "sentence_id": item[0], "lang": lang})
                                position.append(len(left.split(" "))-1)
                                list_id.append(item[0])
                if list_id:
                    sql = "select sentence_id, word, links from {}Data where sentence_id in %s".format(
                        lang.capitalize())

                    with closing(connection.cursor()) as cursor:
                        cursor.execute("select sentence_id, group_concat(links separator ' ') from {}Data where sentence_id in %s group by sentence_id order by sentence_id".format(
                            lang.capitalize()), [list_id])
                        temp1 = cursor.fetchall()
                        cursor.execute("select sentence_id, sentence from {}Sentence where sentence_id in %s order by sentence_id".format(
                            check[0].capitalize()), [list_id])
                        temp2 = cursor.fetchall()
                    for index, item in enumerate(position):
                        list_word = temp1[index][1].split(" ")
                        for i in range(len(list_key_word)):
                            if list_word[int(item)+i] != '-':
                                var = list_word[int(item)+i].split(",")
                                list_temp = temp2[index][1].split(" ")
                                keyword = " ".join(
                                    list_temp[int(var[0])-1:int(var[-1])])
                                result['target'].append({"key": keyword, "left": " ".join(list_temp[:int(
                                    var[0])-1]), "right": " ".join(list_temp[int(var[-1]):]), "sentence_id": temp2[index][0], "lang": check[0]})
                                break
                        else:
                            result['target'].append(
                                {"key": "", "left": "", "right": temp2[index][1], "sentence_id": temp2[index][0], "lang": check[0]})

            return Response(result, 200)
        values = [keyword]

        if pos:
            sql += "AND pos=%s "
            sql_count += "AND pos=%s "
            values.append(pos)

        if ner:
            sql += "AND ner=%s "
            sql_count += "AND ner=%s "
            values.append(ner)
        sql += "order by id "
        sql += "limit 100 offset {} ".format(int(page)*100-100)
        result = {
            "total": 0,
            "source": [],
            "target": []
        }
        with closing(connection.cursor()) as cursor:
            cursor.execute(sql, values)
            temp1 = cursor.fetchall()
            cursor.execute(sql_count, values)
            result["total"] = cursor.fetchall()[0][0]
        id_results = []
        temp = []
        for item in temp1:
            if item[0] not in id_results:
                id_results.append(item[0])
                temp.append(item)

        if id_results:

            sql = "select sentence from {}Sentence where sentence_id in %s order by id"

            with closing(connection.cursor()) as cursor:
                cursor.execute(sql.format(lang.capitalize()), [id_results])
                source = cursor.fetchall()
                cursor.execute(sql.format(check[0].capitalize()), [id_results])
                target = cursor.fetchall()
            for i, v in enumerate(source):
                find = findall("(^{} | {} | {}$)".format(
                    temp[i][1], temp[i][1], temp[i][1]), v[0])[0]
                hold = v[0].split(find, 1)
                result['source'].append(
                    {"key": temp[i][1], "left": hold[0], "right": hold[1], "sentence_id": temp[i][0], "lang": lang})
                key = temp[i][2]
                if key == '-' or key == '0':
                    key = ""
                    left = ""
                    right = target[i][0]
                else:
                    hold = target[i][0].split(" ")
                    key = key.split(",")
                    key = " ".join(hold[int(key[0])-1:int(key[-1])])
                    hold = target[i][0].split(key)
                    left = hold[0]
                    right = hold[1]
                result['target'].append(
                    {"key": key, "left": left, "right": right, "sentence_id": temp[i][0], "lang": check[0]})

        return Response(result, 200)

    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        try:
            data = smart_str(request.FILES['filename'].read(
            ), encoding='utf-8', strings_only=False, errors='strict').split("\n")
            if not data[-1]:
                data.pop(-1)
            pattern = r'lang *= *(\w+).*'
            lang = findall(pattern, data.pop(0))[0]
            check = ['en', 'vn']
            check.remove(lang)

            sql = [["select * from (select sentence_id , GROUP_CONCAT(pos ORDER BY word_id SEPARATOR ' ') as pos_sentence, GROUP_CONCAT(word ORDER BY word_id SEPARATOR ' ') as sentence FROM {}Data group by sentence_id) as new where new.pos_sentence like '%{}%';".format(
                lang.capitalize(), structure), structure] for structure in data]
            result = {
                "source": {},
                "target": {}
            }
            with closing(connection.cursor()) as cursor:

                for i in sql:
                    cursor.execute(i[0])
                    result['source'][i[1]] = cursor.fetchall()

                    id_results = []
                    for item in result['source'][i[1]]:
                        if item[0] not in id_results:
                            id_results.append(item[0])
                    if id_results:

                        sql = "select sentence_id, sentence from {}Sentence where sentence_id in %s order by id"

                        cursor.execute(sql.format(
                            check[0].capitalize()), [id_results])
                        result['target'][i[1]] = cursor.fetchall()
        except:
            Response('input file wrong', 400)
        return Response(result, 200)


class UserAPI(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()

    def create(self, request, format=None):
        req = request.POST
        username = req.get('username')
        password = req.get('password')
        role = req['role']
        user = User.objects.filter(username=username)
        if not user:
            user = User.objects.create(
                username=username, password=make_password(password), role=role)
            return Response(status=200)
        return Response("User already existed", status=400)

    @ action(detail=False, methods=['POST'])
    def loginAdmin(self, request, format=None):
        req = request.body
        req = ast.literal_eval(req.decode('utf-8'))
        req = req['body']
        # req = request.POST
        username = req['username']
        password = req['password']
        user = User.objects.filter(username=username)
        if user:
            if user[0].role == 'admin' and check_password(password, user[0].password):
                encoded_jwt = jwt.encode({'username': username, 'exp': datetime.utcnow(
                ) + timedelta(hours=24)}, 'secret', algorithm='HS256')
                return Response({'token': encoded_jwt, 'username': username, })
            else:
                return Response("wrong username/password", status=400)
        return Response("wrong username/password", status=400)
    
    def update(self, request, pk, format=None):
        put = request.data
        old_password = put['old_password']
        new_password = put['new_password']
        user = User.objects.filter(id=pk)
        if user:
            if check_password(old_password, user[0].password):
                user = User.objects.update(password=make_password(new_password))
                return Response(user, status=200)
            else:
                return Response("Wrong password", status=400)
        return Response("User not existed", status=400)


class EnDataAPI(viewsets.ModelViewSet):
    serializer_class = EndataSerializer
    queryset = Endata.objects.all()
    pagination_class = RawDataPagination


class EnSentenceAPI(viewsets.ModelViewSet):
    serializer_class = EnsentenceSerializer
    queryset = Ensentence.objects.all()
    pagination_class = SentenceDataPagination


class VnSentenceAPI(viewsets.ModelViewSet):
    serializer_class = VnsentenceSerializer
    queryset = Vnsentence.objects.all()
    pagination_class = SentenceDataPagination


class VnDataAPI(viewsets.ModelViewSet):
    serializer_class = VndataSerializer
    queryset = Vndata.objects.all()
    pagination_class = RawDataPagination
