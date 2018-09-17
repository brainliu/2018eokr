#-*-coding:utf8-*-
#user:brian
#created_at:2018/9/14 13:13
# file: flak.py
#location: china chengdu 610000
from werkzeug.wrappers import Request,Response

@Request.application
def hello(request):
    return Response("hello world!")
