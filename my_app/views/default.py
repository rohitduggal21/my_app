import pickle
from pyramid.view import view_config

modelfile = open('model.rd', 'rb')
model = pickle.load(modelfile) 
modelfile.close()

@view_config(route_name='home', renderer='../templates/main.jinja2')
def my_view(request):
    return {'project': 'my_app'}

@view_config(route_name='sentiment', renderer='json')
def get_sentiment(request):
    keyword = request.json_body['keyword']
    return model.predict(keyword)