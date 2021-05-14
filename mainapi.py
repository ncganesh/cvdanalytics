import pickle
import pandas as pd
import numpy as np
import re
from flask import Flask
from flask import json
from flask import request
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

# Connecting to elastic Search
es = Elasticsearch(hosts=["http://3.238.229.207:9200/"])
app = Flask(__name__)

filename = 'finalized_model_new.sav'
# Labels
labels = ['related', 'request', 'offer', 'aid_related',
          'medical_help', 'medical_products',
          'search_and_rescue', 'security', 'military',
          'child_alone', 'water', 'food', 'shelter',
          'clothing', 'money', 'missing_people', 'refugees',
          'death', 'other_aid', 'infrastructure_related',
          'transport', 'buildings', 'electricity', 'tools',
          'hospitals', 'shops', 'aid_centers',
          'other_infrastructure', 'weather_related',
          'floods', 'storm', 'fire', 'earthquake', 'cold',
          'other_weather', 'direct_report']

# tweet = content['text']
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/get_intent_predictions', methods=['POST'])
def get_prob():
    """
    API to get prediction labels for the tweet

    :return:
    """
    try:

        tweet = request.form['tweettext']
        # print(tweet)
        # predictions = loaded_model.predict([tweet])
        # result = np.where(predictions == 1)
        # print("result",result)
        # preds = [labels[xi] for xi in result[1]]
        pred_probs = loaded_model.predict_proba([tweet])
        pred_prob = np.argmax([row[0][1] for row in pred_probs])
        preds = labels[pred_prob]
        #print("PREDS-------", preds)
    except Exception as e:
        print("Exception", e)
        preds = "related"

    return str(preds)


@app.route('/get_last1mindata', methods=['POST'])
def get_lastmindata():
    """
    API to get last 1 min data

    :return:
    """
    result = es.search(index="twitter_india_covid", body={
        "query": {
            "range": {
                "@timestamp": {
                    "gte": datetime.utcnow() - timedelta(minutes=1),
                    "lt": datetime.utcnow()
                }
            }
        },
        # ensure that we return all docs in our test corpus

    })
    lis = []
    for item in result['hits']['hits']:
        lis.append(item['_source'])

    responses = pd.DataFrame(lis).to_json(orient="records")
    return responses


@app.route('/get_fulldata', methods=['POST'])
def get_fulldata():
    """
    API to get all data from index
    :return: json with all data
    """
    s = Search(using=es, index="twitter_india_covid")
    df = pd.DataFrame([hit.to_dict() for hit in s.scan()])
    responses = df.to_json(orient="records")
    return responses


@app.route('/semantic_search', methods=['POST'])
def search():
    """
    API to perform semnatic search
    :return:
    """
    queries = str(request.form['userquery'])
    query = {
        "size": 30,
        "query": {
            "query_string": {"query": queries}
        }
    }

    results = []
    for result in es.search(index="twitter_india_covid", body=query)["hits"]["hits"]:
        source = result["_source"]
        print(source)
        results.append([source["id_str"],source["created_at"], source["user"]["screen_name"],source["text"],min(result["_score"], 18) / 18])

    # similarity = Similarity("valhalla/distilbart-mnli-12-3")
    # results = [text for _, text in search(query, limit * 10)]
    # return [(score, results[x]) for x, score in similarity(query, results)][:limit]

    responses = pd.DataFrame(results, columns=['id_str','created_at','screen_name', 'Text','Score']).to_json(orient="records")

    return responses


if __name__ == '__main__':
    app.run(port=5009)