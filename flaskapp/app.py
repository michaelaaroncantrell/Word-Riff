import flask
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os


# Initialize the app
app = flask.Flask(__name__, static_folder='static', static_url_path='')


#########UNPICKLE##########################
with open("pickled-files/titles_no_dups.pkl", 'rb') as picklefile: 
     titles_no_dups = pickle.load(picklefile)

with open("pickled-files/df_asin_title_2.pkl", 'rb') as picklefile: 
     df_asin_title = pickle.load(picklefile)
   
with open("pickled-files/df_ratings_007_topics.pkl", 'rb') as picklefile: 
     df_ratings = pickle.load(picklefile)

with open("pickled-files/dic_asin.pkl", 'rb') as picklefile: 
     dic_asin = pickle.load(picklefile)

with open("pickled-files/dic_asin_reverse.pkl", 'rb') as picklefile: 
     dic_asin_reverse = pickle.load(picklefile)
   
with open("pickled-files/image_dic.pkl", 'rb') as picklefile: 
     image_dic = pickle.load(picklefile)

with open("pickled-files/model_007.pkl", 'rb') as picklefile: 
     model = pickle.load(picklefile)

with open("pickled-files/nlp_simil_topics.pkl", 'rb') as picklefile: 
     nlp_simil = pickle.load(picklefile)
   
with open("pickled-files/simil.pkl", 'rb') as picklefile: 
     simil = pickle.load(picklefile)

with open("pickled-files/vect_007.pkl", 'rb') as picklefile: 
     vect = pickle.load(picklefile)




###########HOME PAGE#####################
@app.route("/")
def home_page():
    with open("index.html", 'r') as viz_file:
        return viz_file.read()



##########ALBUM TO ALBUM#######################
def nlp_recommender(asin):
    idx = dic_asin_reverse[asin]
    l = [[nlp_simil[idx][i],dic_asin[i]] for i in range(nlp_simil.shape[0])]
    l.sort(reverse=True)
    return l


def collab_recommender(asin):
    idx = dic_asin_reverse[asin]
    l = [[simil[idx][i],dic_asin[i]] for i in range(simil.shape[0])]
    l.sort(reverse=True)
    return l


def recommender(asin,perc):
    '''given album asin and float perc in (0,1), return top 5 nlp recommendations subject to \
    constraint of being in top perc percent of all album recs'''
    
    nlp_recs = nlp_recommender(asin)
    collab_recs = collab_recommender(asin)
    
    num_albums = len(collab_recs)
    cut_off = int(num_albums*perc)
    
    top_perc_albums = [entry[1] for entry in collab_recs[:cut_off]]    
    
    recs = []
    while len(recs)<6:
        next_rec = nlp_recs.pop(0)[1]
        if (next_rec in top_perc_albums) & (get_title(next_rec) in titles_no_dups):
            recs.append(next_rec)
            
    return recs[1:]


def get_asin(title):
    return list(df_asin_title[df_asin_title['title'] == title]['asin'])[0]

def get_title(asin):
    return list(df_asin_title[df_asin_title['asin'] == asin]['title'])[0]

def get_imgurl(title):
    return image_dic[title]

def get_recs_and_images(title, num):
    perc=num/100
    asin = get_asin(title)
    rec = recommender(asin, perc)
    rec_titles = [get_title(asin) for asin in rec]
    imageurls = [get_imgurl(title) for title in rec_titles]
    return [rec_titles, imageurls]



############ALBUM TO ALBUM##################
@app.route("/album-to-album")
def albumtoalbum():
    with open("album-to-album.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/a2a", methods=["POST"])
def a2a():
    """
    arr is an array with album name and number from 0 to 100
    """
    data = flask.request.json  
    arr = data['grid']
    album_name = arr[0]
    num = int(arr[1])

    return flask.jsonify({'grid': get_recs_and_images(album_name, num)}) 



##############TEXT TO ALBUM#############
def make_topics_sentiments(review_list, model, count_vectorizer):
    '''Given a list of sentence reviews for one album, an NMF model, the count_vectorizer that\
     made the model and n_components, returns an array representing the score of the album in the topics.'''
    l  = []
    review_vec = count_vectorizer.transform(review_list)
    matrix = model.transform(review_vec)
    for row in range(matrix.shape[0]):
        topic_probs = matrix[row]
        l.append(topic_probs)
        u = [np.array(coord).sum()/matrix.shape[0] for coord in zip(*l)]

        ###subset on actual topics
        u = u[:1]+u[2:18]+u[19:20]+u[21:27]

    return u


def text_to_album(text):
    
    desired_album = make_topics_sentiments([text], model, vect)
    df_desired = pd.DataFrame(desired_album).transpose()
    df_desired = pd.DataFrame(normalize(df_desired, axis=1), columns = df_desired.columns, \
                                index = df_desired.index).multiply(100).round(2)

    comparisons = cosine_similarity(df_ratings,df_desired)
    asin_recs = [dic_asin[idx] for idx in np.array([x[0] for x in comparisons]).argsort()[-5:]][::-1]
    rec_titles = [get_title(asin) for asin in asin_recs]
    imageurls = [get_imgurl(title) for title in rec_titles]

    ###save RAM on website
    del df_desired, comparisons
    return [rec_titles, imageurls]



###############TEXT TO ALBUM##############
@app.route("/text-to-album")
def viz_page():
    with open("text-to-album.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/t2a", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this url,
    Read the grid from the json, update and send it back
    """
    data = flask.request.json  ###
    arr = data['grid']
    text = arr[0]
    print(text)
    output = text_to_album(text)
    return flask.jsonify({'grid': output}) 



#############MASH UP################
def album_mashup(title1,title2):
    asin1 = get_asin(title1)
    asin2 = get_asin(title2)
    new_album = pd.DataFrame(df_ratings.loc[asin1]+df_ratings.loc[asin2]).transpose() 
    comparisons = cosine_similarity(df_ratings,new_album)
    asin_recs = [dic_asin[idx] for idx in np.array([x[0] for x in comparisons]).argsort()[-5:]][::-1]
    rec_titles = [get_title(asin) for asin in asin_recs]
    imageurls = [get_imgurl(title) for title in rec_titles]
    return [rec_titles, imageurls]


###########MASH UP##################
@app.route("/mashup")
def mash():
    with open("mashup.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/mashup", methods=["POST"])
def mashup():
    """
    When A POST request with json data is made to this url,
    Read the grid from the json, update and send it back
    """
    data = flask.request.json  ###
    arr = data['grid']
    print('mash', arr)
    print(album_mashup(arr[0], arr[1]))


    return flask.jsonify({'grid': album_mashup(arr[0], arr[1])}) 

############ABOUT #################
@app.route("/about")
def about_page():
    with open("about.html", 'r') as viz_file:
        return viz_file.read()

 

############CONTACT#################
@app.route("/contact")
def contact_page():
    with open("contact.html", 'r') as viz_file:
        return viz_file.read()
############################# 



app.run(host='0.0.0.0', port=5001)