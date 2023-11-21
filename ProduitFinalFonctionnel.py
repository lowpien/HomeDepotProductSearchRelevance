# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:08:55 2023

@author: User
"""

# Déclarer les bibliothèques nécessaires
import nltk
import json
import seaborn as sns
import Levenshtein 
from matplotlib import pyplot as plt 
from matplotlib_venn import venn2
import pandas as pd
import numpy as np
import regex as re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer 
import warnings
warnings.filterwarnings('ignore')
stemmer = SnowballStemmer('english')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print('Imports Done!')

print('Loadings starting')


# lire les fichiers de données

## importation De train
dt_train = pd.read_csv("f:/train.csv", encoding='latin-1')

## importation Du fichier test
dt_test = pd.read_csv("f:/test.csv", encoding='latin-1')

## importation Des Attributs
dt_attributes = pd.read_csv("f:/attributes.csv", encoding='latin-1')

## importation De la Description des produits
dt_descriptions = pd.read_csv("f:/product_descriptions.csv")


# La fonction affiche les informations de l'ensemble de données

def show_data_info(dt):
    print(f"dt shape: \n {dt.shape} \n")
    print("bd columns: \n", dt.columns, "\n")
    dt.info()
    return dt.head(2)

# Fichier Train
## Contient les champs de données product_uid, product_title, search_term et pertinence

show_data_info(dt_train)
print(dt_train)
print(dt_train.head())
print(dt_train.describe())

print("Il ya au Total {} produit ".format(len(dt_train.product_title.unique())))
print("Il ya au Total {} requête de recherche ".format(len(dt_train.search_term.unique())))
print("Il ya au Total {} product_uid".format(len(dt_train.product_uid.unique())))


# Statistiques sur le nombre de notes dans l'ensemble dt_train

def count_SR():
    high_SR = [i for i in dt_train['relevance'] if i == 3.00]
    midle_SR = [i for i in dt_train['relevance'] if i > 1.00 and i < 3.00]
    low_SR = [i for i in dt_train['relevance'] if i == 1.00]
    return [len(high_SR), len(midle_SR), len(low_SR)] 


plt.bar(['haut_sr', 'moyen_sr', 'bas_sr'], count_SR(), label='Score de pertinence', color='blue')
plt.xlabel('Notes')
plt.ylabel('Nombres')
plt.title('Nombres de notes')
plt.legend()
plt.show()

# plus de détails sur le degré de pertinence
sns.countplot(x="relevance", data=dt_train)
plt.title('Répartition des scores de pertinence')
plt.show()


# Testing data
# Contient les champs de données product_uid, product_title, search_term
#afficher les informations dt_train

show_data_info(dt_test)

print("Il ya au Total {} produit ".format(len(dt_test.product_title.unique())))
print("Il ya au Total {} requête de recherche ".format(len(dt_test.search_term.unique())))
print("Il ya au Total {} product_uid".format(len(dt_test.product_uid.unique())))

#graphique montrant le nombre de product_uids apparaissant dans les deux ensembles dt_test et dt_train

venn2([set(dt_train["product_uid"]), set(dt_test["product_uid"])], set_labels=('train', 'test'), set_colors=('green', 'blue'))
plt.show()


# le graphique représente le nombre de search_terms apparaissant dans les deux ensembles dt_test et dt_train

venn2([set(dt_train["search_term"]), set(dt_test["search_term"])], set_labels=('train', 'test'), set_colors=('green', 'blue'))
plt.show()

# Descriptions data

## Contient les champs product_uid et product_description
## Contient des descriptions textuelles de chaque produit.

# afficher les informations sur le fichier dt_descriptions
show_data_info(dt_descriptions)

print("Il ya au Total {} product_uid ".format(len(dt_descriptions.product_uid.unique())))
print("Il ya au Total {} product_descriptions ".format(len(dt_descriptions.product_description.unique())))


# Attributes data
#Contient les champs product_uid, name et value
#Fournit des informations détaillées sur un sous-ensemble de produits (représentant généralement des spécifications détaillées). Tous les produits n'auront pas des attributs.
#afficher les informations du fichier dt_attributes

show_data_info(dt_attributes)


# Nettoyer les données
## Transformer les ensembles dt_description et dt_attributes
## fonction pour supprimer les doublons

def remove_duplicates(string):
    lits_tokens = [] 
    [lits_tokens.append(str(_)) for _ in string.split() if _ not in lits_tokens]
    return ' '.join(lits_tokens)

# Récupère la plus grande partie commune dans le champ du nom sous forme de puce au lieu de puce0 x y z

dt_attributes['name'] = [_[:6] if 'bullet' in str(_).lower() else _ for _ in dt_attributes['name'].tolist()]

# créer le champ product_attributes à partir du nom et de la valeur

dt_attributes['product_attributes'] = dt_attributes['name'] + ' ' +  dt_attributes['value']
dt_attributes = dt_attributes.drop(['name', 'value'], axis=1)

# fusionner les attributs qui partagent le même identifiant en un seul et concaténer les product_attributes correspondants ensemble
# Utilisez la fonction astype() pour convertir tous les jetons en chaînes car ils contiennent des nombres dans le champ de valeur.
# Pour concaténer les valeurs après le regroupement, utilisez la fonctionagrégat(). Les valeurs de la colonne valeur qui partagent le même identifiant seront concaténées ensemble, séparées par un espace.

dt_attributes = dt_attributes.groupby('product_uid').aggregate({'product_attributes': lambda _ : ' '.join(_.astype(str))})

# récupère la partie commune du nom et supprime les jetons en double.

dt_attributes['product_attributes'] = [remove_duplicates(_) for _ in dt_attributes.product_attributes.tolist()]


# Joindre la table dt_attributes à la table dt_descriptions en utilisant l'attribut commun product_id

dt_des_attr = pd.merge(dt_descriptions, dt_attributes, on='product_uid', how='left')


#afficher les informations de la table nouvellement rejointe
show_data_info(dt_des_attr)
print(dt_des_attr.head())


# rechercher et remplacer toutes les valeurs Null par ''

dt_des_attr['product_attributes'].fillna('', inplace = True)

# Supprimez les phrases mal copiées dans le champ des descriptions de produits, y compris les descriptions en HTML et les liens.

strings = ['br',
           'src',
           'href',
           'alt',
           'please visit'
           'Click here to review our return policy for additional information regarding returns', 
           'Click here to see Home Depot', 
           'Click here for our Project Guide', 
           'Click here for our Buying Guide', 
           'Click on the More Info tab to download',
           'CLICK HERE to create your own collection',
           'Click Here for details on the services',
           'Click Here for Ideas and Designs',
           'Click Here for a Demo of the Design',
           'Click Here to learn more about',
           'CLICK HERE to view our',
           'Click below to visit our',
           'Click here to purchase a sample of this',
           'click on the link to get started',
           'Click image to enlarge',
           'https://www.ryobitools.com/nation',
           'http://www.homedepot.com/ApplianceDeliveryandInstallation',
           'http://itemvideo-dev.microsite.homedepot.com/111414/26P/online_BB_banner_111114.jpg',
           'http://www.homedepot.com/p/Rev-A-Shelf-Door-Mounting-Kit-5WB-DMKIT/202855698']

for string in strings:
    dt_des_attr['product_description'] = [_.lower().replace(string.lower(), '') for _ in dt_des_attr.product_description]


# afficher le top 5
show_data_info(dt_des_attr)
dt_des_attr.head(5)

# créez un nouveau champ contenant à la fois la description du produit et ses valeurs d'attribut.
# champ product_description_attributes = product_description + product_attributes

dt_des_attr['product_description_attributes'] = dt_des_attr['product_description'] + ' ' + dt_des_attr['product_attributes']

# supprimer 2 anciens champs

dt_des_attr = dt_des_attr.drop(['product_description', 'product_attributes'], axis=1)

# résultat après concaténation.
show_data_info(dt_des_attr)
dt_des_attr.head(5)

# concaténer l'ensemble dt_train avec dt_des_attr
dt_train = pd.merge(dt_train, dt_des_attr, on='product_uid', how='left')

# joindre le fichier dt_test avec le fichier dt_des_attr
dt_test = pd.merge(dt_test, dt_des_attr, on='product_uid', how='left')

#montrer les résultats
show_data_info(dt_train)
dt_train.head(5)

#montrer les résultats
show_data_info(dt_test)
dt_test.head(5)


#Fonctions de données propres
#obtenir les données du fichier .json pour corriger les fautes d'orthographe
#remplacer les valeurs comme les champs clés par des champs de valeur

spell_check = json.load(open('C:/Users/User/spell_check.json', 'r'))
def spell_fix(string):
    for (k,v) in spell_check.items():
        string = string.replace(k, v)
    return string


#Nous n'incluons pas les unités (ft. lb. sq. ...) car le point final sera pris en charge par
# re.split(r'\W+', sent) fonction dans la fonction de prétraitement
# Dans la description et le titre, 'inch' se trouve généralement comme 'in.' où comme dans search_term,
# on le trouve à la fois comme « in » et « in ». nous traitons donc les deux cas séparément. 
#fonction de prétraitement pour les termes de recherche
#Nous n'utilisons plus de stemming ni de mots vides pour le moment. Nous utiliserons le stimming après correction des termes de recherche
#'in.','in' en inch est pris en charge lors de l'étape de prétraitement
#'inches' en inch sera pris charge dans le stemming. 

# Prétraitement des données

def standardize_units(text):
    if text is None:
        return ""
    
    text = " " + text + " "
    text = re.sub('( gal | gals | galon )', ' gallon ', text)
    text = re.sub('( ft | fts | feets | foot | foots )', ' feet ', text)
    text = re.sub('( squares | sq )', ' square ', text)
    text = re.sub('( lb | lbs | pounds )', ' pound ', text)
    text = re.sub('( oz | ozs | ounces | ounc )', ' ounce ', text)
    text = re.sub('( yds | yd | yards )', ' yard ', text)
    return text

def preprocessing(sent):
    if pd.isna(sent):
        return ""
    sent = sent.replace('in.', ' inch ')
    words = re.split(r'\W+', sent)
    words = [word.lower() for word in words]
    res = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", ' '.join(words))
    cleaned = standardize_units(res)
    cleaned = ' '.join(cleaned.split())  # Supprime les espaces supplémentaires
    return cleaned

def preprocessing_search(sent):
    if pd.isna(sent):
        return ""
    sent = sent.replace('in.', ' inch ')
    words = re.split(r'\W+', sent)
    words = [word.lower() for word in words]
    res = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", ' '.join(words))
    res = standardize_units(res)
    res = res.replace(' in ', ' inch ')
    cleaned = ' '.join(res.split())  # Supprime les espaces supplémentaires
    return cleaned


# supprimer les mots vides
# Cette fonction ne fonctionne que sur les champs de description et d'attribut

def remove_stopwords(string):
    return ' '.join([w for w in string.split() if not w in stop_words])


# Formation aux pratiques propres
# Correction de fautes de frappe pour les champs search_term, product_title et product_description_attributes

# correcteur orthographique

def correct_spell(s, spell_check_dict):
    for tyto in spell_check_dict:
        s = s.replace(tyto, spell_check_dict[tyto])
    return s

# Appliquez les fonctions de prétraitement aux colonnes spécifiées

dt_train['search_term'] = dt_train['search_term'].map(lambda x:spell_fix(x))
dt_train['product_title'] = dt_train['product_title'].map(lambda x:spell_fix(x))
dt_train['product_description_attributes'] = dt_train['product_description_attributes'].map(lambda x:spell_fix(x))

# montrer les résultats
show_data_info(dt_train)
dt_train.head(5)


# normaliser le champ search_term
dt_train['search_term'] = [standardize_units(_) for _ in dt_train.search_term]


# standardiser les champs product_title et product_description_attributes

dt_train['product_title'] = [standardize_units(_) for _ in dt_train.product_title]
dt_train['product_description_attributes'] = [standardize_units(_) for _ in dt_train.product_description_attributes]

# supprimer les mots vides dans le champ product_description_attributes

dt_train['product_description_attributes'] = [remove_stopwords(_) for _ in dt_train.product_description_attributes]

show_data_info(dt_train)
dt_train.head(5)


# Test de fichier propre
# Correction de fautes de frappe pour les champs search_term, product_title et product_description_attributes


dt_test['search_term'] = dt_test['search_term'].map(lambda x:spell_fix(x))
dt_test['product_title'] = dt_test['product_title'].map(lambda x:spell_fix(x))
dt_test['product_description_attributes'] = dt_test['product_description_attributes'].map(lambda x:spell_fix(x))

# montrer les résultats
show_data_info(dt_test)
dt_test.head(5)

# normaliser le champ search_term

dt_test['search_term'] = [standardize_units(_) for _ in dt_test.search_term]

# standardiser les champs product_title et product_description_attributes

dt_test['product_title'] = [standardize_units(_) for _ in dt_test.product_title]
dt_test['product_description_attributes'] = [standardize_units(_) for _ in dt_test.product_description_attributes]


# supprimer les mots vides dans le champ product_description_attributes

dt_test['product_description_attributes'] = [remove_stopwords(_) for _ in dt_test.product_description_attributes]

#montrer les résultats
show_data_info(dt_test)
dt_test.head(5)

# Écrivez une fonction qui permet d'observer l'influence de la longueur du champ sur la pertinence

def correlation(dt_sample, dt_field, transform=True):
    # Obtenez la longueur du champ d'information
    # En définissant transform=True, nous pourrons plus tard utiliser transform=False pour obtenir les données dans ce champ lui-même, pas la longueur :) afin que nous puissions voir l'influence de ce champ de données lui-même sur la pertinence
    x_ar = np.array(dt_sample[dt_field].map(lambda x:len(str(x).split())).astype(np.int64)) if transform else dt_sample[dt_field]
   
    # Obtenez la pertinence correspondante
    y_ar = np.array(dt_sample['relevance'])
   
    # Dessiner des points (longueur, pertinence) sur le graphique
    plt.plot(x_ar, y_ar, 'bo')
   
    # calculez m et b pour tracer la droite de régression
    m, b = np.polyfit(x_ar, y_ar, 1)
   
    # Régression de dessin
    plt.plot(x_ar, m * x_ar + b,'r')
    # Afficher le graphique
    plt.show()


# le fichier dt_train brut non transformé
dt_raw_train = pd.read_csv('f:/train.csv', encoding='latin-1')


# représente le champ product_title
correlation(dt_raw_train, 'product_title')

# représente avec le champ search_term
correlation(dt_raw_train, 'search_term')

# représente le champ des attributs de description du produit

correlation(dt_train, 'product_description_attributes')


# Écrivez des fonctions pour créer des fonctionnalités à partir de l'occurrence de mots dans search_term
# Trouver des jetons communs
# séparez les jetons de 2 phrases avec la fonction split() puis calculez l'occurrence des jetons dans la phrase 1 dans la phrase 2

def str_common_tokens(sentence_1, sentence_2):
    return sum(1 for word in str(sentence_2).split() if word in set(str(sentence_1).split()))


# Les mots sont quelque peu courants
# Pour obtenir des mots partiellement courants, nous ne séparons pas les jetons (n'utilisez pas la fonction split() ici)
# Comptez respectivement l'occurrence des mots dans la phrase 1 dans la phrase 2

def str_common_word(sentence_1, sentence_2):
    return sum(1 for word in str(sentence_2) if word in set(sentence_1))



# Calculez le total de tous les jetons qui apparaissent "au total"

def set_shared_words_whole(row_data):
    return str_common_tokens(row_data[0], row_data[1])

# Additionnez tous les mots qui apparaissent "en partie"

def set_shared_words_part(row_data):
    return str_common_word(row_data[0], row_data[1])

# Extraire les fonctionnalités de train

# calculer la longueur de search_term

dt_train['len_of_querry'] = [len(_.split()) for _ in dt_train['search_term'].values]

# calcule le nombre total de fois où les jetons dans le champ search_term apparaissent entièrement dans les champs product_title et product_description_attributes

dt_train['shared_words_whole_st_pt'] = [set_shared_words_whole(_) for _ in  dt_train[['search_term','product_title']].values]
dt_train['shared_words_whole_st_pdat'] = [set_shared_words_whole(_) for _ in  dt_train[['search_term','product_description_attributes']].values]

# calculer le nombre total de fois où les jetons dans le champ search_term apparaissent partiellement dans les champs product_title et product_description_attributes

dt_train['shared_words_part_st_pt'] = [set_shared_words_part(_) for _ in dt_train[['search_term', 'product_title']].values]
dt_train['shared_words_part_st_pdat'] = [set_shared_words_part(_) for _ in dt_train[['search_term', 'product_description_attributes']].values]

# calculer la similarité de search_term et product_title

dt_train['similarity'] = [Levenshtein.ratio(_[0], _[1]) for _ in dt_train[['search_term', 'product_title']].values]

#montrer les résultats
show_data_info(dt_train)
dt_train.head(5)

#Extraire les fonctionnalités de l'ensemble de test

# calculer la longueur de search_term

dt_test['len_of_querry'] = [len(_.split()) for _ in dt_test['search_term'].values]

# calcule le nombre total de fois où les jetons dans le champ search_term apparaissent entièrement dans les champs product_title et product_description_attributes

dt_test['shared_words_whole_st_pt'] = [set_shared_words_whole(_) for _ in dt_test[['search_term', 'product_title']].values]
dt_test['shared_words_whole_st_pdat'] = [set_shared_words_whole(_) for _ in dt_test[['search_term', 'product_description_attributes']].values]

# calculer le nombre total de fois où les jetons dans le champ search_term apparaissent partiellement dans les champs product_title et product_description_attributes

dt_test['shared_words_part_st_pt'] = [set_shared_words_part(_) for _ in dt_test[['search_term', 'product_title']].values]
dt_test['shared_words_part_st_pdat'] = [set_shared_words_part(_) for _ in dt_test[['search_term', 'product_description_attributes']].values]


# calculer la similarité de search_term et product_title

dt_test['similarity'] = [Levenshtein.ratio(_[0], _[1]) for _ in dt_test[['search_term', 'product_title']].values]

#montrer les résultats
show_data_info(dt_test)
dt_test.head(5)

# Réévaluer la qualité des fonctionnalités avec la fonction de corrélation en utilisant l'ensemble dt_train

correlation(dt_train, 'len_of_querry', transform=False)

correlation(dt_train, 'shared_words_whole_st_pt', transform=False)

correlation(dt_train, 'shared_words_whole_st_pdat', transform=False)

correlation(dt_train, 'shared_words_part_st_pt', transform=False)

correlation(dt_train, 'shared_words_part_st_pdat', transform=False)

correlation(dt_train, 'similarity', transform=False)

# supprime la fonctionnalité shared_words_part_st_pdat dans les ensembles de train et de test

sdt_train = dt_train.drop(['shared_words_part_st_pdat'],axis=1)
sdt_test = dt_test.drop(['shared_words_part_st_pdat'],axis=1)

# Fractionner les données

# supprimer les colonnes de texte en ne laissant que les colonnes contenant des fonctionnalités

sdt_train = sdt_train.drop(['product_title','search_term','product_description_attributes'],axis=1)

# définir x_train et y_train

y_train = sdt_train['relevance'].values
X_train = sdt_train.drop(['id','relevance'], axis=1).values

# montrer les résultats
show_data_info(sdt_train)
sdt_train.head(5)

# Dans l'ensemble de test, nous supprimons également tous les autres champs, ne laissant que les champs avec des fonctionnalités

X_test = sdt_test.drop(['id','product_title','search_term','product_description_attributes'],axis=1).values

# L'identifiant du test ici est l'identifiant des paires search_term et titre du produit

id_test = sdt_test['id']

# montrer les résultats
show_data_info(sdt_test)
sdt_test.head(5)


# Régresseur de forêt aléatoire (Random forest regressor)

# Réaliser une formation sur l'ensemble de données avec le modèle "Random forest" fourni dans la bibliothèque sklearn
from sklearn.ensemble import RandomForestRegressor

# ensemble de paramètres est référencé à partir d'une autre source
rfr = RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=17, max_depth=10)

# ajuster
rfr.fit(X_train, y_train)

# prédire avec le modèle ajusté
y_pred = rfr.predict(X_test)



# Train and Test Set
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform



print('Shape of final train data:', X_train.shape, y_train.shape)
print('Shape of final test data:', X_test.shape, id_test.shape)

# Standardiser les données
scaler_final = StandardScaler()
X_train_std = scaler_final.fit_transform(X_train)

# Hyperparameter Tuning
random_grid = {
    'alpha': loguniform(1e-5, 1e4)
}

model = Ridge()
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid, n_iter=100,
                                   scoring='neg_root_mean_squared_error',
                                   cv=5, verbose=False, return_train_score=True,
                                   random_state=42, n_jobs=-1)
results = random_search.fit(X_train_std, y_train)
print('-'*50)
print('Hyperparameter Tuning Results:')
print('best rmse:', 'cv', -results.best_score_, 'train', -results.cv_results_['mean_train_score'][results.best_index_])
print('best param', results.best_params_)

# Final Training
best_params = results.best_params_
ridge_model = Ridge(**best_params)
ridge_model.fit(X_train_std, y_train)

# Final Testing
X_test_std = scaler_final.transform(X_test)
id_test_pred = ridge_model.predict(X_test_std)
rmse = mean_squared_error(id_test, id_test_pred, squared=False)
print('-'*50)
print('Final performance on test set:')
print('Final rmse on test data is:', rmse)

# Importer le stemmer de NLTK
from nltk.stem import PorterStemmer

# Fonction pour mettre le texte dans la forme racine
def set_root_form(text):
    # Utiliser le Porter Stemmer pour la racinisation
    stemmer = PorterStemmer()
    words = text.split()
    root_form_words = [stemmer.stem(word) for word in words]
    return ' '.join(root_form_words)

# Fonction pour vérifier si la requête est contenue dans une chaîne
def contains_query(string, query):
    return query in string

# Fonction pour trouver les produits les plus pertinents en fonction de la requête utilisateur
def find_most_relevant_products(query, model, scaler, dt_train):
    # Prétraitement de la requête de l'utilisateur
    preprocessed_query = preprocessing_search(query)
    preprocessed_query = standardize_units(preprocessed_query)
    preprocessed_query = set_root_form(preprocessed_query)
    
    # Ajout de la fonctionnalité pour vérifier si le terme de recherche contient la requête
    contains_query_feature = int(contains_query(dt_train['search_term'][0], preprocessed_query))

    # Création des fonctionnalités à partir de la requête
    query_features = pd.DataFrame({
        'len_of_querry': [len(preprocessed_query.split())],
        'shared_words_whole_st_pt': [set_shared_words_whole((preprocessed_query, dt_train['product_title'][0]))],
        'shared_words_whole_st_pdat': [set_shared_words_whole((preprocessed_query, dt_train['search_term'][0]))],
        'shared_words_part_st_pt': [set_shared_words_part((preprocessed_query, dt_train['product_title'][0]))],
        'similarity': [Levenshtein.ratio(preprocessed_query, dt_train['product_title'][0])],
        'contains_query': [contains_query_feature]
    })
    
    # Standardiser les données
    query_features_std = scaler.transform(query_features.values)
    
    # Prédire la pertinence avec le modèle
    relevance_prediction = model.predict(query_features_std)[0]
    
    # Ajouter une colonne indiquant si la requête est contenue dans le titre ou la description
    results = dt_train[['product_title', 'search_term', 'relevance']].copy()
    results['relevance_prediction'] = relevance_prediction
    results['contains_query'] = (results['product_title'].str.contains(preprocessed_query, case=False) |
                                 results['search_term'].str.contains(preprocessed_query, case=False))
    
    # Filtrer les résultats pour ne conserver que ceux où la requête est contenue
    results = results[results['contains_query']]

    # Trier par pertinence et présence de la requête
    results = results.sort_values(by=['relevance_prediction'], ascending=[False])
    
    return results[['product_title', 'search_term', 'relevance_prediction']]


# Exemple d'utilisation de la fonction pour trouver les produits les plus pertinents pour une requête utilisateur
user_query = "air conditioner"
relevant_products = find_most_relevant_products(user_query, ridge_model, scaler_final, dt_train)
show_data_info(relevant_products)
print("Most relevant products for the query:")
print(relevant_products)


# Exemple d'utilisation de la fonction pour trouver les produits les plus pertinents pour une requête utilisateur
user_query = "taille haie"
relevant_products = find_most_relevant_products(user_query, ridge_model, scaler_final, dt_train)
show_data_info(relevant_products)
print("Most relevant products for the query:", user_query)
print(relevant_products)



# exporter un fichier csv

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(r'f:\submission_HK.csv', index=False)

submission_db = pd.read_csv('f:/submission_HK.csv', encoding='latin-1')
print(submission_db)
print(submission_db.head())
print(submission_db.describe())

