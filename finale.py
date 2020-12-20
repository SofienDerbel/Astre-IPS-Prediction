import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))


@app.route('/')
def sondage():
    questions = [{'10-Quel langage de programmation as-tu déjà utilisé ?': 'C;HTML5 CSS;Assembleur'},
                 {'11-As-tu déjà utilisé ? [Des logiciels de CAO (solidworks)]': 'Régulièrement'},
                 {'11-As-tu déjà utilisé ? [Des logiciels de création de jeux vidéos (Unity, Unreal)]': 'Jamais'},
                 {'11-As-tu déjà utilisé ? [Des logiciels de montage vidéo]': 'Un peu'},
                 {'11-As-tu déjà utilisé ? [Des logiciels de retouche (Paint, Photoshop)]': 'Régulièrement'},
                 {'11-As-tu déjà utilisé ? [Linux]': 'Tout le temps'}, {'11-As-tu déjà utilisé ? [Mac]': 'Un peu'},
                 {'11-As-tu déjà utilisé ? [Une carte Arduino/Raspberry]': 'Régulièrement'},
                 {'11-As-tu déjà utilisé ? [Windows]': 'Un peu'}, {'12-As-tu déjà réalisé ? [Des logos]': 'Un peu'},
                 {'12-As-tu déjà réalisé ? [Un jeu video]': 'Un peu'},
                 {'12-As-tu déjà réalisé ? [Un logiciel sur pc]': 'Un peu'},
                 {'12-As-tu déjà réalisé ? [Un robot]': 'Un peu'},
                 {'12-As-tu déjà réalisé ? [Un système électronique (capteurs...)]': 'Régulièrement'},
                 {'12-As-tu déjà réalisé ? [Une application mobile]': 'Un peu'},
                 {'13-Que penses-tu de ? [Anaconda]': 'Connais pas'},
                 {'13-Que penses-tu de ? [AndroidStudio]': 'Connais pas'},
                 {'13-Que penses-tu de ? [Atom]': 'Connais pas'}, {"13-Que penses-tu de ? [Codblock]": "J'aime bien"},
                 {'13-Que penses-tu de ? [Eclipse]': 'Connais pas'}, {'13-Que penses-tu de ? [IntellJ]': 'Connais pas'},
                 {'13-Que penses-tu de ? [Ionic]': 'Connais pas'}, {"13-Que penses-tu de ? [LabView]": "J'aime bien"},
                 {"13-Que penses-tu de ? [NotePad++]": "J'aime bien"},
                 {'13-Que penses-tu de ? [SublimeText]': 'Connais pas'},
                 {"13-Que penses-tu de ? [Unity]": "Je n'aime pas"},
                 {'13-Que penses-tu de ? [UnrealEngine]': 'Connais pas'},
                 {'13-Que penses-tu de ? [Visual studio]': 'Connais pas'}, {"2-D'où viens tu ?": "France; Paris"},
                 {"3-Quel parcours avez vous fait avant d’intégrer l'ensim": "Prepa BL; BTS"},
                 {'4-As-tu déjà passé le TOEIC ?Tu peux mettre ton score dans autre': '980; Yezs'},
                 {'5-Plutôt...': 'Android'},
                 {
                     "5-Quelles sont les associations de l'ensim qui t’intéresse ?": "BDE;BDLC;Jensim;Infographie;GALA;KFET;Trublions du Plateau;ENSIM'Elec"},
                 {
                     '6-Quel est ton personnage de fiction préféré ?': 'Mara des Acoma (Guerres de la faille, Raymond E. FEIST); Batman'},
                 {
                     '7-Que fais-tu de ton temps libre ?': 'Discuter; dessiner; écrire; grimper; nager, lire, ecouter de la musique (beaucoup), glander, des maths, jeux videos, roller, VTT, courir, penser, observer, aider, traîner sur YouTube...'},
                 {'8-Quels sont tes films/séries préférées ?': 'Fringe (serie); Enigma; Porco Rosso...'},
                 {'9-Quel est ton navigateur préféré ?': 'Firefox;Ghostery'}]

    return render_template("sondage.html.twig", questions=questions)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data = pd.read_csv("Questionnaire_3A-1.csv")

    data2 = data.drop(columns=['Timestamp']);
    print(data2)
    obj_data = data2.select_dtypes(include=['object']).copy()

    # DictVectorizer
    from sklearn.feature_extraction import DictVectorizer
    # instantiate a Dictvectorizer object for X
    data2_dv = DictVectorizer(sparse=False)
    # sparse = False makes the output is not a sparse matrix
    #je pense il faut metre les données qui viennt du form dans data2_dict



    data2_dict = obj_data.to_dict(orient='records')  # turn each row as key-value pairs

    # apply dv_X on X_dict
    data_encoded = data2_dv.fit_transform(data2_dict)

    new = pd.DataFrame.from_dict(data_encoded)

    dataf = pd.concat([data2["1-Quel est ton numéro étudiant ?"], pd.DataFrame(new, index=data2.index), ], axis=1)

    kmeans = KMeans(2)
    kmeans.fit(new)
    pred = kmeans.predict(new)
    pred2 = pred.reshape(-1, 1)

    df = pd.DataFrame(data=pred2, columns=["spécialité"])

    datafinale = pd.concat([df, new], axis=1)

    X = datafinale.iloc[:, 1:307].values
    y = datafinale.iloc[:, 0].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rOs = RandomOverSampler()
    X_ro, y_ro = rOs.fit_resample(X_train, y_train)
    # Entraînement du modèle de régression logistique
    lr = LogisticRegression()
    lr.fit(X_ro, y_ro)
    # Affichage des résultats
    y_pred = lr.predict(X_test)

    knn = KNeighborsClassifier(7)
    knn_model = knn.fit(X_ro, y_ro)
    y_pred_knn = knn_model.predict(X_test)

    filename = 'finalized_model.pkl'
    pickle.dump(knn_model, open(filename, 'wb'))

    model = pickle.load(open(filename, 'rb'))

    y_pred_knn_test = pd.DataFrame(y_pred_knn).replace(0, "ASTRE")
    y_pred_knn_test.replace(1, "IPS", inplace=True)
    a = np.array(y_pred_knn_test)

    form = request.form
    training_data = []
    for question in form:
        survey = {question + "={}".format(form.get(question)): 1.0}
        training_data.append(survey)
        survey = {}
    #data2_dict = training_data


    xtest = data2_dv.fit_transform(training_data)
    f = data2_dv.inverse_transform(xtest)
    probability = a[0]
    return render_template("sondage.html.twig", probability = probability)


if __name__ == '__main__':
    app.run(debug=True)
