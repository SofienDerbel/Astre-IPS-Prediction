from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    # predict_proba(final) kenet hakeka w hatina fiha list
    dat = {'10-Quel langage de programmation as-tu déjà utilisé ?=Java;C#;HTML5 CSS;Php;Python;JavaScript;CMS': 1.0,
            '11-As-tu déjà utilisé ? [Des logiciels de CAO (solidworks)]=Un peu': 1.0,
            '11-As-tu déjà utilisé ? [Des logiciels de création de jeux vidéos (Unity, Unreal)]=Un peu': 1.0,
            '11-As-tu déjà utilisé ? [Des logiciels de montage vidéo]=Un peu': 1.0,
            '11-As-tu déjà utilisé ? [Des logiciels de retouche (Paint, Photoshop)]=Un peu': 1.0,
            '11-As-tu déjà utilisé ? [Linux]=Régulièrement': 1.0, '11-As-tu déjà utilisé ? [Mac]=Jamais': 1.0,
            '11-As-tu déjà utilisé ? [Une carte Arduino/Raspberry]=Régulièrement': 1.0,
            '11-As-tu déjà utilisé ? [Windows]=Tout le temps': 1.0, '12-As-tu déjà réalisé ? [Des logos]=Un peu': 1.0,
            '12-As-tu déjà réalisé ? [Un jeu video]=Jamais': 1.0,
            '12-As-tu déjà réalisé ? [Un logiciel sur pc]=Régulièrement': 1.0,
            '12-As-tu déjà réalisé ? [Un robot]=Jamais': 1.0,
            '12-As-tu déjà réalisé ? [Un système électronique (capteurs...)]=Un peu': 1.0,
            '12-As-tu déjà réalisé ? [Une application mobile]=Régulièrement': 1.0,
            "13-Que penses-tu de ? [Anaconda]=J'aime bien": 1.0,
            '13-Que penses-tu de ? [AndroidStudio]=Je suis fan': 1.0, "13-Que penses-tu de ? [Atom]=Je n'aime pas": 1.0,
            "13-Que penses-tu de ? [Codblock]=Je n'aime pas": 1.0, "13-Que penses-tu de ? [Eclipse]=Je n'aime pas": 1.0,
            '13-Que penses-tu de ? [IntellJ]=Je suis fan': 1.0, '13-Que penses-tu de ? [Ionic]=Connais pas': 1.0,
            '13-Que penses-tu de ? [LabView]=Connais pas': 1.0, "13-Que penses-tu de ? [NotePad++]=Je n'aime pas": 1.0,
            "13-Que penses-tu de ? [SublimeText]=J'aime bien": 1.0, "13-Que penses-tu de ? [Unity]=J'aime bien": 1.0,
            '13-Que penses-tu de ? [UnrealEngine]=Connais pas': 1.0,
            "13-Que penses-tu de ? [Visual studio]=J'aime bien": 1.0, "2-D'où viens tu ?=France": 1.0,
            "3-Quel parcours avez vous fait avant d’intégrer l'ensim=BTS": 1.0,
            '4-As-tu déjà passé le TOEIC ? Tu peux mettre ton score dans autre=Non': 1.0,
            '5-Plutôt...=Apple;Android': 1.0,
            "5-Quelles sont les associations de l'ensim qui t’intéresse ?=Jensim": 1.0,
            '6-Quel est ton personnage de fiction préféré ?=Ironman': 1.0,
            '7-Que fais-tu de ton temps libre ?=Netflix, babbel, mooc, ...': 1.0,
            '8-Quels sont tes films/séries préférées ?= Series Marvel, films CAT8, films Hunger Games, films divergente, ': 1.0,
            '9-Quel est ton navigateur préféré ?=Chrome;Brave': 1.0}
    data = [["9-Quel est ton navigateur préféré ?=Chrome;Brave", 1.0],
            ["8-Quels sont tes films/séries préférées ?= Series Marvel, films CAT8, films Hunger Games, films divergente, ",1.0]]
    v = DictVectorizer(sparse=False)
    vd = v.fit_transform(dat)
    """data = [{
        '10-Quel langage de programmation as-tu déjà utilisé ?=Java;C#;HTML5 CSS;Php;Python;JavaScript;CMS': 1.0}
        , {'9-Quel est ton navigateur préféré ?=Chrome;Brave': 1.0}]"""
    xx = v.inverse_transform(vd)
    X = xx.iloc[:, 1:307].values
    y = xx.iloc[:, 0].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #prediction = model.predict(np.array(x).reshape(1,-1))
    print(X_test)
    """f = v.inverse_transform(data)
    print(f)"""
    return "yt"
    """prediction = model.predict_proba(x)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('forest_fire.html',
                               pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),
                               bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',
                               pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),
                               bhai="Your Forest is Safe for now")
"""

if __name__ == '__main__':
    app.run(debug=True)
