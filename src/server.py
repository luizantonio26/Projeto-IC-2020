from flask import Flask, jsonify, request
from flask_cors import CORS
import ExtractFeatures
import LoadModels
import SimilarityModel
import Util

app = Flask(__name__)
CORS(app)
models = LoadModels.LoadModels()
spacy = models.loadSpacy()
stanfordNlp = models.loadModelStanfordNLP()
embeddings = models.loadWorldEmbeddings()
ef = ExtractFeatures.ExtractFeatures(stanfordNlp, embeddings, spacy, "assin2-train-only")
sm = SimilarityModel.SimilarityModel(ef)
@app.route('/', methods=['POST'])
def isSimilarity():
    sentences = request.get_json()
    features = ef.getFeaturesIndividual(sentences['s1'], sentences['s2'])
    result = sm.randomForestPredict(sentences['s1'], sentences['s2'])
    return jsonify({"isEntailment":result, "features":features})
    #data = request.get_json()


@app.route('/add_example', methods=['POST'])
def addExample():
    dt = request.get_json()
    train = Util.readJson('training_model.json')
    if dt['isSimilarity']:
        dt['features']['entailments'] = True
    else:
        dt['features']['entailments'] = False
    train.append(dt)
        
    

app.run(port=3000)