from flask import Flask, jsonify, request
from flask_cors import CORS
import ExtractFeatures
import LoadModels
import SimilarityModel

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
    s1 = request.args.get('s1')
    s2 = request.args.get('s2')
    print(sentences, s1, s2)
    features = ef.getFeaturesIndividual(sentences['s1'], sentences['s2'])
    result = sm.randomForestPredict(sentences['s1'], sentences['s2'])
    return jsonify({"isEntailment":result, "features":features})
    #data = request.get_json()


app.run(port=3000)