import SimilarityModel
import ExtractFeatures as ef
import Util
class VerifySimilarity:
    def __init__(self, text1, text2, model):
        self.text1 = text1
        self.text2 = text2
        self.model = model
    
    
    def preProcess(self):
        exFeatures = ef.ExtractFeatures()
        return exFeatures.getFeaturesIndividual(self.text1, self.text2)
        
    def verify(self):
        model = SimilarityModel.SimilarityModel("test_features_assin2-train-only10.0")
        predData = model.datasetScikit(self.preProcess())
        model = self.model
        
        print(model.predict(predData['features']))
        