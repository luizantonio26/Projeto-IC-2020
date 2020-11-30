from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, Lasso, LassoLars 
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xml.etree import cElementTree as ET
from commons import read_xml, entailment_to_str, tokenize_sentence
from ExtractFeatures import ExtractFeatures
import os
import Util
import assinEvalAdapt

class SimilarityModel:
    
    def __init__(self, ef, train="./assets/test_features_assin2-train-only10.0", test = "./assets/test_features_assin2-test10.0"):
        self.train = train
        self.test = test
        self.ef = ef
        
    
    def datasetScikit(self, data, feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals','parafraseamento','hiperonimo'], isRegression = False):
        #sin = Util.readJson(self.file+"-sin"+".json")
        labels_names = ['none','entailment']
        labels = []
        #feature_names = ['isRoot','hasNeg','entityQtd','cosEmbleddings','wmd','cos','qtdSin','subj','ant']
        features = []
        for i in range(0, len(data)):
            aux = []
            
            for k in feature_names:
                aux.append(data[i][k])
            
            # aux.append(data[i]['root'])
            # aux.append(data[i]['neg'])
            # aux.append(data[i]['entity'])
            # aux.append(data[i]['cos_embeddings'])
            # aux.append(data[i]['wmd'])    
            # aux.append(data[i]['sim'])
            # aux.append(data[i]['sinonimos'])
            # aux.append(data[i]['subjEquals'])
            # aux.append(data[i]['antonimos'])
            if not isRegression:
                if(data[i]['entailment']=='None'):
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                labels.append(data[i]['similarity'])
            features.append(aux)
            
        return {'label_names':labels_names, 'labels':labels, 'feature_names': feature_names, 'features': features}
    
    
   
    def linearRegression(self, data):
        features = data['features']
        labels = data['labels']
        
        model = LinearRegression()
        
        model.fit(features, labels)
        
        return model
    
    def logisticRegression(self, data):
        features = data['features']
        labels = data['labels']
        
        model = LogisticRegression()
        
        model.fit(features, labels)
        
        return model

    def bayesianRidge(self, data):
        features = data['features']
        labels = data['labels']
        
        model = BayesianRidge()
        
        model.fit(features, labels)
        
        return model
    
    def lasso(self, data):
        features = data['features']
        labels = data['labels']
        
        model = Lasso()
        
        model.fit(features, labels)
        
        return model
    
    def lassoLars(self, data):
        features = data['features']
        labels = data['labels']
        
        model = LassoLars()
        
        model.fit(features, labels)
        
        return model
    
    def randomForestRegressor(self, data):
        features = data['features']
        labels = data['labels']
        
        model = RandomForestRegressor()
        
        model.fit(features, labels)
        
        return model
    
    def decisionTreeRegressor(self, data):
        features = data['features']
        labels = data['labels']
        
        model = DecisionTreeRegressor()
        
        model.fit(features, labels)
        
        return model
    
    def redesNeuraisRegressor(self, data):
        features = data['features']
        labels = data['labels']
        
        model = MLPRegressor(random_state=1, max_iter=500)
        
        model.fit(features, labels)
        
        return model
    def naiveBayesTrain(self, data, nfolds=10):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        #train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
        # Inicializar classificador
        clf = GaussianNB()
        scores = cross_validate(clf, features, labels, cv=nfolds, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        # Treinar classificador
        train = clf.fit(features, labels)
        # # fazer predições
        # preds = gnb.predict(test)
        
        print("\n\nNaive Bayes")
        
        results = []
        t = 0
        for i in scores['test_accuracy']:
            t += i
        results.append({"Accuracy": t/len(scores['test_accuracy'])})
        t = 0
        for i in scores['test_precision']:
            t += i
        results.append({"Precision": t/len(scores['test_precision'])})
        t = 0
        for i in scores['test_f1']:
            t += i
        results.append({"F1_score": t/len(scores['test_f1'])})
        t = 0
        for i in scores['test_recall']:
            t += i
        results.append({"Recall": t/len(scores['test_recall'])})
        
        
        result = {"recall":results[3]['Recall'], "accuracy":results[0]['Accuracy'], "precision":results[1]['Precision'], "f1_score":results[2]['F1_score']}
        return {"model":train,"results":result, "totrain":clf}
        #return gnb

    def supportVMR(self, data):
        features = data['features']
        labels = data['labels']
        
        model = SVR()
        
        model.fit(features, labels)
        
        return model

    def supportVM(self, data, nfolds=10):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        #train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
        # Inicializar classificador
        clf = SVC(gamma='auto')
        
        scores = cross_validate(clf, features, labels, cv=nfolds, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        # Treinar classificador
        train = clf.fit(features, labels)
        
        # # fazer predições
        # preds = clf.predict(test)
        
        print("\n\nSupport Vector Machine")
        
        results = []
        t = 0
        for i in scores['test_accuracy']:
            t += i
        results.append({"Accuracy": t/len(scores['test_accuracy'])})
        t = 0
        for i in scores['test_precision']:
            t += i
        results.append({"Precision": t/len(scores['test_precision'])})
        t = 0
        for i in scores['test_f1']:
            t += i
        results.append({"F1_score": t/len(scores['test_f1'])})
        t = 0
        for i in scores['test_recall']:
            t += i
        results.append({"Recall": t/len(scores['test_recall'])})
        
        
        result = {"recall":results[3]['Recall'], "accuracy":results[0]['Accuracy'], "precision":results[1]['Precision'], "f1_score":results[2]['F1_score']}
        return {"model":train,"results":result, "totrain":clf}
        
        #return clf
        
    def decisionTreeModel(self, data, nfolds=10):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        #train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

        clf = DecisionTreeClassifier()
        scores = cross_validate(clf, features, labels, cv=nfolds, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        train = clf.fit(features, labels)
        # preds = clf.predict(test)
        
        print("\n\nArvore de decisão")
        
        results = []
        t = 0
        for i in scores['test_accuracy']:
            t += i
        results.append({"Accuracy": t/len(scores['test_accuracy'])})
        t = 0
        for i in scores['test_precision']:
            t += i
        results.append({"Precision": t/len(scores['test_precision'])})
        t = 0
        for i in scores['test_f1']:
            t += i
        results.append({"F1_score": t/len(scores['test_f1'])})
        t = 0
        for i in scores['test_recall']:
            t += i
        results.append({"Recall": t/len(scores['test_recall'])})
        
        
        result = {"recall":results[3]['Recall'], "accuracy":results[0]['Accuracy'], "precision":results[1]['Precision'], "f1_score":results[2]['F1_score']}
        return {"model":train,"results":result, "totrain":clf}
        #return clf

    def randomForest(self, data, nfolds=10):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        clf = RandomForestClassifier()
        print("\n\nRandom Forest")
        scores = cross_validate(clf, features, labels, cv=nfolds,  return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        
        train = clf.fit(features, labels)
        
        results = []
        t = 0
        for i in scores['test_accuracy']:
            t += i
        results.append({"Accuracy": t/len(scores['test_accuracy'])})
        t = 0
        for i in scores['test_precision']:
            t += i
        results.append({"Precision": t/len(scores['test_precision'])})
        t = 0
        for i in scores['test_f1']:
            t += i
        results.append({"F1_score": t/len(scores['test_f1'])})
        t = 0
        for i in scores['test_recall']:
            t += i
        results.append({"Recall": t/len(scores['test_recall'])})
        
        
        result = {"recall":results[3]['Recall'], "accuracy":results[0]['Accuracy'], "precision":results[1]['Precision'], "f1_score":results[2]['F1_score']}
        return {"model":train,"results":result, "totrain":clf}
        #return clf
        
    def redesNeurais(self, data, nfolds=10):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        scores = cross_validate(clf, features, labels, cv=nfolds, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        train = clf.fit(features, labels)
        
        # preds = clf.predict(test)
        
        print("\n\nRedes Neurais")
        results = []
        t = 0
        for i in scores['test_accuracy']:
            t += i
        results.append({"Accuracy": t/len(scores['test_accuracy'])})
        t = 0
        for i in scores['test_precision']:
            t += i
        results.append({"Precision": t/len(scores['test_precision'])})
        t = 0
        for i in scores['test_f1']:
            t += i
        results.append({"F1_score": t/len(scores['test_f1'])})
        t = 0
        for i in scores['test_recall']:
            t += i
        results.append({"Recall": t/len(scores['test_recall'])})
        
        result = {"recall":results[3]['Recall'], "accuracy":results[0]['Accuracy'], "precision":results[1]['Precision'], "f1_score":results[2]['F1_score']}
        return {"model":train,"results":result, "totrain":clf}
        #return clf
    
    def predictSimilarity(self, features, algorithm):
        data = Util.readJson(self.train+".json")
        data += Util.readJson(self.test+".json")
        data = self.datasetScikit(data, isRegression=True)
        
        if algorithm == "linearRegression":
            model = self.linearRegression(data)
        elif algorithm == "logisticRegression":
            model = self.logisticRegression(data)
        elif algorithm == "bayesianRidge":
            model = self.bayesianRidge(data)
        elif algorithm == "lasso":
            model = self.lasso(data)
        elif algorithm == "lassoLars":
            model = self.lassoLars(data)
        pred = model.predict(features)
        
        return pred
    
    def gerarArff(self, name):
        data = Util.readJson("../"+self.train+".json")
        data += Util.readJson(self.test+".json")
        Util.writeJson(data, "testeee")
        arff = open(name+'.arff','w')
        head = '@relation similarity-sentences\n'
        head += '@attribute rootEquals       {sim, nao}\n'
        head += '@attribute hasNeg           numeric\n'
        head += '@attribute entityQtd        numeric\n'
        head += '@attribute cosEmbleddings   real\n'
        head += '@attribute wmd              real\n'
        head += '@attribute tfidf            real\n'
        head += '@attribute qtdSin           numeric\n'
        head += '@attribute antonimos        numeric\n'
        head += '@attribute hiperonimos      numeric\n'
        head += '@attribute obj              {sim, nao}\n'
        head += '@attribute subj             {sim, nao}\n'
        head += '@attribute tokensEquals     numeric\n'
        head += '@attribute lemmasEquals     numeric\n'
        head += '@attribute parafraseamento  numeric\n'
        head += '@attribute class            {none, entailment}\n'
        head += '@data \n'
        arff.write(head)
        print(len(data))
        for i in range(0, len(data)):
            string = ''
            if data[i]['root']:
                string+='sim,'
            else:
                string+='nao,'
                
            string+=str(data[i]['neg'])+','
            string+=str(data[i]['entity'])+','
            string+=str(data[i]['cos_embeddings'])+','
            string+=str(data[i]['wmd'])+','    
            string+=str(data[i]['sim'])+','
            string+=str(data[i]['sinonimos'])+','
            string+=str(data[i]['antonimos'])+','
            string+=str(data[i]['hiperonimo'])+','
            if data[i]['objEquals']:
                string+='sim,'
            else:
                string+='nao,'
            if data[i]['subjEquals']:
                string+='sim,'
            else:
                string+='nao,'
            string+=str(data[i]['qtdTokensEquals'])+','
            string+=str(data[i]['qtdLemmasEquals'])+','
            string+=str(data[i]['parafraseamento'])+','
            string+=data[i]['entailment'].lower()+'\n'
            arff.write(string)
        arff.close()
    
    def testModels(self, feature_names, output):
        dataTrain = Util.readJson(self.train+".json")
        dataTest = Util.readJson(self.test+".json")
        test = self.datasetScikit(dataTest, feature_names)
        train = self.datasetScikit(dataTrain, feature_names)
        lresult = []
        randomForest = self.randomForest(train)
        randomForest = randomForest['model']
        svm = self.supportVM(train)
        svm = svm['model']
        
        resultRF = randomForest.predict(test['features'])
        resultSVM = svm.predict(test['features'])
        aux = []      
        print("\n\nRandom Forest: ")
        aux.append({"algoritmo":"Random Forest","Acuracy: ": accuracy_score(test['labels'], resultRF),
        "Precision: ": precision_score(test['labels'], resultRF),
        "F1 Score: ": f1_score(test['labels'], resultRF),
        "Average Precision: ": average_precision_score(test['labels'], resultRF),
        "Recall: ": recall_score(test['labels'], resultRF)
        })

        print("\n\nSupport Vector Machine: ")
        aux.append({"algoritmo":"Support Vector Machine","Acuracy: ": accuracy_score(test['labels'], resultSVM),
        "Precision: ": precision_score(test['labels'], resultSVM),
        "F1 Score: ": f1_score(test['labels'], resultSVM),
        "Average Precision: ": average_precision_score(test['labels'], resultSVM),
        "Recall: ": recall_score(test['labels'], resultSVM)
        })
    
        lresult = aux
        Util.writeJson(lresult, output)
        
    def gerarArqTest(self, entailment_predict, similarity_predict, output):
        base_path = os.path.dirname(os.path.realpath(__file__))
        fileTest = base_path+"\\assets\\corpus\\2016\\assin-ptbr-test.xml"
        test_pairs = read_xml(fileTest, need_labels=False)
        tree = ET.parse(fileTest)
        root = tree.getroot()
        for i in range(len(test_pairs)):
            pair = root[i]
            entailment_str = entailment_to_str[entailment_predict[i]]
            pair.set('entailment', entailment_str)
            pair.set('similarity', str(similarity_predict[i]))
        tree.write(output+".xml", 'utf-8')
    
    def testAlgorithms(self, feature_names, output):
        #train = Util.readJson("./assets/assin-mix-features-train.json")
        train = Util.readJson("./assets/test_features-assin-ptbr-train-ent.json")
        #train += Util.readJson("./assets/test_features-assin-ptpt-train-ent.json")
        #train += Util.readJson("./assets/test_features-assin-ptbr-test-ent.json")
        test = Util.readJson("./assets/test_features-assin-ptbr-test-ent.json")
        trainRegression = self.datasetScikit(train, feature_names, isRegression=True)
        trainClassifier = self.datasetScikit(train, feature_names)
        test = self.datasetScikit(test, feature_names)
        testpath = "./assets/corpus/2016/assin-ptbr-test.xml"
        
        
        entailment = self.randomForest(trainClassifier)
        entailment_predict = entailment['model'].predict(test['features'])
        
        similarity = self.linearRegression(trainRegression)        
        similarity_predict = similarity.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "assets/databases/assin/result_mix/linearRegression")
        linearR = self.assinEval(testpath, "./assets/databases/assin/result_mix/linearRegression.xml")
        
        similarity = self.logisticRegression(trainRegression)        
        similarity_predict = similarity.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "assets/databases/assin/result_mix/logisticRegression")
        logisticR = self.assinEval(testpath, "./assets/databases/assin/result_mix/logisticRegression.xml")
        
        similarity = self.bayesianRidge(trainRegression)       
        similarity_predict = similarity.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "assets/databases/assin/result_mix/bayesianRidge")
        bayesR = self.assinEval(testpath, "./assets/databases/assin/result_mix/bayesianRidge.xml")
        
        
        similarity = self.decisionTreeRegressor(trainRegression)       
        similarity_predict = similarity.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "assets/databases/assin/result_mix/decisionTree")
        decisionR = self.assinEval(testpath, "./assets/databases/assin/result_mix/decisionTree.xml")
        
        similarity = self.randomForestRegressor(trainRegression)       
        similarity_predict = similarity.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "assets/databases/assin/result_mix/randomForest")
        randomR = self.assinEval(testpath, "./assets/databases/assin/result_mix/randomForest.xml")
        
        
        
        # similarity = self.redesNeuraisRegressor(trainRegression)       
        # similarity_predict = similarity.predict(test['features'])
        # self.gerarArqTest(entailment_predict, similarity_predict, "neuralNetwort")
        
        similarity = self.supportVMR(trainRegression)       
        similarity_predict = similarity.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "assets/databases/assin/result_mix/svm")
        svmR = self.assinEval(testpath, "./assets/databases/assin/result_mix/svm.xml")

        results = {"randomForest":randomR, "svm":svmR}
        result = {"results":results, "features":feature_names}
        
        Util.writeJson(result, output)
        
    
    def crossvalTest(self, fnames, nfolds):
        cp = Util.readJson("./assets/"+self.train+".json")
        cp += Util.readJson("./assets/"+self.test+".json")
        data = self.datasetScikit(cp, fnames)

        try:
            result = Util.readJson("./assets/results_crossval.json")
        except:
            result = []
            
        rf = self.randomForest(data, nfolds)
        nb = self.naiveBayesTrain(data, nfolds)
        rn = self.redesNeurais(data, nfolds)
        dt = self.decisionTreeModel(data, nfolds)
        svm = self.supportVM(data, nfolds)
        result.append({"folds":nfolds,"features":fnames, "result":{"random-forest":rf['results'], "naive-bayes":nb['results'], "redes-neurais":rn['results'], "decision-tree":dt['results'], "svm":svm['results']}})
        
        Util.writeJson(result, "./assets/results_crossval")
        
    def percentSplitTest(self, fnames, percent):
        cp = Util.readJson(self.train+".json")
        cp += Util.readJson(self.test+".json")
        data = self.datasetScikit(cp, fnames)
        
        train_x, test_x, train_y, test_y = train_test_split(data['features'], data['labels'], test_size=percent, random_state=42)
        
        rf = self.randomForest(data)
        nb = self.naiveBayesTrain(data)
        rn = self.redesNeurais(data)
        dt = self.decisionTreeModel(data)
        svm = self.supportVM(data)
        
        
        print(len(train_x), len(train_y), len(test_x), len(test_y))
        rft = rf['totrain'].fit(train_x, train_y)
        nbt = nb['totrain'].fit(train_x, train_y)
        rnt = rn['totrain'].fit(train_x, train_y)
        dtt = dt['totrain'].fit(train_x, train_y)
        svmt = svm['totrain'].fit(train_x, train_y)
        
        
        rfp = rft.predict(test_x)
        nbp = nbt.predict(test_x)
        rnp = rnt.predict(test_x)
        dtp = dtt.predict(test_x)
        svmp = svmt.predict(test_x)
        
        testes = {
            "random-forest":{
                "precision":precision_score(test_y, rfp),
                "accuracy":accuracy_score(test_y, rfp),
                "recall":recall_score(test_y, rfp),
                "f1-score":f1_score(test_y, rfp)
            },
            "naive-bayes":{
                "precision":precision_score(test_y, nbp),
                "accuracy":accuracy_score(test_y, nbp),
                "recall":recall_score(test_y, nbp),
                "f1-score":f1_score(test_y, nbp)
            },
            "redes-neurais":{
                "precision":precision_score(test_y, rnp),
                "accuracy":accuracy_score(test_y, rnp),
                "recall":recall_score(test_y, rnp),
                "f1-score":f1_score(test_y, rnp)
            },
            "decision-tree":{
                "precision":precision_score(test_y, dtp),
                "accuracy":accuracy_score(test_y, dtp),
                "recall":recall_score(test_y, dtp),
                "f1-score":f1_score(test_y, dtp)
            },
            "suport-vector-machine":{
                "precision":precision_score(test_y, svmp),
                "accuracy":accuracy_score(test_y, svmp),
                "recall":recall_score(test_y, svmp),
                "f1-score":f1_score(test_y, svmp)
            },
        }
        
        try:
            ld = Util.readJson("./assets/results_percent.json")
        except:
            ld = []
            
        ld.append({"features":fnames, "percent":percent, "results":testes})
        
        Util.writeJson(ld, "./assets/results_percent")
            
        
        # rft.predict(train_y)
        
    def dataTraining(self):
        data = Util.readJson(self.train+".json")
        data += Util.readJson(self.test+".json")
        
        Util.writeJson(data,"training_model")
    
    def randomForestPredict(self, s1, s2):
        data = Util.readJson(self.train+".json")
        data += Util.readJson(self.test+".json")
        data = self.datasetScikit(data)
        rand = self.randomForest(data)
        #filename = "./corpus/2019/assin2-train-only"
        features = self.datasetScikit(self.ef.getFeaturesIndividual(s1, s2))
        
        if rand['model'].predict(features['features']):
            return True
        
        return False
        
    def assinEval(self, test, entrada):
        test = read_xml(test, True)
        entrada = read_xml(entrada, True)
        rte = assinEvalAdapt.eval_rte(test, entrada)
        similarity = assinEvalAdapt.eval_similarity(test, entrada)
        
        return {"rte":rte, "similarity":similarity}
    
        
if __name__ == "__main__":
    ef = ""
    
    train = SimilarityModel(ef, "./assets/test_features-assin-ptbr-train-ent", "./assets/test_features-assin-ptbr-test-ent")
    feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals', 'parafraseamento', 'hiperonimo', 'entity'] 
    train.testModels(feature_names, "./results/assin1/all features")
    feature_names = ['sim'] 
    train.testModels(feature_names, "./results/assin1/tf-idf")
    feature_names = ['cos_embeddings']
    train.testModels(feature_names, "./results/assin1/cos embeddings")
    feature_names = ['wmd']
    train.testModels(feature_names, "./results/assin1/wmd")
    feature_names = ['cos_embeddings', 'sim', 'wmd']
    train.testModels(feature_names, "./results/assin1/wmd-sim-coss_embeddings")
    #train.dataTraining()
# #     #train = SimilarityModel("test_features_assin2-train-only10.0", "test_features_assin2-test10.0")

# #     #train.randomForestTest()
#     ef = ""
#     #feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals','parafraseamento']
#     train = SimilarityModel(ef, "test_features_assin2-train-only10.0", "test_features_assin2-test10.0")
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals','hiperonimo', 'parafraseamento', 'entity']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals','hiperonimo', 'parafraseamento']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals','hiperonimo']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','parafraseamento']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','parafraseamento','qtdTokensEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','parafraseamento','qtdLemmasEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','parafraseamento','qtdLemmasEquals','qtdTokensEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
#     # feature_names = ['root','neg','cos_embeddings','wmd','sim','parafraseamento','qtdLemmasEquals','qtdTokensEquals', 'subjEquals']
#     # train.crossvalTest(feature_names, 5)
#     # train.crossvalTest(feature_names, 10)
#     # train.crossvalTest(feature_names, 15)
#     # train.crossvalTest(feature_names, 20)
#     # train.percentSplitTest(feature_names, 0.1)
#     # train.percentSplitTest(feature_names, 0.2)
#     # train.percentSplitTest(feature_names, 0.3)
#     # train.percentSplitTest(feature_names, 0.4)
#     # train.percentSplitTest(feature_names, 0.5)
    
    
    
#     feature_names = ['cos_embeddings','wmd']
#     train.crossvalTest(feature_names, 5)
#     train.crossvalTest(feature_names, 10)
#     train.crossvalTest(feature_names, 15)
#     train.crossvalTest(feature_names, 20)
#     train.percentSplitTest(feature_names, 0.1)
#     train.percentSplitTest(feature_names, 0.2)
    
#     feature_names = ['wmd']
#     train.crossvalTest(feature_names, 5)
#     train.crossvalTest(feature_names, 10)
#     train.crossvalTest(feature_names, 15)
#     train.crossvalTest(feature_names, 20)
#     train.percentSplitTest(feature_names, 0.1)
#     train.percentSplitTest(feature_names, 0.2)
    
#     feature_names = ['cos_embeddings']
#     train.crossvalTest(feature_names, 5)
#     train.crossvalTest(feature_names, 10)
#     train.crossvalTest(feature_names, 15)
#     train.crossvalTest(feature_names, 20)
#     train.percentSplitTest(feature_names, 0.1)
#     train.percentSplitTest(feature_names, 0.2)
    
#     feature_names = ['sim']
#     train.crossvalTest(feature_names, 5)
#     train.crossvalTest(feature_names, 10)
#     train.crossvalTest(feature_names, 15)
#     train.crossvalTest(feature_names, 20)
#     train.percentSplitTest(feature_names, 0.1)
#     train.percentSplitTest(feature_names, 0.2)
    #feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals']
    #feature_names = ['root','neg','cos_embeddings','wmd','sim','qtdTokensEquals','qtdLemmasEquals', 'parafraseamento']
    # train.crossvalTest(feature_names, 5)
    # train.crossvalTest(feature_names, 10)
    # train.crossvalTest(feature_names, 15)
    # train.crossvalTest(feature_names, 20)
#     print("-----------------------")
#     train.crossvalTest(feature_names)
    #train.gerarArff("test_222")
    
    