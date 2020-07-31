from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    
    def naiveBayesTrain(self, data):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        #train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
        # Inicializar classificador
        gnb = GaussianNB()
        scores = cross_validate(gnb, features, labels, cv=10, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        # Treinar classificador
        gnb = gnb.fit(features, labels)
        # # fazer predições
        # preds = gnb.predict(test)
        
        print("\n\nNaive Bayes")
        
        # print(preds)
        # #avaliar precisão    
        # print(accuracy_score(test_labels, preds))

        t = 0
        for i in scores['test_accuracy']:
            t += i
        print("Accuracy: ", t/len(scores['test_accuracy']))
        t = 0
        for i in scores['test_precision']:
            t += i
        print("Precision", t/len(scores['test_precision']))
        t = 0
        for i in scores['test_average_precision']:
            t += i
        print("Average Precision", t/len(scores['test_average_precision']))
        t = 0
        for i in scores['test_f1']:
            t += i
        print("F1 score:", t/len(scores['test_f1']))
        t = 0
        for i in scores['test_recall']:
            t += i
        print("Recall:", t/len(scores['test_recall']))
        
        return gnb

    def supportVM(self, data):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        #train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
        # Inicializar classificador
        clf = SVC(gamma='auto')
        
        scores = cross_validate(clf, features, labels, cv=10, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        # Treinar classificador
        clf = clf.fit(features, labels)
        
        # # fazer predições
        # preds = clf.predict(test)
        
        print("\n\nSupport Vector Machine")
        
        # print(preds)
        # #avaliar precisão    
        # print(accuracy_score(test_labels, preds))
        
        t = 0
        for i in scores['test_accuracy']:
            t += i
        print("Accuracy: ", t/len(scores['test_accuracy']))
        t = 0
        for i in scores['test_precision']:
            t += i
        print("Precision", t/len(scores['test_precision']))
        t = 0
        for i in scores['test_average_precision']:
            t += i
        print("Average Precision", t/len(scores['test_average_precision']))
        t = 0
        for i in scores['test_f1']:
            t += i
        print("F1 score:", t/len(scores['test_f1']))
        t = 0
        for i in scores['test_recall']:
            t += i
        print("Recall:", t/len(scores['test_recall']))
        
        
        return clf
        
    def decisionTreeModel(self, data):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        #train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

        clf = DecisionTreeClassifier()
        scores = cross_validate(clf, features, labels, cv=10, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        clf = clf.fit(features, labels)
        # preds = clf.predict(test)
        
        print("\n\nArvore de decisão")
        
        # print(preds)
        
        # print(accuracy_score(test_labels, preds))
        
        t = 0
        for i in scores['test_accuracy']:
            t += i
        print("Accuracy: ", t/len(scores['test_accuracy']))
        t = 0
        for i in scores['test_precision']:
            t += i
        print("Precision", t/len(scores['test_precision']))
        t = 0
        for i in scores['test_average_precision']:
            t += i
        print("Average Precision", t/len(scores['test_average_precision']))
        t = 0
        for i in scores['test_f1']:
            t += i
        print("F1 score:", t/len(scores['test_f1']))
        t = 0
        for i in scores['test_recall']:
            t += i
        print("Recall:", t/len(scores['test_recall']))
        
        return clf

    def randomForest(self, data):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        clf = RandomForestClassifier()
        print("\n\nRandom Forest")
        scores = cross_validate(clf, features, labels, cv=10,  return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        
        clf = clf.fit(features, labels)
        
        t = 0
        for i in scores['test_accuracy']:
            t += i
        print("Accuracy: ", t/len(scores['test_accuracy']))
        t = 0
        for i in scores['test_precision']:
            t += i
        print("Precision", t/len(scores['test_precision']))
        t = 0
        for i in scores['test_average_precision']:
            t += i
        print("Average Precision", t/len(scores['test_average_precision']))
        t = 0
        for i in scores['test_f1']:
            t += i
        print("F1 score:", t/len(scores['test_f1']))
        t = 0
        for i in scores['test_recall']:
            t += i
        print("Recall:", t/len(scores['test_recall']))
        
        return clf
        
    def redesNeurais(self, data):
        features = data['features']
        labels = data['labels']
        label_names = data['label_names']
        feature_names = data['feature_names']
        
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        scores = cross_validate(clf, features, labels, cv=10, return_train_score=False, scoring=
                                ['accuracy',
                                'average_precision',
                                'f1',
                                'precision',
                                'recall',
                                'roc_auc'])
        clf = clf.fit(features, labels)
        
        # preds = clf.predict(test)
        
        print("\n\nRedes Neurais")
        
        # print(preds)

        # print(accuracy_score(test_labels, preds))
        t = 0
        for i in scores['test_accuracy']:
            t += i
        print("Accuracy: ", t/len(scores['test_accuracy']))
        t = 0
        for i in scores['test_precision']:
            t += i
        print("Precision", t/len(scores['test_precision']))
        t = 0
        for i in scores['test_average_precision']:
            t += i
        print("Average Precision", t/len(scores['test_average_precision']))
        t = 0
        for i in scores['test_f1']:
            t += i
        print("F1 score:", t/len(scores['test_f1']))
        t = 0
        for i in scores['test_recall']:
            t += i
        print("Recall:", t/len(scores['test_recall']))
        
        
        return clf
    
    def predictSimilarity(self, features):
        data = Util.readJson(self.train+".json")
        data = self.datasetScikit(data, isRegression=True)
        model = self.linearRegression(data)
        
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
    
    def testModels(self, feature_names):
        dataTrain = Util.readJson(self.train+".json")
        train = self.datasetScikit(dataTrain, feature_names)
        dataTest = Util.readJson(self.test+".json")
        test = self.datasetScikit(dataTest, feature_names)
        lresult = []
        neuralNetwork = self.redesNeurais(train)
        naiveBayes = self.naiveBayesTrain(train)
        randomForest = self.randomForest(train)
        decisionTree = self.decisionTreeModel(train)
        svm = self.supportVM(train)
        
        resultNN = neuralNetwork.predict(test['features'])
        resultNB = naiveBayes.predict(test['features'])
        resultRF = randomForest.predict(test['features'])
        resultDT = decisionTree.predict(test['features'])
        resultSVM = svm.predict(test['features'])
        aux = []
        print("\n\nRedes neurais: ")
        aux.append({"algoritmo":"Redes neurais","Acuracy: ":accuracy_score(test['labels'], resultNN),
        "Precision: ": precision_score(test['labels'], resultNN),
        "F1 Score: ": f1_score(test['labels'], resultNN),
        "Average Precision: ": average_precision_score(test['labels'], resultNN),
        "Recall: ": recall_score(test['labels'], resultNN),
        })

        print("\n\nNaive Bayes: ")
        aux.append({"algoritmo":"Naive Bayes", "Acuracy: ": accuracy_score(test['labels'], resultNB),
        "Precision: ": precision_score(test['labels'], resultNB),
        "F1 Score: ": f1_score(test['labels'], resultNB),
        "Average Precision: ": average_precision_score(test['labels'], resultNB),
        "Recall: ": recall_score(test['labels'], resultNB)
        })
                   
        print("\n\nRandom Forest: ")
        aux.append({"algoritmo":"Random Forest","Acuracy: ": accuracy_score(test['labels'], resultRF),
        "Precision: ": precision_score(test['labels'], resultRF),
        "F1 Score: ": f1_score(test['labels'], resultRF),
        "Average Precision: ": average_precision_score(test['labels'], resultRF),
        "Recall: ": recall_score(test['labels'], resultRF)
        })

        print("\n\nDecision Tree: ")
        aux.append({"algoritmo":"Decision Tree","Acuracy: ": accuracy_score(test['labels'], resultDT),
        "Precision: ": precision_score(test['labels'], resultDT),
        "F1 Score: ": f1_score(test['labels'], resultDT),
        "Average Precision: ": average_precision_score(test['labels'], resultDT),
        "Recall: ": recall_score(test['labels'], resultDT)
        })
        
        print("\n\nSupport Vector Machine: ")
        aux.append({"algoritmo":"Support Vector Machine","Acuracy: ": accuracy_score(test['labels'], resultSVM),
        "Precision: ": precision_score(test['labels'], resultSVM),
        "F1 Score: ": f1_score(test['labels'], resultSVM),
        "Average Precision: ": average_precision_score(test['labels'], resultSVM),
        "Recall: ": recall_score(test['labels'], resultSVM)
        })
    
        lresult = aux
        Util.writeJson(lresult, "resutados1")
        
    def gerarArqTest(self, entailment_predict, similarity_predict, output):
        base_path = os.path.dirname(os.path.realpath(__file__))
        fileTest = base_path+"\\corpus\\2019\\assin2-test.xml"
        test_pairs = read_xml(fileTest, need_labels=False)

        tree = ET.parse(fileTest)
        root = tree.getroot()
        for i in range(len(test_pairs)):
            pair = root[i]
            entailment_str = entailment_to_str[entailment_predict[i]]
            pair.set('entailment', entailment_str)
            pair.set('similarity', str(similarity_predict[i]))

        tree.write(output+".xml", 'utf-8')
    def testAlgorithms(self, output, feature_names):
        train = Util.readJson("../"+self.train+".json")
        test = Util.readJson("../"+self.test+".json")
        trainRegression = self.datasetScikit(train, feature_names, isRegression=True)
        trainClassifier = self.datasetScikit(train, feature_names)
        test = self.datasetScikit(test, feature_names)
        
        similarity = self.linearRegression(trainRegression)        
        similarity_predict = similarity.predict(test['features'])
        
        entailment = self.randomForest(trainClassifier)
        entailment_predict = entailment.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "randomForest")
        
        entailment = self.naiveBayesTrain(trainClassifier)
        entailment_predict = entailment.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "naiveBayes")
        
        entailment = self.redesNeurais(trainClassifier)
        entailment_predict = entailment.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "redesNeurais")
        
        entailment = self.supportVM(trainClassifier)
        entailment_predict = entailment.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "svm")
        
        entailment = self.decisionTreeModel(trainClassifier)
        entailment_predict = entailment.predict(test['features'])
        self.gerarArqTest(entailment_predict, similarity_predict, "decisionTree")
    
    def crossvalTest(self, fnames):
        cp = Util.readJson("../"+self.train+".json")
        cp += Util.readJson(self.test+".json")
        data = self.datasetScikit(cp, fnames)
        
        self.randomForest(data)
        self.naiveBayesTrain(data)
        self.redesNeurais(data)
        self.decisionTreeModel(data)
        self.supportVM(data)
    
    def randomForestPredict(self, s1, s2):
        data = Util.readJson(self.train+".json")
        data += Util.readJson(self.test+".json")
        data = self.datasetScikit(data)
        rand = self.randomForest(data)
        #filename = "./corpus/2019/assin2-train-only"
        features = self.datasetScikit(self.ef.getFeaturesIndividual(s1, s2))
        
        if rand.predict(features['features']):
            return True
        
        return False
        
        
    
        
# if __name__ == "__main__":
#     #train = SimilarityModel("test_features_assin2-train-only10.0", "test_features_assin2-test10.0")
    
#     #train.randomForestTest()
#     ef = ""
#     feature_names = ['root','neg','cos_embeddings','wmd','sim','sinonimos','subjEquals','antonimos','objEquals','qtdTokensEquals','qtdLemmasEquals','parafraseamento','hiperonimo']
#     train = SimilarityModel(ef, "test_features_assin2-train-only10.0", "../test_features_assin2-test10.0")
#     print("-----------------------")
#     train.crossvalTest(feature_names)
    #train.gerarArff("test_222")
    
    