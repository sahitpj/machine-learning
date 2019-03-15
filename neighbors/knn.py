import numpy
import scipy.spatial.distance

class KNearestNeighbors(object):
    def __init__(self, X_train, Y_train, X_test, n, type_, distance_funtion):
        self.X = X_train
        self.Y = Y_train
        self.test = X_test
        self.distance_function = distance_function
        self.type = type_
        self.samples = self.X.shape[0]
        self.test_samples = self.test.shape[0]
        self.n = n
        if self.distance_function != 'cosine' and self.distance_function != 'manhatten' and self.distance_function != 'euclidean':
            raise ValueError('Distance function not valid. please type a valid distance function: cosine, manhatten, euclidean')

        if self.type_ != 'classification' and self.type_ != 'regression':
            raise ValueError('KNN type not valid. Please type a valid KNN function: classificatio, regression')

    
    def cosine_dis(self, a, b):
        return scipy.spatial.distance.cosine(a,b)
    
    def euclidean_dis(self, a, b):
        return scipy.spatial.distance.euclidean(a,b)

    def manhatten_dis(self, a, b):
        return scipy.spatial.distance.cityblock(a,b)

    def predict_indi(self, point):
        n_keys = list()
        distances = {}
        for i in range(self.samples):
            if self.distance_function == 'cosine':
                distances[i] = self.cosine_dis(point, self.X[i, :])
            elif self.distance_function == 'manhatten':
                distances[i] = self.manhatten_dis(point, self.X[i, :])
            elif self.distance_function == 'euclidean':
                distances[i] = self.euclidean_dis(point, self.X[i, :])
        sorted_dict = sorted(distances.items(), key=operator.itemgetter(1))
        n_keys = sorted_dict.keys()[:self.n]
        if self.type_ == 'classification'
            classes = {}
            for i in range(len(n_keys)):
                try:
                    classes[self.Y[n_keys[i]]] += 1
                except:
                    classes[self.Y[n_keys[i]]] = 1
            sorted_classes = sorted(classes.items(), key=operator.itemgetter(1))
            return sorted_classes.keys()[-1]
        elif self.type_ == 'regression':
            l = []
            for i in range(len(n_keys)):
                l.append(self.Y[n_keys[i]])
            return sum(l)/len(l)
        
    def predict(self):
        all_ans = list()
        for i in range(self.test_samples):
            ans = self.predict_indi(X_test[i, :])
            all_ans.append(ans)
        return all_ans

    
                
            



