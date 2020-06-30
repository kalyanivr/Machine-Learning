import numpy as np

class Naive_Bayes():

    def __int__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))  # y is the target variable therefore represents the class


    def fit(self, X, y):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]  # pick examples from 1 specific class one at a time

            self.classes_mean[str(c)] = np.mean(X_c, axis = 0)
            self.classes_variance[str(c)] = np.var(X_c, axis = 0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]   #self.num_examples




    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.gaussian_density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)


    def gaussian_density_function(self, x, mean, variance):
        # Use a multi variate gaussian density function

        const = -self.num_feature/2 * np.log(2* np.pi) - 0.5 * np.sum(np.log(variance + self.eps))
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (variance + self.eps), 1)
        return const - probs

if __name__ == '__main__':
    X = np.loadtxt()  # Add file and delimited
    y = np.loadtxt()-1  # Add file

    NB = Naive_Bayes(X, y)
    NB.fit(X, y)
    y_pred = NB.predict(X)








