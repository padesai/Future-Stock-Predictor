import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):

        self.leaf_size = leaf_size;

        self.verbose = verbose;

    def author(self):

        return 'pdesai75'

    def addEvidence(self, Xtrain, Ytrain):

        self.model = self.__buildTree(Xtrain, Ytrain);

    def __buildTree(self, dataX, dataY):

        if (dataX.shape[0] <= self.leaf_size):
            u, indices = np.unique(dataY, return_inverse=True)
            return np.array([[-1, u[np.argmax(np.bincount(indices))], np.NaN, np.NaN]]);

        elif (len(set(dataY.tolist())) == 1):
            u, indices = np.unique(dataY, return_inverse=True)
            return np.array([[-1, u[np.argmax(np.bincount(indices))], np.NaN, np.NaN]]);

        else:

            randomFeature, splitVal, leftX, rightX, leftY, rightY = self.__findNonEmptyRandomSplit(dataX, dataY);
            a = 1;
            while ((leftX.size == 0 or leftY.size == 0 or rightX.size == 0 or rightY.size == 0) and a < 10):
                randomFeature, splitVal, leftX, rightX, leftY, rightY = self.__findNonEmptyRandomSplit(dataX, dataY);
                a = a + 1;

            if (a==10):
                u, indices = np.unique(dataY, return_inverse=True)
                return np.array([[-1, u[np.argmax(np.bincount(indices))], np.NaN, np.NaN]]);

            lefttree = self.__buildTree(leftX, leftY);
            righttree = self.__buildTree(rightX, rightY);
            root = np.array([[randomFeature, splitVal, 1, lefttree.shape[0] + 1]]);

            return np.vstack((root,lefttree,righttree));

    def __findNonEmptyRandomSplit(self,X,Y):
        factor = np.random.randint(low=0,high=X.shape[1]);
        r1 = np.random.randint(low=0,high=X.shape[0]);
        r2 = np.random.randint(low=0,high=X.shape[0]);

        if (r1 == r2):
            if (X.shape[0] == 2):
                r1 = 0;
                r2 = 1;

            else:
                r2 = np.random.randint(low=0,high=X.shape[0]);
                while (r1==r2):
                    r2 = np.random.randint(low=0,high=X.shape[0]);

        splVal = (X[r1, factor]+ X[r2, factor])/2;
        rowIndexL = X[:, factor] <= splVal;
        rowIndexR = X[:, factor] > splVal;
        lX = X[rowIndexL,];
        lY = Y[rowIndexL,];
        rX = X[rowIndexR,];
        rY = Y[rowIndexR,];
        return factor, splVal, lX, rX, lY, rY

    def query(self, Xtest):

        Y = np.apply_along_axis(self.__queryPT,1,Xtest);
        return Y;

    def __queryPT(self, Xarray):

        curr_node = 0;
        root = self.model[curr_node];
        fac, sv, indexL, indexR = root;

        while (fac != -1):

            if(Xarray[fac] <= sv):
                curr_node = indexL + curr_node;
                fac, sv, indexL, indexR = self.model[curr_node];

            else:

                curr_node = indexR + curr_node;
                fac, sv, indexL, indexR = self.model[curr_node];

        return sv

class BagLearner(object):

    def __init__(self,learner = RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.learner = learner;
        self.bags = bags;
        self.boost = boost;
        self.verbose = verbose;
        self.learners = [];
        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs));

    def author(self):
        return 'pdesai75'

    def addEvidence(self, Xtrain, Ytrain):
        for i in range(0,self.bags):
            indices = np.random.randint(Xtrain.shape[0], size=Xtrain.shape[0]);
            X = Xtrain[indices[:], :];
            Y = Ytrain[indices[:]];
            self.learners[i].addEvidence(X,Y);

    def query(self, Xtest):

        outputs = []
        for i in range(0,self.bags):
            output = self.learners[i].query(Xtest);
            outputs.append(output);

        outputs_array = np.array(outputs)
        axis = 0
        u, indices = np.unique(outputs_array, return_inverse=True)

        return u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(outputs_array.shape), None, np.max(indices) + 1), axis=axis)]

if __name__ == "__main__":
    pass