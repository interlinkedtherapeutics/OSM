import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
import numpy
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt 

from scipy.stats import ks_2samp
from sklearn import tree


def read_data(file_name):
    '''
    Creates a dataframe of the input file
    '''
    df = pd.read_csv(file_name)
    return df

def segment_data(data):
    '''
    Performs sklearns train/test split with 15% leave out 
    '''
    X_train, X_test, y_train, y_test = train_test_split(data['Smiles'],data['Ion Regulation Activity'] , test_size=0.20)
    
    return X_train, X_test, y_train, y_test


def generate_fingerprints(mols):
    '''
    Generates a molecular fingerprint; an arbtrarily long array of signed bits (0 or 1),
    representing the presence of different molecular groups within a compound 

    Args:
        mols (array(RDKit obj)) : object representing the molecular representaiton of a compound in RDKit
    
    Returns:         
        fps (array of arrays) : 2048 signed bit array representation of a given compound
    '''

    temp = []
    fp = [AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048) for x in mols]  
    
    for arr in fp:
        temp.append([int(x) for x in arr])
    fps = numpy.array(temp, dtype=float)

    return fps 


def knn(x_train, x_test, y_train, y_test):
    '''
    Fits the data to a KNN model (3 neighbors)
    '''
    
    model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)
    
    return score 

def rf(x_train, x_test, y_train, y_test):
    '''
    Fits the data to a SVM model
    '''

    model = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)

    return score 

def nb(x_train, x_test, y_train, y_test):
    model = GaussianNB().fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)

    return score 

def lin_svm(x_train, x_test, y_train, y_test):
    '''
    Fits the data to a SVM model
    '''

    model = svm.SVC().fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)

    return score 

def dt(x_train, x_test, y_train, y_test):
    model = tree.DecisionTreeClassifier().fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)
    
    return score



def guassian_process(x_train, x_test, y_train, y_test):
    '''
    Fits data using a Gaussian Process and makes predictions 
    '''

    gpc = GaussianProcessClassifier().fit(x_train, y_train)
    predictions = gpc.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)
    
    return score

def main():
    osm_data = read_data('data/osm_itx.csv')
    
    #Drop all of the molecules that have '0.5' in the Ion Regulation Activity columns 
    osm_data = osm_data[osm_data['Ion Regulation Activity'] != '0.5']
    mcc_svm = []
    mcc_gaussian = [] 
    mcc_knn = []
    mcc_rf = [] 
    mcc_nb = []
    mcc_dt = []

    # Compare each model 100 times, and get distributions of MCC values 
    for _ in range(150):
        X_train, X_test, y_train, y_test = segment_data(osm_data)
        
        # Convert Smiles to mols
        x_train_mols = [Chem.MolFromSmiles(x) for x in X_train]
        x_test_mols = [Chem.MolFromSmiles(x) for x in X_test]
        x_train_fp = generate_fingerprints(x_train_mols)
        x_test_fp = generate_fingerprints(x_test_mols)
        
        # Recast the data into int     
        y_train = [int(x) for x in y_train]
        y_test = [int(x) for x in y_test]

        mcc_gaussian.append(guassian_process(x_train_fp, x_test_fp, y_train, y_test))
        mcc_knn.append(knn(x_train_fp, x_test_fp, y_train, y_test))
        mcc_svm.append(lin_svm(x_train_fp, x_test_fp, y_train, y_test))
        mcc_rf.append(rf(x_train_fp, x_test_fp, y_train, y_test))
        mcc_nb.append(nb(x_train_fp, x_test_fp, y_train, y_test))
        mcc_dt.append(dt(x_train_fp, x_test_fp, y_train, y_test))

    all_mcc = zip(mcc_gaussian, mcc_knn, mcc_rf, mcc_nb, mcc_dt)
    all_mcc = pd.DataFrame(all_mcc)
    all_mcc.columns = ['Gaussian', 'KNN', 'SVM', 'RF', 'NB', 'DecisionTree']
    all_mcc.to_csv('all_mcc.csv', index=False)
    fig, ax = plt.subplots()
    all_mcc = all_mcc.reindex(all_mcc.mean().sort_values().index, axis=1)
    g = sns.violinplot(data=all_mcc, palette='GnBu_d')
    g.set_xticklabels(['Gaussian', 'KNN', 'SVM', 'RF', 'NB', 'DecisionTree'])
    plt.show()

    all_models =  [mcc_gaussian, mcc_knn, mcc_svm, mcc_rf, mcc_nb, mcc_dt]
    model_names = ['mcc_gaussian','mcc_knn', 'mcc_svm', 'mcc_rf', 'mcc_nb','mcc_dt']
    results = [] 
    
    for i, model in enumerate(all_models):
        for z, comparison in enumerate(all_models):
            _ , p_val = ks_2samp(model, comparison)
            results.append([model_names[i], model_names[z], p_val])
    

    #print(results)
    df = pd.DataFrame(results)
    df.columns = ['Method A', 'Method B', 'p-value']
    df.to_csv('MethodComparison.csv', index=False)

    

if __name__ == '__main__':

    main()