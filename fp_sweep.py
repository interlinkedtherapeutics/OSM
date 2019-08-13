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
from sklearn.svm import LinearSVC
from scipy.stats import ks_2samp
from sklearn import tree
from sklearn.linear_model import LogisticRegression

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


def generate_fingerprints(mols, bit_num, depth):
    '''
    Generates a molecular fingerprint; an arbtrarily long array of signed bits (0 or 1),
    representing the presence of different molecular groups within a compound 

    Args:
        mols (array(RDKit obj)) : object representing the molecular representaiton of a compound in RDKit
    
    Returns:         
        fps (array of arrays) : 2048 signed bit array representation of a given compound
    '''

    temp = []
    fp = [AllChem.GetMorganFingerprintAsBitVect(x, depth, nBits=bit_num) for x in mols]  
    
    for arr in fp:
        temp.append([int(x) for x in arr])
    fps = numpy.array(temp, dtype=float)

    return fps 

def lr(x_train, x_test, y_train, y_test):
    '''
    Fits the data to a LR model
    '''
    model = LogisticRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)
    
    return score

def lin_svm(x_train, x_test, y_train, y_test):
    '''
    Fits the data to a SVM model
    '''

    model = LinearSVC().fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = matthews_corrcoef(y_test, predictions)

    return score 


def main():
    osm_data = read_data('data/osm_itx.csv')
    
    #Drop all of the molecules that have '0.5' in the Ion Regulation Activity columns 
    osm_data = osm_data[osm_data['Ion Regulation Activity'] != '0.5']
    mcc_svm = []
    mcc_lr = []
    
    # Output 
    output = {} 

    temp_df = pd.DataFrame() 

    # Compare each model 100 times, and get distributions of MCC values 
    for _ in range(125):
        X_train, X_test, y_train, y_test = segment_data(osm_data)
        
        # Define range of bits 
        bits = [1024, 2048, 4096]
        depth = [4,5,6]
        
        for bit_num in bits:
            for d in depth:
                mcc_svm = [] 
                mcc_lr = [] 
                # Convert Smiles to mols
                x_train_mols = [Chem.MolFromSmiles(x) for x in X_train]
                x_test_mols = [Chem.MolFromSmiles(x) for x in X_test]
                x_train_fp = generate_fingerprints(x_train_mols, bit_num, d)
                x_test_fp = generate_fingerprints(x_test_mols, bit_num, d)
                
                # Recast the data into int     
                y_train = [int(x) for x in y_train]
                y_test = [int(x) for x in y_test]

                #mcc_gaussian.append(guassian_process(x_train_fp, x_test_fp, y_train, y_test))
                mcc_svm.append(lin_svm(x_train_fp, x_test_fp, y_train, y_test))
                mcc_lr.append(lr(x_train_fp, x_test_fp, y_train, y_test))
                if '{}_{}_{}'.format('svm', d, bit_num) in output:
                    output['{}_{}_{}'.format('svm', d, bit_num)].append(lin_svm(x_train_fp, x_test_fp, y_train, y_test))
                else:
                    output['{}_{}_{}'.format('svm', d, bit_num)] = [lin_svm(x_train_fp, x_test_fp, y_train, y_test)]
                
                if '{}_{}_{}'.format('lr', d, bit_num) in output:
                    output['{}_{}_{}'.format('lr', d, bit_num)].append(lin_svm(x_train_fp, x_test_fp, y_train, y_test))
                else:
                    output['{}_{}_{}'.format('lr', d, bit_num)] = [lin_svm(x_train_fp, x_test_fp, y_train, y_test)]
                
                
                
                #temp_df['{}_{}_{}'.format('lr', d, bit_num)] = mcc_lr 
    df = pd.DataFrame(output)
    g = sns.violinplot(data=df)
    plt.show()
    df.to_csv('parameter_sweep.csv', index=False)

    

if __name__ == '__main__':

    main()