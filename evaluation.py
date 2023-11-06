from sklearn.metrics import accuracy_score, confusion_matrix
import sys, getopt
import pandas as pd

try:
    opts, args = getopt.getopt(sys.argv[1:], 'f:t:p:')
except getopt.GetoptError:
    sys.exit(2)
    
for opt, arg in opts:
    if opt == '-h':
        sys.exit()
    elif opt in ("-o", "--outfile"):
        fileToWrite = arg
    elif opt in ("-t", "--truefile"):
        truefile = arg
    elif opt in ("-p", "--predfile"):
        predfile = arg

preds = []
with open(predfile, 'r') as f:
      for line in f:
         preds.append(line.split(', ')[1])
         
true = []
true_df = pd.read_csv(truefile).to_numpy()
for line in true_df:
    true.append(line[4])
         
preds = preds[:481701]
true = true[:481701]

accuracy = accuracy_score(true, preds)
tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
sens =  tp/(tp + fn)
spec = tn/(tn + fp)

print(accuracy, sens, spec)
with open(fileToWrite, 'w') as f:
    f.write('Accuracy: %0.10f\n'%accuracy)
    f.write('Sensitivity: %0.10f\n'%sens)
    f.write('Specificity: %0.10f\n'%spec)