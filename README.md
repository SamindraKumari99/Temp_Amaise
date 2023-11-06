# AMAISE_PRO

for testing
python3 host_depletion.py -i <inputfile> -t <typefile> -o <outfolder> -m <model>
model = 'models_and_references/model_new'

for training
python3 train.py -m <pathtosavemodel> -i <trainset> -l <labelset>
pathtosavemodel  = 'models_and_references/model_new'
trainset = 'train_data/set1.csv'

for evaluating
python3 evaluation.py -o <fileToWrite> -t <truefile> -p <predfile>
fileToWrite = 'Testings/eval_summary.txt
truefile = 'train_data/human.csv'
predfile = 'Testings/test/mlprobs.txt'