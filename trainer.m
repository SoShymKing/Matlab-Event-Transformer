% training

% load parameters
fname = 'all_params.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
json = jsondecode(str);

% set train data
datas = load('data/data_labels.mat');
test_set = datas.train_list;
ds_test = EvtMatDatastore(test_set,json.data_params.batch_size);
val_set = datas.val_list;
ds_val = EvtMatDatastore(val_set,json.data_params.batch_size);

% set train option
options = trainingOptions("adam", ...
    Plots="training-progress", ...
    InputDataFormats=["SBTC","SBTC"], ...
    TargetDataFormats="CBT", ...
    MaxEpochs=json.training_params.max_epochs, ...
    InitialLearnRate=json.optim_params.optim_params.lr, ...
    MiniBatchSize=json.data_params.batch_size, ... 
    CheckpointPath="chk", ...
    Shuffle = "never", ...
    ValidationData= ds_val,...
    Metrics= accuracyMetric(ClassificationMode = "multilabel"), ...
    Verbose=false);
% load init data
data = load('data/mat/back6/back6_1.mat');
% load dl network
dl = EvtTransformer('evt',json,dlarray(data.pols,"BSTC"),dlarray(data.pixels,"BSTC"));
net = trainnet_t(ds_test,dl.net,"crossentropy",options);





