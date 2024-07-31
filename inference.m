% training

% load parameters
fname = 'all_params.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
json = jsondecode(str);

% load data
data = load('data/mat/back6/back6_1.mat');

% load dl network
dl = EvtTransformer('evt',json,dlarray(data.pols,"BSTC"),dlarray(data.pixels,"BSTC"));

% inference
res = predict(dl.net,data.pols,data.pixels);





