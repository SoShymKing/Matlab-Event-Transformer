% preprocess
% load parameters
fname = 'all_params.json'; 
afid = fopen(fname); 
raw = fread(afid,inf); 
str = char(raw'); 
fclose(afid); 
json = jsondecode(str);

datasets = natsortfiles(dir('data/mat'));
datasets = datasets(3:end);

for l = 1:length(datasets)
    base_dir = strcat('data/pckl/',datasets(l).name);
    % load evt dir
    evt_dir = natsortfiles(dir(strcat(base_dir,'/evt')));
    evt_dir = evt_dir(3:end);
    % laod pdl dir
    pdl_dir = natsortfiles(dir(strcat(base_dir,'/pdl')));
    pdl_dir = pdl_dir(3:end);
    parfor i = 1:length(evt_dir)
        % laod evt datas
        j = evt_dir(i);
        name = split(j.name,'_');
        afdir = strcat(j.folder,'/',j.name);
        afid = py.open(afdir,'rb');
        atmp = py.pickle.load(afid);
        evts = double(atmp.coords);
        % laod pdl datas
        k = pdl_dir(i);
        bfdir = strcat(k.folder,'/',k.name);
        bfid = py.open(bfdir,'rb');
        btmp = py.pickle.load(bfid);
        pdls = double(btmp);
        % preprocess
        [pol,pixel,label] = getitem(json,evts,pdls,true); 
        [pols,pixels,labels] = collate({pol;pixel;label});
        save_name = strcat(name{1},'_',int2str(i));
        save_dir = strcat('data/mat/',name{1},'/',save_name);
        save(save_dir,"-fromstruct",struct("pols",pols,"pixels",pixels,"labels",labels));
    end
end

