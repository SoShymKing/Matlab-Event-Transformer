
% load parameters
fname = 'all_params.json'; 
afid = fopen(fname); 
raw = fread(afid,inf); 
str = char(raw'); 
fclose(afid); 
json = jsondecode(str);

base_dir = 'data/mat';
f_dir = natsortfiles(dir(base_dir));
f_dir = f_dir(3:end);

f_list = [];
for f_index = 1:length(f_dir)
    tmp = natsortfiles(dir(strcat(base_dir,'/',f_dir(f_index).name)));
    f_list = cat(1,f_list,tmp(3:end));
end

val_ind = sort(randsample(length(f_list),floor(length(f_list)/3)));
train_list = f_list; 
val_list = f_list(1);
for i = 1:length(val_ind)
    index = val_ind(i)-i+1;
    val_list(i) = train_list(index);
    if index == 1
        train_list = train_list(2:end);
    else
        train_list = cat(1,train_list(1:index-1),train_list(index+1:end));
    end
end
val_list = val_list';

save data/data_labels_s train_list val_list