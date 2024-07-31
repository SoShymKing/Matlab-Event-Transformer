classdef EvtMatDatastore < matlab.io.Datastore & ...
                       matlab.io.datastore.MiniBatchable
    
    properties
        Datastore
        Labels
        NumClasses
        SequenceDimension
        CurrentFileIndex
        Data % Cell array to store the data
        CurrentIndex % Current index for reading data
        MiniBatchSize % Mini batch size
    end
    
    properties(SetAccess = protected)
        NumObservations
    end

    properties(Access = private)
        % This property is inherited from Datastore
    end


    methods
        
        function ds = EvtMatDatastore(file_lists, batch_size)
            ds.Data = file_lists;
            ds.CurrentIndex = 1;
            ds.MiniBatchSize = batch_size; % Default mini batch size
        end

        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= numel(ds.Data);
        end

        function [data,info] = read(ds)            
            % Read one mini-batch batch of data
            if ~hasdata(ds)
                error('No more data to read.');
            end
            
            endIndex = min(ds.CurrentIndex + ds.MiniBatchSize - 1, numel(ds.Data));

            batchs = {};
            for i = ds.CurrentIndex:endIndex
                tmp = load(strcat(ds.Data(i).folder,'/',ds.Data(i).name));
                s = size(batchs,1);
                idx = s + 1;
                l = zeros(3,1,1);
                l(tmp.labels+1)=1;
                batchs{idx,1} = tmp.pols;
                batchs{idx,2} = tmp.pixels;
                batchs{idx,3} = l;
            end

            data = batchs;
            ds.CurrentIndex = endIndex + 1;
            info = struct(); % Return empty info structure
        end

        function reset(ds)
            ds.CurrentIndex = 1;
        end

        function dsNew = partition(ds, numPartitions, index)
            partitionSize = ceil(numel(ds.Data) / numPartitions);
            startIdx = (index - 1) * partitionSize + 1;
            endIdx = min(index * partitionSize, numel(ds.Data));
            dsNew = EvtMatDatastore(ds.Data(startIdx:endIdx),ds.MiniBatchSize);
        end

        function ds = set.MiniBatchSize(ds, batchSize)
            ds.MiniBatchSize = batchSize;
        end
        
    end 
    

    methods (Hidden = true)
        function frac = progress(ds)
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end

    end

end % end class definition