classdef MLPBlock < nnet.layer.Layer & nnet.layer.Acceleratable

    properties
        net = dlnetwork;
        add_x = false;
        mask = [];
        embed_dim;
        dropout;
        inner_net = dlnetwork;
        b_s;

        seq_init;
        mask_ptr = libpointer
        q_mask_ptr = libpointer
    end

    methods
        function layer = MLPBlock(Name,ipt_dim, in_embed_dim, init_layers, add_x_input, in_dropout, batch_size,varargin)
            layer.Description = "MLPBlock";
            layer.Name = Name;
            layer.b_s = batch_size;
            if add_x_input ~= false
                layer.add_x = true;
            end
            layer.embed_dim = in_embed_dim;
            layer.dropout = in_dropout;
            if in_dropout > 0.0
                layer.dropout = in_dropout;
            end    
            layer.net = layer.get_sequential_block(init_layers, ipt_dim);
            layer.net = addLayers(layer.net,[functionLayer(@(X1,X2,X3) layer.check_add_x(X1, X2,X3), Name="check_add",NumInputs=3, NumOutputs=1,InputNames=["in1","in2","in3"], Formattable = 1, Acceleratable=1)]);
            layer.net = connectLayers(layer.net,"size/x","check_add/in1");
            layer.net = connectLayers(layer.net,layer.net.OutputNames{1},"check_add/in2");
            layer.net = connectLayers(layer.net,"size/x_s","check_add/in3");
            layer.net = networkLayer(layer.net,Name = layer.Name);
        end
        function Y = get_sequential_block(layer, num_layers, ipt_dim)
            tempNet = functionLayer(@(X) layer.save_x_size(X),NumOutputs=2,Name = "size",OutputNames=["x","x_s"], Formattable = 1, Acceleratable=1);            % [inputLayer([NaN 1 NaN ipt_dim],"TSBC", "Name","z")]; %inputLayer [1,910,1,ipt_dim] "TBSC",
            for i = 1:length(num_layers)
                l = num_layers{i};
                l_split = split(l,'_');
                l_name = l_split{1};
                opt_dim = str2num(l_split{2});
                activation = l_split{3};
                if isempty(opt_dim) || opt_dim == -1
                    opt_dim = layer.embed_dim;
                end
                if layer.dropout
                    tempNet(end+1) = dropoutLayer(layer.dropout,"Name","dropout"+i);
                end
                % Layer type
                if strcmp(l_name,'ff')
                    tempNet(end+1) = fullyConnectedLayer(opt_dim,"Name","fc"+i);
                end
                % Layer activation
                if strcmp(activation,'rel')
                    tempNet(end+1) = reluLayer("Name","relu"+i);
                elseif strcmp(activation,'gel')
                    tempNet(end+1) = geluLayer("Name","gelu"+i);
                end
                ipt_dim = opt_dim;
            end
            layer.inner_net = addLayers(layer.inner_net,tempNet);
            clear tempNet;
            Y = layer.inner_net;
        end

        function res = check_add_x(layer,X,Y,x_t)
            x_s = size(X);
            if isNull(layer.mask_ptr) == false || isempty(layer.mask_ptr.DataType) == false
                sz = size(layer.mask_ptr.Value);
                layer.mask_ptr.Value = reshape(layer.mask_ptr.Value,sz(2),sz(1));
            end
            if layer.add_x
                Y = reshape(Y,x_s);
                Y = Y + X;
                y_s = size(Y);
                res = dlarray(reshape(Y,y_s(2),y_s(3),y_s(4)),"CSB");
            else
                res = Y;
            end
        end
        
        function [Y1,Y2, layer] = save_x_size(layer,X)
            if length(size(X)) == 3
                Y1 = dlarray(X,"BCTS");
                if size(Y1,1)==layer.b_s
                    Y1 = dlarray(X,"BCTS");
                end
            else
                Y1 = dlarray(X,"SCBT");
                if size(Y1,1)==layer.b_s
                    Y1 = dlarray(X,"BCTS");
                end
            end
            s_y = size(Y1);
            if s_y(1)*s_y(2)~=160 && s_y(1)*s_y(2)~=144 && s_y(1)*s_y(2)~=128
                Y1 = dlarray(Y1,"BCTS");
            end
            Y2 = dlarray(zeros(size(X)),"SCBT");
        end
        function Y = predict(layer,X, varargin)
            Y = predict(layer.net,X);
        end
    end
end