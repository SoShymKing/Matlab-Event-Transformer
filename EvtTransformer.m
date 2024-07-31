classdef EvtTransformer < nnet.layer.Layer & nnet.layer.Acceleratable

    properties
        evt = dlnetwork;
        clf = dlnetwork;
        net = dlnetwork;
    end

    methods
        function layer = EvtTransformer(Name, json,tmp1,tmp2)
            layer.Description = "EvtTransformer";
            layer.Name = Name;
            layer.NumInputs = 2;
            layer.net = addLayers(layer.net, inputLayer([NaN 1 NaN json.backbone_params.token_dim],"BSTC", "Name","pol"));
            layer.net = addLayers(layer.net, inputLayer([NaN 1 NaN 2],"BSTC", "Name","pixel"));
            layer.evt = EvNetBackbone('EVT', ...
                json.backbone_params.pos_encoding, ...
                json.backbone_params.token_dim, ...
                json.backbone_params.embed_dim, ...
                json.backbone_params.num_latent_vectors, ...
                json.backbone_params.event_projection, ...
                json.backbone_params.preproc_events, ...
                json.backbone_params.proc_events, ...
                json.backbone_params.proc_memory, ...
                json.backbone_params.return_last_q, ...
                json.backbone_params.proc_embs, ...
                json.backbone_params.downsample_pos_enc, ...
                json.backbone_params.pos_enc_grad, ...
                json.data_params.batch_size);
            layer.clf = CLFBlock('CLF', json.backbone_params.embed_dim, json.clf_params.opt_classes);

            layer.net = addLayers(layer.net, layer.evt.net);
            layer.net = addLayers(layer.net, layer.clf.net);

            layer.net = connectLayers(layer.net,"pol", strcat(layer.evt.net.Name,'/',layer.evt.net.InputNames{1}));
            layer.net = connectLayers(layer.net,"pixel", strcat(layer.evt.net.Name,'/',layer.evt.net.InputNames{2}));
            layer.net = connectLayers(layer.net,strcat(layer.evt.net.Name,'/',layer.evt.net.OutputNames{1}), strcat(layer.clf.net.Name,'/',layer.clf.net.InputNames{1}));

            layer.net = initialize(layer.net);
            % analyzeNetwork(layer.net,tmp1,tmp2)

        end
        

        function Y = predict(layer,X1,X2)
            x1 = permute(X1,[1,3,4,2]);
            x2 = permute(X2,[1,3,4,2]);
            x1_s = size(x1);
            x2_s = size(x2);
            res = [];
            for i = 1:x1_s(2)
                Y = predict(layer.evt,x1(:,i,:,:),x2(:,i,:,:));
                [A,B] = max(predict(layer.clf,Y));
                res(end+1) = B-1;
            end
            Y = dlarray(reshape(res,1,1,length(res),1));
        end
    end
end