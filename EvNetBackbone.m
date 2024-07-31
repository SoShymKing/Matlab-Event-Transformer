classdef EvNetBackbone < nnet.layer.Layer & nnet.layer.Acceleratable

    properties
        net = dlnetwork;
        clf
        emb
        return_last_q
        num_latent_vectors
        downsample_pos_enc
        memory_vertical
        pos_encoding
        preproc_events
        event_projection
        preproc_block_events
        proc_event_blocks
        proc_memory_blocks
        proc_embs_block
        q_mask_ptr = libpointer
        mask_ptr = libpointer
        test
        block_batch_s
    end

    methods
        function layer = EvNetBackbone(Name, ...
                 pos_encoding,...                  % Positional encoding params -> {name, {params}}
                 token_dim,...                     % Size of the flattened patch_tokens
                 embed_dim,...                     % Dimensionality of the latent_vectors and patch tokens
                 num_latent_vectors,...            % Number of latent vectors
                 event_projection,...              % Event Pre-processing Net {name, {params}}
                 preproc_events, ...               % Event Pre-processing Net (after positional encoding) {name, {params}}
                 proc_events,  ...                 % Event Processing Net (with skip connection) {name, {params}}
                 proc_memory,   ...                % Attention Layers {name, {params}}
                 return_last_q,   ...              % Return processed latent vectors or updated ones
                 proc_embs,       ...              % Latent Vectors summarization Net {clf_mode, params}
                 downsample_pos_enc,   ...         % Minimize positional encoding size
                 pos_enc_grad,    ...              % Learnable latent vectors? =False
                 batch_size, ...
                 varargin)
            
            layer.Description = "EvNetBackbone";
            layer.Name = Name;
            layer.block_batch_s = batch_size;

            layer.return_last_q = return_last_q;

            layer.num_latent_vectors = num_latent_vectors;
            layer.downsample_pos_enc = downsample_pos_enc;
            layer.memory_vertical = clip(random(makedist('Normal','mu',0.0,'sigma',0.2),num_latent_vectors, embed_dim),-2,2);
            layer.emb = event_projection.params.init_layers;
            layer.emb = split(layer.emb,'_');
            layer.emb = str2num(layer.emb{2});
    
            % Positional encodings
            if isempty(pos_encoding) == false 
                if strcmp(pos_encoding.name,'fourier') 
                    if pos_encoding.params.bands == -1
                        pos_encoding.params.bands = quorem(sym(embed_dim),sym(4));
                    end
                    % frame_shape, fourier_bands
                    pos_enc_params = pos_encoding.params; 
                    pos_enc_params.shape = [quorem(sym(pos_encoding.params.shape(1)),sym(downsample_pos_enc)) quorem(sym(pos_encoding.params.shape(2)),sym(downsample_pos_enc))];
                    
                    layer.pos_encoding = permute(fourier_features(reshape(pos_encoding.params.shape,1,[]),pos_encoding.params.bands), [2 3 1]);
                    pe_s = size(layer.pos_encoding);
                    pos_emb_dim = pe_s(3);
                else
                end
            else
                layer.pos_encoding = [];
            end
                
            
            % Event pre-proc block -> Linear transformation on tokens
            event_projection.params.embed_dim = embed_dim;
            % event_projection.params.init_layers
            if  strcmp(event_projection.name,'MLP')
                layer.event_projection = get_block(event_projection.name,'Block_1', ...
                    token_dim, ...
                    event_projection.params.embed_dim, ...
                    event_projection.params.init_layers, ...
                    false, ...
                    0.0, ...
                    batch_size);
                layer.event_projection.q_mask_ptr = layer.q_mask_ptr;
                layer.event_projection.mask_ptr = layer.mask_ptr;
            elseif strcmp(event_projection.name,'TransformerBlock')
                layer.event_projection = get_block(event_projection.name, ...
                    'Block_1',token_dim,event_projection.params.latent_blocks,event_projection.params.dropout, ...
                    event_projection.params.att_dropout,event_projection.params.heads,event_projection.params.cross_heads);
                layer.event_projection.q_mask_ptr = layer.q_mask_ptr;
                layer.event_projection.mask_ptr = layer.mask_ptr;
            end
            
            % Events preprocessing -> Linear transformation on tokens
            layer.preproc_events = preproc_events;
            preproc_events.params.embed_dim = embed_dim;
            tmp_dim = split(event_projection.params.init_layers(end),'_');
            tmp_dim = str2num(tmp_dim{2}) + pos_emb_dim;
            if  strcmp(preproc_events.name,'MLP')
                layer.preproc_block_events = get_block(preproc_events.name,'Block_2' ...
                    ,tmp_dim,preproc_events.params.embed_dim, ...
                    preproc_events.params.init_layers,false,0.0,batch_size);
                layer.preproc_block_events.q_mask_ptr = layer.q_mask_ptr;
                layer.preproc_block_events.mask_ptr = layer.mask_ptr;
            elseif strcmp(preproc_events.name,'TransformerBlock')
                layer.preproc_block_events = get_block(preproc_events.name, ...
                    'Block_2',tmp_dim,preproc_events.params.latent_blocks,preproc_events.params.dropout, ...
                    preproc_events.params.att_dropout,preproc_events.params.heads,preproc_events.params.cross_heads);
                layer.preproc_block_events.q_mask_ptr = layer.q_mask_ptr;
                layer.preproc_block_events.mask_ptr = layer.mask_ptr;
            end
    

            % Transforms events at each level
            proc_events.params.opt_dim = embed_dim;
            proc_events.params.ipt_dim = embed_dim;
            proc_events.params.embed_dim = embed_dim;
            if  strcmp(proc_events.name,'MLP')
                layer.proc_event_blocks = get_block(proc_events.name,'Block_3', ...
                    proc_events.params.ipt_dim,proc_events.params.embed_dim, ...
                    proc_events.params.init_layers,proc_events.params.add_x_input, ...
                    proc_events.params.dropout,batch_size);
                layer.proc_event_blocks.q_mask_ptr = layer.q_mask_ptr;
                layer.proc_event_blocks.mask_ptr = layer.mask_ptr;
            elseif strcmp(proc_events.name,'TransformerBlock')
                layer.proc_event_blocks = get_block(proc_events.name, ...
                    'Block_3',tmp_dim,proc_events.params.latent_blocks,proc_events.params.dropout, ...
                    proc_events.params.att_dropout,proc_events.params.heads,proc_events.params.cross_heads);
                layer.proc_event_blocks.q_mask_ptr = layer.q_mask_ptr;
                layer.proc_event_blocks.mask_ptr = layer.mask_ptr;
            end
            % self.proc_event_blocks = nn.ModuleList([ get_block(proc_events.name, proc_events['params']) ])
    
            % Transforms latent embeddings at each level
            proc_memory.params.opt_dim = embed_dim;
            proc_memory.params.embed_dim = embed_dim;
            if  strcmp(proc_memory.name,'MLP')
                layer.proc_memory_blocks = get_block(proc_memory.name,'Block_4', ...
                    proc_memory.params.ipt_dim,proc_memory.params.embed_dim, ...
                    proc_memory.params.init_layers,proc_memory.params.add_x_input, ...
                    proc_memory.params.dropout,batch_size);
            elseif strcmp(proc_memory.name,'TransformerBlock')
                layer.proc_memory_blocks = get_block(proc_memory.name, ...
                    'Block_4',proc_memory.params.opt_dim, ...
                    proc_memory.params.latent_blocks,proc_memory.params.dropout, ...
                    proc_memory.params.att_dropout,proc_memory.params.heads, ...
                    proc_memory.params.cross_heads,layer.num_latent_vectors);
                layer.proc_memory_blocks.q_mask_ptr = layer.q_mask_ptr;
                layer.proc_memory_blocks.mask_ptr = layer.mask_ptr;
            end
            proc_embs.opt_dim = embed_dim;
            proc_embs.params.embed_dim = embed_dim;
            layer.proc_embs_block = LatentEmbsCompressor('Block_5',layer.num_latent_vectors, proc_embs.opt_dim, proc_embs.clf_mode, proc_embs.embs_norm, varargin);

            layer.net = addLayers(layer.net,functionLayer(@(X1,X2)layer.reformat_datas(X1,X2), NumInputs=2,NumOutputs=5, Name="reform", InputNames=["kv","pixels"], OutputNames =["kv","pixels","kv_s","batch_size","num_time_steps"], Acceleratable=1));
            layer.net = addLayers(layer.net, functionLayer(@(X1,X2)layer.pre_projetion(X1,X2), NumInputs=2,NumOutputs=3, Name="pre_proj", InputNames=["kv", "kv_s"], OutputNames =["kv","sample_mask","sample_mask_time"], Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net,layer.event_projection.net);
            layer.net = addLayers(layer.net,functionLayer(@(X1,X2,X3)layer.pos_encoder(X1,X2,X3), NumInputs=3,NumOutputs=2, Name="pos_enc", InputNames=["kv","pixels","kv_s"], OutputNames =["kv","pos_embs"], Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net,layer.preproc_block_events.net);
            layer.net = addLayers(layer.net,functionLayer(@(X1,X2)layer.init_latent(X1,X2), NumInputs=2,NumOutputs=3, Name="init_latent", InputNames=["kv","batch_size"], OutputNames =["kv", "inp_q", "latent_vector"], Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net, functionLayer(@(X1,X2,X3,X4,X5,X6)layer.start_of_loop(X1,X2,X3,X4,X5,X6), NumInputs=6,NumOutputs=3, Name="start_point", InputNames=["kv","pos_embs","kv_s","sample_mask","sample_mask_time","num_time_steps"], OutputNames =["inp_kv","pos_embs_t", "mask_time_t"], Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net,layer.proc_event_blocks.net);
            layer.net = addLayers(layer.net,functionLayer(@(X1,X2)layer.reshape_datas(X1,X2), NumInputs=2,NumOutputs=1, Name="reshape", InputNames=["inp_kv","pos_embs_t"], OutputNames ="inp_kv", Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net,layer.proc_memory_blocks.net);
            layer.net = addLayers(layer.net, functionLayer(@(X1,X2,X3)layer.end_of_loop(X1,X2,X3), NumInputs=3,NumOutputs=2, Name="end_point", InputNames=["inp_q","latent_vector","mask_time_t"], OutputNames =["inp_q","latent_vector"], Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net,functionLayer(@(X1,X2)layer.create_embs(X1,X2), NumInputs=2,NumOutputs=1, Name="create_embs", InputNames=["inp_q","latent_vector"], OutputNames ="embs", Formattable = 1, Acceleratable=1));
            layer.net = addLayers(layer.net,layer.proc_embs_block.net);

            layer.net = connectLayers(layer.net,"reform/kv","pre_proj/kv");
            layer.net = connectLayers(layer.net,"reform/kv_s","pre_proj/kv_s");
            layer.net = connectLayers(layer.net,"reform/kv_s","pos_enc/kv_s");
            layer.net = connectLayers(layer.net,"reform/kv_s","start_point/kv_s");
            layer.net = connectLayers(layer.net,"reform/batch_size","init_latent/batch_size");
            layer.net = connectLayers(layer.net,"reform/num_time_steps","start_point/num_time_steps");
            layer.net = connectLayers(layer.net,"pre_proj/sample_mask","start_point/sample_mask");
            layer.net = connectLayers(layer.net,"pre_proj/sample_mask_time","start_point/sample_mask_time");
            layer.net = connectLayers(layer.net,"pre_proj/kv",strcat(layer.event_projection.net.Name,'/',layer.event_projection.net.InputNames{1}));
            layer.net = connectLayers(layer.net, strcat(layer.event_projection.net.Name,'/',layer.event_projection.net.OutputNames{1}),"pos_enc/kv");
            layer.net = connectLayers(layer.net,"pos_enc/kv",strcat(layer.preproc_block_events.net.Name,'/',layer.preproc_block_events.net.InputNames{1}));
            layer.net = connectLayers(layer.net,strcat(layer.preproc_block_events.net.Name,'/',layer.preproc_block_events.net.OutputNames{1}),"init_latent/kv");
            layer.net = connectLayers(layer.net,"init_latent/kv","start_point/kv");
            layer.net = connectLayers(layer.net,"pos_enc/pos_embs","start_point/pos_embs");
            layer.net = connectLayers(layer.net,"start_point/inp_kv",strcat(layer.proc_event_blocks.net.Name,'/',layer.proc_event_blocks.net.InputNames{1}));
            layer.net = connectLayers(layer.net, strcat(layer.proc_event_blocks.net.Name,'/',layer.proc_event_blocks.net.OutputNames{1}),"reshape/inp_kv");
            layer.net = connectLayers(layer.net,"start_point/pos_embs_t","reshape/pos_embs_t");
            layer.net = connectLayers(layer.net,"reshape/inp_kv",strcat(layer.proc_memory_blocks.net.Name,'/',layer.proc_memory_blocks.net.InputNames{1}));
            layer.net = connectLayers(layer.net,"init_latent/inp_q",strcat(layer.proc_memory_blocks.net.Name,'/',layer.proc_memory_blocks.net.InputNames{2}));
            layer.net = connectLayers(layer.net,"start_point/mask_time_t","end_point/mask_time_t");
            layer.net = connectLayers(layer.net, strcat(layer.proc_memory_blocks.net.Name,'/',layer.proc_memory_blocks.net.OutputNames{1}),"end_point/inp_q");
            layer.net = connectLayers(layer.net, "init_latent/latent_vector","end_point/latent_vector");
            layer.net = connectLayers(layer.net,"end_point/latent_vector","create_embs/latent_vector");
            layer.net = connectLayers(layer.net,"end_point/inp_q", "create_embs/inp_q");
            layer.net = connectLayers(layer.net, "create_embs/embs",strcat(layer.proc_embs_block.net.Name,'/',layer.proc_embs_block.net.InputNames{1}));
            
            layer.net = networkLayer(layer.net,Name=layer.Name);
        end
        function [Y1,Y2,Y3,Y4,Y5,layer] = reformat_datas(layer, kv, pixels)  % pols, pixels
            pixels = floor(pixels/layer.downsample_pos_enc); 
            kv_s = size(kv);
            batch_size = kv_s(3);
            num_time_steps = kv_s(1);
            Y1 = kv;
            Y2 = pixels;
            Y3 = dlarray(zeros(kv_s));
            Y4 = dlarray(zeros(batch_size,1));
            Y5 = dlarray(zeros(num_time_steps,1));
        end
        function layer = chg_kv(layer,kv_s)
        end
        function [Y1,Y2,Y3] = pre_projetion(layer,X, kv)
            kv_s = size(kv);
            cal_x = reshape(X,kv_s(1),kv_s(3),kv_s(4),kv_s(2));
            samples_mask = sum(cal_x,length(kv_s)) == 0; 
            samples_mask_time = sum(sum(cal_x,length(kv_s)),length(kv_s)-1) == 0;
            Y1 = X;
            Y2 = dlarray(samples_mask,"TSB");
            if size(samples_mask_time,1) ~= 0
                Y3 = dlarray(samples_mask_time,"UU");
            else
                Y3 = dlarray(zeros(samples_mask_time),"UU");
            end
        end
        function [Y1,Y2] = pos_encoder(layer,kv,pixels, kv_t)
            kv_s = size(kv_t);
            kv = reshape(kv,[kv_s(1), kv_s(3), kv_s(4),layer.emb]);
            p_s = size(pixels);
            pixels = reshape(pixels,p_s(1),p_s(3),p_s(4),p_s(2));
            pos_embs = [];
            if isempty(layer.pos_encoding) == false
                px_s = size(pixels);
                pe_s = size(layer.pos_encoding);
                x_coord = pixels(:,:,:,1)+1;
                y_coord = pixels(:,:,:,2)+1;
                for i = 1:px_s(1)
                    for j = 1:px_s(2)
                        for k = 1:px_s(3)
                            x_coord = pixels(i,j,k,1)+1;
                            y_coord = pixels(i,j,k,2)+1;
                            pos_embs(i, j, k, :) = layer.pos_encoding(y_coord, x_coord, :);
                        end
                    end
                end
                kv_s2 = size(kv);
                kv = cat(length(kv_s2),kv, pos_embs);
            else
                pos_embs = [];
            end
            Y1 = dlarray(kv,"TSBC");
            Y2 = dlarray(pos_embs,"TSBC");
        end
        function [Y1,Y2,Y3] = init_latent(layer,kv,batch_size)
            kv_s2 = size(kv);
            if length(kv_s2)>=3
                kv = reshape(kv, kv_s2(3), kv_s2(2), 1, kv_s2(1)); 
            elseif length(kv_s2)==2
                kv = reshape(kv, 1, 1, 1, kv_s2(2)); 
            end
            % Initial latent vectors
            b_s = size(batch_size);
            mv_s = size(layer.memory_vertical);
            latent_vectors = reshape(layer.memory_vertical,[mv_s(1:1), 1, mv_s(2:end)]);
            lv_s = size(latent_vectors);
            latent_vectors = repmat(latent_vectors,[1, b_s(1), 1]);             % (num_latent_vectors, batch_size, embed_dim)
            % Initialize inp_q
            inp_q = latent_vectors;
            Y1 = dlarray(kv,"TBSC");
            Y2 = dlarray(inp_q,"TBC");
            Y3 = dlarray(latent_vectors,"TBC");
            
        end
        function [Y1, Y2, Y3, layer] = start_of_loop(layer,kv, pos_embs, kv_t, samples_mask, samples_mask_time, num_time_steps)
            kv_s = size(kv_t);
            spl_s = size(samples_mask);
            spl_m = reshape(samples_mask,spl_s(1),spl_s(3),spl_s(2));
            smt = size(samples_mask_time);
                inp_kv = kv(1,:,:,:);
                msk = double(spl_m(1,:,:,:)); 
                layer = layer.change_msk_ptr(msk);
                mask_time_t = smt(1,:,:,:);
                if isempty(pos_embs) == false
                    em_s = size(pos_embs);
                    pos_embs_t = reshape(pos_embs(1,:,:,:),em_s(3), [], em_s(2));
                else
                  pos_embs_t = [];  
                end 
            inp_kv = reshape(inp_kv,1,[],kv_s(3),kv_s(4));
            Y1 = dlarray(inp_kv,"TCSB");
            Y2 = dlarray(pos_embs_t,"BSC");
            Y3 = dlarray(mask_time_t,"BC");
        end
        function layer = change_msk_ptr(layer,msk)
                layer.mask_ptr = libpointer('mxArray',msk);  
        end
        function Y = reshape_datas(layer,inp_kv, pos_embs_t)
                inp_s = size(inp_kv);
                if layer.block_batch_s == inp_s(1)
                    inp_kv = reshape(inp_kv,inp_s(3),inp_s(1),inp_s(2));
                else
                    inp_kv = reshape(inp_kv,inp_s(1),inp_s(3),inp_s(2));
                end
            Y = dlarray(inp_kv,"TBC");
        end

        function [Y1,Y2] = end_of_loop(layer, inp_q, latent_vectors, mask_time_t)
            inq_s = size(inp_q);
                inp_q(:, mask_time_t+1) = latent_vectors(:, mask_time_t+1);
                % Update latent_vectors
                latent_vectors = inp_q + latent_vectors;
                Y1 = inp_q;
                Y2 = latent_vectors;
        end
        function Y = create_embs(layer, inp_q, latent_vectors)
            if layer.return_last_q
                embs = inp_q;
            else 
                embs = latent_vectors;
            end
            e_s = size(embs);
            Y = dlarray(reshape(embs,e_s(3),e_s(2),e_s(1)),"TBC");
        end
        function Y = predict(layer,X1, X2)
            Y = predict(layer,X1,X2);   
        end
    end
end


